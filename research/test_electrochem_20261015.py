"""Independent constitutive/manufactured-solution tests, then original thermal/sampling contracts."""

from pathlib import Path as _ProjectPath
PROJECT_ROOT = next(p for p in _ProjectPath(__file__).resolve().parents if (p / ".sofc-project").is_file())
from pathlib import Path
import sys,json,subprocess,shutil
import numpy as np,pandas as pd,torch,yaml
ROOT=Path(str(PROJECT_ROOT))
EXP=ROOT/'outputs/pinn_sofc/experiments/electrochem_corrected_20261015';OUT=ROOT/'outputs/microgrid_sys/results/electrochem_corrected_20261015'
(EXP / "tests").mkdir(parents=True, exist_ok=True)
(OUT / "tests").mkdir(parents=True, exist_ok=True)
sys.path.insert(0,str(EXP/'trainer'))
from physics.electrochem import ionic_conductivity,limited_butler_volmer,reaction_current
from physics import params as P
from physics.pde import PDEResiduals,I,REF,sigma_l_eff,sigma_s_eff
from data.dataset import SOFCDataModule
torch.set_num_threads(4);tests=[]
def passed(name,**kw):tests.append({'test':name,'status':'pass',**kw});print('PASS',name,flush=True)
T=torch.tensor([1073.,1100.,1148.35294782],dtype=torch.float64,requires_grad=True);s=ionic_conductivity(T)
assert torch.allclose(s,33400*torch.exp(-10300/T),rtol=1e-14)
ds=torch.autograd.grad(s.sum(),T)[0];assert torch.allclose(ds,s*10300/T**2,rtol=1e-12)
assert sigma_s_eff('air_electrode')==10000 and sigma_s_eff('fuel_electrode')==35000
for dom in ['air_electrode','fuel_electrode','electrolyte']:assert torch.equal(sigma_l_eff(dom,T),s)
passed('Arrhenius_values_derivative_and_NoCorr',sigma_S_m=s.detach().tolist())
eta=torch.tensor([-1.,-.1,0.,.1,1.],dtype=torch.float64,requires_grad=True);temp=torch.full_like(eta,1100.);lim=torch.full_like(eta,2000.)
for i0 in [50.,500.]:
 value=limited_butler_volmer(eta,temp,i0,lim);z=P.F_CONST*eta/(P.R_CONST*temp);plain=i0*(torch.exp(.5*z)-torch.exp(-.5*z));expected=plain/(1+plain.abs()/lim)
 assert torch.allclose(value,expected,rtol=1e-13,atol=1e-12)
 assert abs(float(value[2].detach()))==0
 assert torch.isclose(torch.autograd.grad(value.sum(),eta,retain_graph=True)[0][2],torch.tensor(i0*P.F_CONST/(P.R_CONST*1100.),dtype=torch.float64),rtol=1e-12)
assert torch.autograd.gradcheck(lambda e:limited_butler_volmer(e,temp,50.,lim),(eta,),eps=1e-6,atol=1e-4,rtol=2e-4)
passed('documented_limited_BV_equivalence_equilibrium_and_gradient')
eta_extreme=torch.tensor([-1e6,-10.,0.,10.,1e6],requires_grad=True);lim_extreme=torch.tensor([2.,0.,0.,5.,10.],requires_grad=True)
v=limited_butler_volmer(eta_extreme,torch.full_like(eta_extreme,1100.),50.,lim_extreme)
grads=torch.autograd.grad(v.sum(),(eta_extreme,lim_extreme));assert torch.isfinite(v).all() and all(torch.isfinite(g).all() for g in grads)
assert (v.abs()<=lim_extreme+1e-6).all() and v[1]==0 and v[2]==0
passed('finite_extreme_overpotential_zero_reactant_and_bounds',currents_A_m2=v.detach().tolist())
checks=pd.read_csv(OUT/'tests/COMSOL_training_kinetics_formula_check.csv');assert checks.relative_RMS_error.max()<1e-5
passed('reference_training_exports_reproduced_without_fit',max_relative_RMS_error=float(checks.relative_RMS_error.max()),cases=['S3','S5'])
# Eeq stays at 1 V at every temperature, rather than incorporating reaction-heat entropy.
for temp in [1000.,1073.,1200.]:
 t=torch.tensor([temp],dtype=torch.float64);rho=torch.tensor([.3],dtype=torch.float64)
 ca=reaction_current('air_electrode',torch.ones_like(t),torch.zeros_like(t),t,rho,t*0,t*0+.21)
 an=reaction_current('fuel_electrode',t*0,t*0,t,rho,t*0+.95,t*0)
 assert torch.equal(ca,t*0) and torch.equal(an,t*0)
passed('fixed_equilibrium_voltage_for_both_electrodes')
cfg=yaml.safe_load((EXP/'configs/seed42_pinn.yaml').read_text());dm=SOFCDataModule(cfg,torch.device('cpu'))
xm=torch.tensor([dm.x_stats[c]['mean'] for c in dm.cont_ids],dtype=torch.float64);xs=torch.tensor([dm.x_stats[c]['std'] for c in dm.cont_ids],dtype=torch.float64)
ym=torch.tensor([dm.y_stats[c]['mean'] for c in dm.target_vars],dtype=torch.float64);ys=torch.tensor([dm.y_stats[c]['std'] for c in dm.target_vars],dtype=torch.float64)
class Manufactured(torch.nn.Module):
 def forward(self,X):
  physical=X[:,:10]*xs+xm;x=physical[:,0];zero=x*0;dom=X[:,10:].argmax(1)
  q=torch.where(dom==1,zero-10.,torch.where(dom==3,zero+10.,zero));sigma_s=torch.where(dom==1,zero+10000.,zero+35000.)
  # sigma(T(x)) = 2*(1+10*x); integrate phi'_l=-(q*x+1)/sigma exactly.
  T=10300/torch.log(33400/(2*(1+10*x)))
  phil=-q/(2*10)*x-(1-q/10)/(2*10)*torch.log1p(10*x)
  phis=q/(2*sigma_s)*x*x
  columns=[T,zero,zero,zero,zero+.95,zero+.05,zero+.21,phis,phil,zero,zero+10.,zero-10.]
  return (torch.stack(columns,1)-ym)/ys
pde=PDEResiduals(Manufactured(),dm,torch.device('cpu'),balance=True,eq_weights={},hotspot_weights={})
for attr,value in [('xm',xm),('xs',xs),('ym',ym),('ys',ys)]:setattr(pde,attr,value)
case=dict(Vc=.9,amp=0.,stepT=0.,wtype=0.,Tin=1073.,la=8.,fu=.6);errors=[]
for dom in ['air_electrode','fuel_electrode','electrolyte']:
 x=torch.linspace(.001,.099,12,dtype=torch.float64,requires_grad=True);lo,hi=P.Y_LAYERS[dom];y=torch.full_like(x,(lo+hi)/2,requires_grad=True);t=torch.zeros_like(x,requires_grad=True)
 r=pde.residuals(dom,x,y,t,case,transient=False)
 ionic=float((r['phil']*REF['phil']).abs().max().detach());assert ionic<1e-8,(dom,ionic)
 if dom!='electrolyte':
  electronic=float((r['phis']*REF['phis']).abs().max().detach());assert electronic<1e-8,(dom,electronic)
 errors.append({'domain':dom,'ionic_unscaled_maxabs_A_m3':ionic})
passed('manufactured_full_charge_sources_and_variable_conductivity_divergence',errors=errors)
# Flux continuity must use each side's local temperature, including when Ta != Tb.
ta=torch.tensor([1073.,1148.],dtype=torch.float64);tb=ta.flip(0);flux=torch.tensor([3.,-2.],dtype=torch.float64)
ga=flux/sigma_l_eff('air_electrode',ta);gb=flux/sigma_l_eff('electrolyte',tb)
assert torch.allclose(sigma_l_eff('air_electrode',ta)*ga-sigma_l_eff('electrolyte',tb)*gb,torch.zeros_like(ta),atol=1e-14)
passed('local_temperature_on_each_side_of_ionic_flux_interface')
(OUT/'tests/electrochem_tests.json').write_text(json.dumps({'status':'pass','tests':tests},ensure_ascii=False,indent=2),encoding='utf-8')
shutil.copy2(__file__,EXP/'tests'/Path(__file__).name)
print('ELECTROCHEM_TESTS_DONE',flush=True)
