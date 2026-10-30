"""Physics/operator/sampling contracts on the isolated corrected trainer."""

from pathlib import Path as _ProjectPath
PROJECT_ROOT = next(p for p in _ProjectPath(__file__).resolve().parents if (p / ".sofc-project").is_file())
from pathlib import Path
import sys,json,copy,shutil
import numpy as np,pandas as pd,torch,yaml
ROOT=Path(str(PROJECT_ROOT));EXP=ROOT/'outputs/pinn_sofc/experiments/electrochem_corrected_20261015';OUT=ROOT/'outputs/microgrid_sys/results/electrochem_corrected_20261015';sys.path.insert(0,str(EXP/'trainer'))
(EXP / "tests").mkdir(parents=True, exist_ok=True)
(OUT / "tests").mkdir(parents=True, exist_ok=True)
from physics.pde import PDEResiduals,I,REF,k_eff
from physics import params as P
from physics.domains import DOMAIN_ORDER
from physics.thermal_operators import danckwerts_inflow,newton_wall,outlet_normal_stress
from physics.gradient_budget import GradientBudget
from physics.waveforms import v_wave
from data.dataset import SOFCDataModule
torch.set_num_threads(4);results=[]
def passed(name,**details):results.append({'test':name,'status':'pass',**details});print('PASS',name,flush=True)
# Density must be independent of gauge pressure and its analytic derivatives match autograd,
# including the stabilizing clamps outside the physically admissible region.
o=torch.zeros((4,12),dtype=torch.float64);o[:,0]=torch.tensor([1100.,1073.,.5,1000.]);o[:,3]=torch.tensor([0.,100.,-500.,101325.]);o[:,4:7]=torch.tensor([[.0,.0,.21],[.95,.05,0.],[.3,.2,.1],[2.,-1.,0.]])
o.requires_grad_(True);p=object.__new__(PDEResiduals);rho,deriv=p._rho(o);g=torch.autograd.grad(rho.sum(),o)[0]
for col,analytic in zip([3,0,4,5,6],deriv):assert torch.allclose(g[:,col],analytic,rtol=1e-10,atol=1e-10),(col,g[:,col],analytic)
changed=o.detach().clone();changed[:,3]+=50000;assert torch.equal(p._rho(changed)[0],rho.detach())
passed('constant_reference_density_and_clamp_derivatives',rho_kg_m3=rho.detach().tolist())
# Nontrivial exact profiles check the signs and units of both inflow and wall operators.
Tin=1100.;rho0=.3;cp=1500.;u=2.;k=.1;A=3.;Tx=A*rho0*cp*u/k
assert abs(danckwerts_inflow(Tin+A,Tx,rho0,cp,u,Tin,k))<1e-10
for normal in [-1,1]:
 T=Tin+20.;Ty=-normal*5*20/k;assert abs(newton_wall(T,Ty*normal,None,Tin,k,5))<1e-10
ux,vy=30.,-10.;pressure=2*P.MU_G*ux
assert abs(outlet_normal_stress(pressure,ux,vy,P.MU_G))<1e-14
passed('manufactured_Danckwerts_Robin_and_gauge_traction')
cfg=yaml.safe_load((EXP/'configs/seed42_pinn.yaml').read_text());dm=SOFCDataModule(cfg,torch.device('cpu'))
ym=torch.tensor([dm.y_stats[v]['mean'] for v in dm.target_vars]);ys=torch.tensor([dm.y_stats[v]['std'] for v in dm.target_vars]);xm=torch.tensor([dm.x_stats[c]['mean'] for c in dm.cont_ids]);xs=torch.tensor([dm.x_stats[c]['std'] for c in dm.cont_ids])
class Analytic(torch.nn.Module):
 def forward(self,X):
  physical=X[:,:10]*xs+xm;x,y,t=physical[:,0],physical[:,1],physical[:,2];z=x*0+y*0+t*0
  T=physical[:,7]+5000*x*x+100000*y*y+.02*t
  columns=[T,z+2.,z+.1,z+10.,z+.1,z+.1,z+.1,z+.7,z+.1,z+100000.,z+1e6,z+1e6]
  return (torch.stack(columns,1)-ym)/ys
pde=PDEResiduals(Analytic(),dm,torch.device('cpu'),balance=True,eq_weights={},hotspot_weights={})
assert set(pde._case_ids)==set(cfg['data']['steady_train']) and set(pde._tcase_ids)==set(cfg['data']['transient_train'])
assert not set(pde._tcase_ids)&{'T3','T7','IT01','IT02','IT03'}
passed('physical_case_scope_training_only',steady=len(pde._case_ids),transient=len(pde._tcase_ids))
times=torch.tensor([0.,99.999,100.,150.,199.999,200.,300.,400.,500.,600.,700.,800.,900.,1000.,1100.,1200.])
for code in range(5):
 c={'Vc':torch.full_like(times,1.),'amp':torch.full_like(times,.2),'stepT':torch.full_like(times,100.),'wtype':torch.full_like(times,code)}
 assert torch.equal(pde._voltage(c,times),v_wave(times,1.,.2,code,100.)),code
passed('vectorized_voltage_matches_all_five_COMSOL_waveforms')
case={'Vc':1.,'amp':.15,'stepT':100.,'wtype':0,'Tin':1073.,'la':8.,'fu':.6}
for dom in DOMAIN_ORDER:
 x=torch.linspace(.01,.09,8,requires_grad=True);y0,y1=P.Y_LAYERS[dom];y=torch.linspace(y0+.1*(y1-y0),y1-.1*(y1-y0),8,requires_grad=True);t=torch.full((8,),300.,requires_grad=True)
 transient=pde.residuals(dom,x,y,t,case,True)['T'];stationary=pde.residuals(dom,x,y,t,case,False)['T'];out=pde._forward(DOMAIN_ORDER.index(dom),x,y,t,case)
 if dom!='electrolyte':assert torch.allclose(pde.residuals(dom,x,y,t,case,True)['mass'],torch.zeros_like(x),atol=1e-7)
 rho=pde._rho(out)[0];solid=dom not in ['air_channel','fuel_channel'];storage=(P.RHO_S*P.CP_S*torch.ones_like(rho) if solid else rho*P.CP_G)*.02
 advection=0. if solid else rho*P.CP_G*(2*10000*x+.1*200000*y)
 expected=(storage+advection-k_eff(dom)*(10000+200000)-100000.)/REF['T']
 assert torch.allclose(transient,expected,rtol=5e-5,atol=2e-5),(dom,transient,expected)
 assert torch.allclose(transient-stationary,storage/REF['T'],rtol=5e-4,atol=2e-5),dom
passed('manufactured_all_domain_energy_and_zero_stationary_storage')
rng=np.random.default_rng(20261015);x,y,t,case,cids=pde._samples('transient',192,rng,'air_channel')
tb=np.searchsorted([200,400,800],t,side='right');xb=np.searchsorted([.025,.05,.075],x,side='right')
assert len(set(zip(cids,tb,xb)))==12*4*4;assert int((t>=800).sum())==48
passed('joint_case_time_axial_cells_guaranteed',covered_cells=192,late_points=48)
pool=dm.pools['transient_train'];idx=pool.sample_indices_strat(dm.domain_quota(2048),np.random.default_rng(42),sample_weight=pool.sample_weight)
assert set(pool.sampling_labels[idx])==set(pool.sampling_labels)
fraction=float((pool.t[idx]>=800).mean());assert .23<fraction<.27
passed('supervised_time_axial_stratification',late_fraction=fraction,actual_points=len(idx))
audit=json.loads((EXP/'initial_reference/audit.json').read_text());r3=next(x for x in audit if x['transient_case']=='R3');assert r3['steady_source']=='S3';bank=pd.read_parquet(cfg['data']['initial_temperature_file']);b=bank[bank.case_id=='R3'];assert b['T'].max()-b['T'].min()>15.
assert max(x['max_T_difference_to_original_t0_K'] for x in audit)<.0001
passed('nonuniform_training_steady_initial_field_preserved',R3_T_range_K=[float(b['T'].min()),float(b['T'].max())])
model=torch.nn.Linear(2,1,bias=False).double();model.weight.data[:]=torch.tensor([[1.,1.]])
data_loss=((model.weight-torch.tensor([[0.,1.5]]))**2).sum();data_loss.backward();original=model.weight.grad.clone();gb=GradientBudget(model,1.,{'a':.5,'b':.5})
def action(a,b):
 v=a*model.weight[0,0]+b*model.weight[0,1];v.backward();return float(v.detach())
gb.run('a',lambda:action(100000.,100000.));gb.run('b',lambda:action(-100000.,100000.));log=gb.finish()
expected=original+sum(gb.physics);assert torch.allclose(model.weight.grad,expected);assert log['physics_to_data_ratio']<=1+1e-12
assert torch.equal(model.weight.detach(),torch.tensor([[1.,1.]],dtype=torch.float64))
passed('gradient_budget_keeps_data_and_caps_total_physics',**log)
(OUT/'equation_and_sampling_tests.json').write_text(json.dumps({'status':'pass','tests':results},ensure_ascii=False,indent=2),encoding='utf-8');shutil.copy2(__file__,EXP/'tests'/Path(__file__).name)
