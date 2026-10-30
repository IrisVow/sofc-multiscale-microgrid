
from pathlib import Path as _ProjectPath
PROJECT_ROOT = next(p for p in _ProjectPath(__file__).resolve().parents if (p / ".sofc-project").is_file())
from pathlib import Path
import sys,copy,json
import numpy as np,torch,yaml
ROOT=Path(str(PROJECT_ROOT))
EXP=ROOT/'outputs/pinn_sofc/experiments/interface_semantic_budget_20261015'
(EXP / "tests").mkdir(parents=True, exist_ok=True)
OUT=ROOT/'outputs/microgrid_sys/results/interface_semantic_budget_20261015'
(OUT / "tests").mkdir(parents=True, exist_ok=True)
PARENT=ROOT/'outputs/pinn_sofc/experiments/electrochem_corrected_20261015'
sys.path.insert(0,str(EXP/'trainer'))
from physics.gradient_budget import GradientBudget
from physics.semantic_interface import TERMS,validate_shares,vector_norm
from checkpointing import restore,SCHEMA
from train import build_model,set_seed
from data.dataset import SOFCDataModule
from physics.pde import PDEResiduals
torch.set_num_threads(4);set_seed(42);tests=[];shares={k:1/6 for k in TERMS}
def passed(name,**kw):tests.append({'test':name,'status':'pass',**kw});print('PASS',name,flush=True)
def payload(gs):return {'gradients':{n:[gs.get(n,torch.zeros(1,2,dtype=torch.float64))] for n in TERMS},'term_mean_losses':{n:0. for n in TERMS},'unscaled_mean_loss':0.}
def toy():
 model=torch.nn.Linear(2,1,bias=False).double();model.weight.grad=torch.tensor([[3.,4.]],dtype=torch.float64)
 return model,GradientBudget(model,1.,{'interface':.15,'other':.85})
model,budget=toy();budget.run_partitioned('interface',lambda:payload({'electronic_insulation':torch.tensor([[1e4,0.]],dtype=torch.float64),'ionic_flux':torch.tensor([[0.,1.]],dtype=torch.float64)}),shares)
expected=torch.tensor([[.125,.125]],dtype=torch.float64)
assert torch.allclose(budget.physics[0],expected,rtol=0,atol=1e-15)
assert all(x['used_grad_norm']<=.125+1e-15 for x in budget.logs[-1]['subcomponents'])
assert budget.finish()['physics_to_data_ratio']<=.15
passed('analytic_sixway_caps_and_original_total_cap')
model,budget=toy();model.weight.grad.zero_();budget=GradientBudget(model,1.,{'interface':.15})
budget.run_partitioned('interface',lambda:payload({'ionic_flux':torch.ones_like(model.weight)}),shares);assert torch.equal(budget.physics[0],torch.zeros_like(model.weight))
assert budget.finish()['physics_to_data_ratio']==0
passed('zero_data_gradient_gives_zero_physics_cap')
for change in ['missing','negative','nan','sum']:
 bad=dict(shares)
 if change=='missing':bad.pop(TERMS[0])
 elif change=='negative':bad[TERMS[0]]=-1.
 elif change=='nan':bad[TERMS[0]]=float('nan')
 else:bad[TERMS[0]]=.5
 try:validate_shares(bad)
 except ValueError:pass
 else:raise AssertionError(change)
passed('invalid_missing_or_nonfinite_budget_rejected')
model,budget=toy()
try:budget.run_partitioned('interface',lambda:payload({'ionic_flux':torch.full_like(model.weight,float('nan'))}),shares)
except FloatingPointError:pass
else:raise AssertionError('Nonfinite gradient accepted')
passed('nonfinite_partition_gradient_rejected')
ck=torch.load(PARENT/'runs/seed42_pinn/checkpoints/epoch_0030.pt',weights_only=False,map_location='cpu')
try:restore(ck,*([None]*7),ck['config'],ck['data_identity'],ck['code_identity'],torch.device('cuda'))
except ValueError as e:assert 'cannot fully resume' in str(e)
else:raise AssertionError('Old schema accepted as direct resume')
passed('schema3_requires_declared_warm_branch_not_direct_resume',new_schema=SCHEMA)
cfg=yaml.safe_load((EXP/'configs/semantic.yaml').read_text(encoding='utf-8'));dm=SOFCDataModule(cfg,torch.device('cpu'))
for trainmode in [False,True]:
 model=build_model(cfg,12,10).cuda();model.load_state_dict(ck['model']);model.train(trainmode)
 pde=PDEResiduals(model,dm,torch.device('cuda'),balance=True,eq_weights=cfg['pde']['eq_weights'],hotspot_weights=cfg['pde']['hotspot_weights'])
 original={k:v.clone() for k,v in model.state_dict().items()}
 state=torch.cuda.get_rng_state_all();cpu_state=torch.get_rng_state().clone()
 model.zero_grad(set_to_none=True);value=pde.interface_loss(np.random.default_rng(163),256,scale=.02)
 legacy=[torch.zeros_like(p) if p.grad is None else p.grad.detach().clone() for p in model.parameters()];after=torch.cuda.get_rng_state_all()
 torch.cuda.set_rng_state_all(state);torch.set_rng_state(cpu_state);model.zero_grad(set_to_none=True)
 parts=pde.interface_partitioned_gradients(np.random.default_rng(163),256,scale=.02)
 combined=[sum((parts['gradients'][n][i] for n in TERMS),torch.zeros_like(p)) for i,p in enumerate(model.parameters())]
 diff=vector_norm([a-b for a,b in zip(legacy,combined)])/max(vector_norm(legacy),1e-30)
 assert diff<3e-5,(trainmode,diff)
 assert abs(parts['unscaled_mean_loss']-value)<=max(1e-5,abs(value)*1e-5)
 assert all(torch.equal(x,y) for x,y in zip(after,torch.cuda.get_rng_state_all()))
 assert all(torch.equal(v,model.state_dict()[k]) for k,v in original.items())
 passed('full_interface_loss_gradient_and_forward_RNG_parity_'+('train' if trainmode else 'eval'),relative_gradient_error=diff)
 del model,pde,original,legacy,combined,parts;torch.cuda.empty_cache()
report={'status':'pass','tests':tests,'schema':SCHEMA,'source_parameters_changed':False}
(OUT/'tests/semantic_unit_and_parity_tests.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
print('SEMANTIC_TESTS_COMPLETE')
