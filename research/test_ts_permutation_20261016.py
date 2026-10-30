
from pathlib import Path as _ProjectPath
PROJECT_ROOT = next(p for p in _ProjectPath(__file__).resolve().parents if (p / ".sofc-project").is_file())
from pathlib import Path
import numpy as np,json,datetime
O=Path(str(PROJECT_ROOT / 'outputs/microgrid_sys/results/trace_scale_mesh_20261016'))
(O/'tests').mkdir(parents=True,exist_ok=True)
rng=np.random.default_rng(20261016);per=np.array([0,3,2,1]);M=np.array([.002016,.018015,.031998,.028014]);worst=0.
for _ in range(1000):
 w=rng.dirichlet(np.ones(4));D=rng.uniform(1e-5,2e-4,(4,4));D=(D+D.T)/2;np.fill_diagonal(D,0);S=rng.normal(size=4);wp=w[per];Mp=M[per];Dp=D[np.ix_(per,per)];Sp=S[per]
 assert np.array_equal(wp[per],w) and np.array_equal(Dp[np.ix_(per,per)],D)
 assert np.allclose(np.sum(w/M),np.sum(wp/Mp),atol=1e-13,rtol=1e-15)
 assert np.isclose(S.sum(),Sp.sum(),atol=1e-14)
 x=(w/M)/(w/M).sum();xp=(wp/Mp)/(wp/Mp).sum();assert np.array_equal(xp,x[per]) or np.allclose(xp,x[per],atol=1e-16)
 # Pairwise MS driving-force operator is covariant under simultaneous relabeling.
 vel=rng.normal(size=4);op=np.array([sum(x[i]*x[j]*(vel[i]-vel[j])/D[i,j] for j in range(4) if j!=i) for i in range(4)]);vp=vel[per];opp=np.array([sum(xp[i]*xp[j]*(vp[i]-vp[j])/Dp[i,j] for j in range(4) if j!=i) for i in range(4)]);worst=max(worst,float(np.max(abs(opp-op[per]))));assert np.allclose(opp,op[per],atol=1e-10,rtol=1e-12)
assert np.array_equal(np.array([.95,.05,0,0])[per],np.array([.95,0,0,.05]))
(O/'tests/species_permutation.json').write_text(json.dumps(dict(passed=True,cases=1000,seed=20261016,maximum_MS_operator_absolute_difference=worst,scope='Exact permutation of four-species fractions, masses, diffusion pairs, sources and inlet values. Does not demonstrate numerical PDE convergence.'),indent=2),encoding='utf-8');print('PERMUTATION_TEST_PASS',worst)
