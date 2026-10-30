"""Run released entry points with a module-local cwd and the active interpreter."""
from pathlib import Path
import argparse
import os
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
M = ROOT / 'outputs/microgrid_sys'
P = ROOT / 'outputs/pinn_sofc'
CHECKS = ['check_components', 'check_ems', 'check_load', 'check_sizing',
          'check_scaling', 'check_lut', 'check_surrogate', 'check_inference', 'check_capacity']

def launch(cwd, arguments):
    env = dict(os.environ, PYTHONUTF8='1', PYTHONIOENCODING='utf-8')
    subprocess.run([sys.executable, *map(str, arguments)], cwd=cwd, env=env, check=True)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['check', 'dispatch', 'capacity', 'sensitivity', 'train-smoke', 'evaluate', 'sampling'])
    args, extra = parser.parse_known_args()
    if args.command == 'check':
        if extra:
            parser.error('check does not accept extra arguments')
        for name in CHECKS:
            print(f'Running {name}', flush=True)
            launch(M, [f'scripts/{name}.py'])
        launch(ROOT, ['research/test_ts_permutation_20261016.py'])
    elif args.command == 'train-smoke':
        launch(P, ['train.py', '--config', 'config_release_smoke.yaml', *extra])
    elif args.command == 'evaluate':
        launch(P, ['evaluate.py', '--run_dir', 'runs/stage3_pde_v20', '--config', 'config_v20.yaml', '--ckpt', 'final.pt', '--pools', 'steady_val', 'steady_test', 'transient_test', *extra])
    elif args.command == 'sampling':
        launch(ROOT, ['tools/reproduce_sampling.py', *extra])
    else:
        script = {'dispatch': 'run_day_sim', 'capacity': 'run_capacity_opt', 'sensitivity': 'sensitivity_tornado'}[args.command]
        launch(M, [f'scripts/{script}.py', *extra])

if __name__ == '__main__':
    main()
