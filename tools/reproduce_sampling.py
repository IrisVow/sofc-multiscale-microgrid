"""Replay the frozen 13-run, seven-day EMS comparison using the released three LUTs."""
from pathlib import Path
import argparse
import hashlib
import json
import sys
import time
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
M = ROOT / 'outputs/microgrid_sys'
sys.path.insert(0, str(M))
from src.analysis import LoadProfile
from src.ems import Microgrid
from src.coupling import PinnCoupling
from src.coupling.lut import PinnLUT

MODES = {
    'monitor': {'constraint_mode': 'monitor'},
    'limit1173': {'constraint_mode': 'derate'},
    'limit1135_m2': {'constraint_mode': 'derate', 'tmax_limit_k': 1135., 'derate_deadband_k': 0., 'derate_floor_pu': .5, 'derate_margin_k': 2.},
    'limit1135_m15': {'constraint_mode': 'derate', 'tmax_limit_k': 1135., 'derate_deadband_k': 0., 'derate_floor_pu': .5, 'derate_margin_k': 15.},
}

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output', type=Path, default=M/'results/sampling_replay')
    args = ap.parse_args()
    out = args.output.resolve()
    if out.exists():
        raise FileExistsError('Choose a new --output directory to preserve prior results')
    tables = {name: M/'data'/file for name, file in {
        'five_probe': 'pinn_lut_v20.npz', 'axial101': 'pinn_lut_v20_axial_101.npz',
        'full_mesh': 'pinn_lut_v20_full_training_mesh.npz'}.items()}
    loaded = {k: PinnLUT(p) for k,p in tables.items()}
    first = loaded['five_probe']
    for lut in loaded.values():
        for axis in first.AXES:
            np.testing.assert_array_equal(first.axes[axis], lut.axes[axis])
        for field in first.FIELDS:
            if field != 'tmax_k':
                np.testing.assert_array_equal(first.data[field], lut.data[field])
    load, heat = LoadProfile().generate_series(days=7, dt_s=60., start='2026-07-06', time_index=False)
    out.mkdir(parents=True)
    (out/'timeseries').mkdir()
    frames, rows = {}, []
    scenarios = [('baseline', None)] + [(name+'__'+mode, {'use_pinn_table': True, 'lut_path': str(path), **params}) for name,path in tables.items() for mode,params in MODES.items()]
    for name, params in scenarios:
        started = time.perf_counter()
        coupling = PinnCoupling(params=params) if params else None
        grid = Microgrid(coupling=coupling)
        grid.run_days(7, np.asarray(load), np.asarray(heat), dt_s=60.)
        frame, summary = grid.to_dataframe(), grid.summary()
        assert len(frame) == 10080 and frame.imbalance_kw.abs().max() < 1e-6
        assert frame.soc.between(0, 1).all()
        assert np.isfinite(frame[['sofc_kw', 'genset_kw', 'bat_kw', 'soc']]).all().all()
        summary.update(run=name, soc_end=float(frame.soc.iloc[-1]), fuel_total_kwh=sum(summary['fuel_'+k+'_kwh'] for k in ['sofc','genset','boiler']))
        if coupling:
            valid = (frame.sofc_kw > 0) & frame.pinn_in_domain.astype(bool)
            summary['temperature_valid_steps'] = int(valid.sum())
            summary['temperature_unassessed_generating_steps'] = int(((frame.sofc_kw > 0) & ~valid).sum())
        frame.to_csv(out/'timeseries'/f'{name}.csv', index=False)
        frames[name] = frame
        rows.append(summary)
        print(f'{name}: {time.perf_counter()-started:.2f}s', flush=True)
    power = ['sofc_kw','genset_kw','bat_kw','soc','fuel_sofc_kw','fuel_genset_kw','fuel_boiler_kw']
    for name in tables:
        np.testing.assert_array_equal(frames[name+'__monitor'][power], frames['baseline'][power])
    summary = pd.DataFrame(rows).set_index('run')
    summary.to_csv(out/'summary.csv')
    expected = pd.read_csv(ROOT/'reference_results/ems_three_sampling/summary.csv').set_index('run')
    fields = ['energy_sofc_kwh','energy_genset_kwh','ghg_kg_co2eq','fuel_total_kwh','soc_end']
    for field in fields:
        np.testing.assert_allclose(summary.loc[expected.index,field], expected[field], rtol=1e-9, atol=1e-7)
    hashes = {p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in tables.values()}
    (out/'validation.json').write_text(json.dumps({'passed':True,'runs':13,'historical_summary_fields_checked':fields,'LUT_sha256':hashes},indent=2),encoding='utf-8')
    print('Historical dispatch/fuel/emissions/SOC summaries match.',flush=True)

if __name__ == '__main__':
    main()
