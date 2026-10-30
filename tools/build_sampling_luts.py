"""Build axial/full-training-mesh temperature LUTs from the frozen v20 checkpoint.

Only tmax_k changes. Other LUT fields and four operating axes remain frozen.
"""
from pathlib import Path
import argparse
import hashlib
import itertools
import json
import sys
import numpy as np
import pandas as pd
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
M = ROOT/'outputs/microgrid_sys'
P = ROOT/'outputs/pinn_sofc'
sys.path.insert(0,str(P))
sys.path.insert(0,str(M))
from train import build_model
from data.dataset import CONT_IDS_V19
from data.split import DOMAIN_ORDER
from src.coupling.lut import PinnLUT

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--sampling',choices=['axial_101','full_training_mesh'],required=True)
    ap.add_argument('--output',type=Path,required=True,help='New output .npz file; existing files are protected')
    args=ap.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    checkpoint=P/'runs/stage3_pde_v20/final.pt'
    ck=torch.load(checkpoint,map_location='cpu',weights_only=False)
    stats=json.loads((checkpoint.parent/'norm_stats.json').read_text())
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net=build_model(ck['config'],len(stats['target_vars']),10).to(device)
    net.load_state_dict(ck['model']);net.eval()
    cfg=yaml.safe_load((M/'configs/coupling.yaml').read_text(encoding='utf-8'))['pinn']
    probes=[(p['y_m'],DOMAIN_ORDER.index(p['domain'])) for p in cfg['probes']]
    if args.sampling=='axial_101':
        geo=np.array([[x,y,d] for x in np.linspace(0,.1,101) for y,d in probes])
    else:
        g=pd.read_csv(ROOT/'outputs/pinndata/samples/full_training_mesh_geometry.csv')
        assert len(g)==15567
        geo=np.column_stack([g.x,g.y,g.domain.map({n:i for i,n in enumerate(DOMAIN_ORDER)})])
    xm=np.array([stats['x'][c]['mean'] for c in CONT_IDS_V19],dtype='float32')
    xs=np.array([stats['x'][c]['std'] for c in CONT_IDS_V19],dtype='float32')
    original=PinnLUT(M/'data/pinn_lut_v20.npz')
    values=[]
    with torch.no_grad():
        for index, (v,tin,lam,fu) in enumerate(itertools.product(*(original.axes[a] for a in original.AXES))):
            physical=np.column_stack([geo[:,:2],np.tile([0,v,0,0,0,tin,lam,fu],(len(geo),1))]).astype('float32')
            inputs=np.concatenate([(physical-xm)/xs,np.eye(5,dtype='float32')[geo[:,2].astype(int)]],axis=1)
            temperature=[]
            for start in range(0,len(inputs),8192):
                pred=net(torch.from_numpy(inputs[start:start+8192]).to(device))[:,0].cpu().numpy()
                temperature.append(pred*stats['y']['T']['std']+stats['y']['T']['mean'])
            values.append(float(np.concatenate(temperature).max()))
            if index%25==0: print(f'{index+1}/297',flush=True)
    meta=dict(original.meta)
    meta.update(temperature_sampling=args.sampling,n_temperature_points=len(geo),quasisteady_only=True,not_deployed=True,
                temperature_metric='predicted_training_mesh_max' if args.sampling=='full_training_mesh' else 'predicted_axial_domain_midpoint_max',
                frozen_checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),continuous_global_maximum_guaranteed=False,
                other_fields_retained_from_legacy=True,temperature_units='K')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('xb') as stream:
        np.savez(stream,**original.axes,**{f:np.array(values).reshape(original.shape) if f=='tmax_k' else original.data[f] for f in original.FIELDS},meta=np.array(meta))
    print(args.output,flush=True)

if __name__=='__main__':
    main()
