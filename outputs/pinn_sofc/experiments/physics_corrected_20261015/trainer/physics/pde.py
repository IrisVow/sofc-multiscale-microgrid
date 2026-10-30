"""COMSOL-aligned thermal/pressure conventions and train-only stratified policies.

Gas density uses the saved COMSOL constant thermodynamic pressure; p is gauge.
Electrodes and electrolyte are heat-transfer solids. Initial temperature targets
come from verified training steady solutions. No test/reference case is sampled.
Other transport/electrochemical closures remain the documented v20 approximations.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import torch
from . import params as P
from .domains import DOMAIN_ORDER,DOMAIN_ID,INTERFACES
from .equations_core import PDEResiduals as Core,I,REF,DT_REF,U_REF,W_REF,PHIS_REF,PHIL_REF,CHANNELS,ELECTRODES,k_eff,sigma_l_eff,sigma_s_eff
from .thermal_operators import danckwerts_inflow,newton_wall,outlet_normal_stress

def smooth_mean(values):
    return torch.stack([torch.mean(torch.sqrt(v*v+1e-8)) for v in values]).mean()

class PDEResiduals(Core):
    def _grad(self,out,x):
        if not out.requires_grad:return x*0.
        result=torch.autograd.grad(out,x,torch.ones_like(out),create_graph=True,retain_graph=True,allow_unused=True)[0]
        return x*0. if result is None else result

    def _rho(self,out):
        T=out[:,I['T']];Ts=T.clamp_min(1.)
        w1,w2,w3=(out[:,I[v]] for v in ['w1','w2','w3']);w4=1-w1-w2-w3
        inv=w1/P.M_H2+w2/P.M_H2O+w3/P.M_O2+w4/P.M_N2
        raw=1/inv.clamp_min(1e-9);mm=raw.clamp(min=P.M_H2,max=P.M_O2)
        active=((inv>=1e-9)&(raw>=P.M_H2)&(raw<=P.M_O2)).to(out.dtype)
        rho0=P.P_REF*mm/(P.R_CONST*Ts);rho=rho0.clamp(min=1e-6,max=5.)
        ar=((rho0>=1e-6)&(rho0<=5.)).to(out.dtype);factor=P.P_REF/(P.R_CONST*Ts)*ar
        deriv=[-mm**2*(1/m-1/P.M_N2)*active*factor for m in [P.M_H2,P.M_H2O,P.M_O2]]
        return rho,(torch.zeros_like(T),-rho0/Ts*(T>=1.).to(out.dtype)*ar,*deriv)

    def _case_params(self):
        cfg=self.dm.cfg;self.cfg=cfg;self.pc=cfg['pde']
        if self.pc.get('physical_case_scope')!='training_only':raise ValueError('Corrected trainer requires explicit training-only physical cases')
        self._cases={};self._tcases={};self._inlets={};self.sampling_audit=[]
        for cid in cfg['data']['steady_train']:
            frame=self.dm.df_steady[self.dm.df_steady.case_id==cid];r=frame.iloc[0]
            self._cases[cid]=[float(r[k]) for k in ['V_applied','T_in_K','lambda_air','FU']]
            self._store_inlets(cid,frame)
        for cid in cfg['data']['transient_train']:
            frame=self.dm.df_transient[self.dm.df_transient.case_id==cid];r=frame.iloc[0]
            self._tcases[cid]=(float(r.V_cell),float(r.T_in_K),float(r.lambda_air),float(r.FU),float(r.amp),float(r.stepT),int(r.wave_type),float(frame.t_s.max()))
            self._store_inlets(cid,frame[frame.t_s==0])
        self._case_ids=list(self._cases);self._case_list=list(self._cases.values());self._tcase_ids=list(self._tcases);self._tcase_list=list(self._tcases.values())
        bank=pd.read_parquet(cfg['data']['initial_temperature_file'])
        if set(bank.case_id)!=set(self._tcase_ids):raise ValueError('Initial reference case set differs from training cases')
        if not set(bank.steady_source).issubset(self._cases):raise ValueError('Initial reference uses non-training steady truth')
        self.initial_bank={(cid,dom):g.reset_index(drop=True) for (cid,dom),g in bank.groupby(['case_id','domain'])}

    def _store_inlets(self,cid,frame):
        for dom in ['air_channel','fuel_channel']:
            g=frame[(frame.domain==dom)&np.isclose(frame.x,0.,atol=1e-12)].groupby('y')[['u','v','w1','w2','w3']].mean().sort_index()
            if g.empty:raise ValueError(f'No inlet reference for training case {cid}/{dom}')
            self._inlets[(cid,dom)]=g

    def _case_dict(self,cids,mode):
        if mode=='steady':
            a=np.array([self._cases[c] for c in cids]);params={'Vc':a[:,0],'Tin':a[:,1],'la':a[:,2],'fu':a[:,3],'amp':np.zeros(len(cids)),'stepT':np.zeros(len(cids)),'wtype':np.zeros(len(cids))}
        else:
            a=np.array([self._tcases[c] for c in cids]);params={'Vc':a[:,0],'Tin':a[:,1],'la':a[:,2],'fu':a[:,3],'amp':a[:,4],'stepT':a[:,5],'wtype':a[:,6]}
        return {k:torch.as_tensor(v,dtype=torch.float32,device=self.device) for k,v in params.items()}

    def _samples(self,mode,n,rng,dom,axis='volume',fixed=None):
        ids=self._case_ids if mode=='steady' else self._tcase_ids
        edges=self.pc['time_strata'];bins=1 if mode=='steady' else len(edges)-1
        cells=[(cid,b,q) for cid in ids for b in range(bins) for q in range(4)]
        if n<len(cells):raise ValueError(f'{n} points cannot cover {len(cells)} case/time/space cells')
        indices=np.tile(np.arange(len(cells)),n//len(cells));rem=n%len(cells)
        if rem:indices=np.concatenate([indices,rng.permutation(len(cells))[:rem]])
        rng.shuffle(indices);cids=[cells[i][0] for i in indices];bb=np.array([cells[i][1] for i in indices]);qq=np.array([cells[i][2] for i in indices])
        if mode=='steady':tt=np.zeros(n)
        else:tt=rng.uniform(np.array(edges)[bb],np.array(edges)[bb+1])
        free=(qq+rng.random(n))/4.;y0,y1=P.Y_LAYERS[dom]
        x=.1*free;y=rng.uniform(y0,y1,n)
        if axis in ['inlet','outlet']:
            x[:]=0. if axis=='inlet' else .1;y=y0+(y1-y0)*free
        elif axis=='horizontal':y[:]=float(fixed)
        if mode=='transient':
            self.sampling_audit.append({'kind':axis,'domain':dom,'n':n,'cells_covered':len(set(indices.tolist())),'expected_cells':len(cells),'late_points':int((tt>=800).sum()),'case_ids':sorted(set(cids))})
        return x,y,tt,self._case_dict(cids,mode),cids

    def _tensors(self,x,y,t):
        return tuple(torch.tensor(v,dtype=torch.float32,device=self.device,requires_grad=True) for v in [x,y,t])

    def domain_loss(self,dom,rng,n_per_domain,use_bv=True,transient=False,t_fixed=None,pts_override=None):
        if t_fixed is not None or pts_override is not None:raise ValueError('Use actual initial-field loss instead of endpoint PDE masquerading as an initial condition')
        n=int(n_per_domain[dom]) if isinstance(n_per_domain,dict) else int(n_per_domain)
        x,y,t,case,cids=self._samples('transient' if transient else 'steady',n,rng,dom)
        xt,yt,tt=self._tensors(x,y,t);res=self.residuals(dom,xt,yt,tt,case,transient=transient)
        wt=torch.tensor(self._hotspot_mask(dom,np.column_stack([x,y])),dtype=torch.float32,device=self.device)
        terms=[];weights=[]
        for name,r in res.items():
            if name=='bv' and not use_bv:continue
            mean=(torch.sqrt(r*r+1e-8)*wt).sum()/wt.sum()
            key=f'{"transient" if transient else "steady"}|{dom}|{name}'
            ema=.9*self._ema.get(key,float(mean.detach()))+.1*float(mean.detach());self._ema[key]=ema
            balance=min(1/max(ema,1e-12),1e6) if self.balance else 1.
            terms.append(balance*mean);weights.append(float(self._eq_weights.get(name,1.)))
        return sum(w*v for w,v in zip(weights,terms))/sum(weights)

    def _voltage(self,case,t):
        c=case['wtype'].long();step=(t>=case['stepT']).to(t.dtype);ramp=(t/600.).clamp(0.,1.);sine=.5+.5*torch.sin(2*torch.pi*t/600.)
        multi=torch.zeros_like(t);random=torch.zeros_like(t)
        for b,v in zip([200,400,600,800,1000],[.2,.6,1.,.4,.8]):multi=torch.where(t>=b,v,multi)
        for b,v in zip([150,300,500,700,900,1100],[.6,.2,1.,.4,.8,.6]):random=torch.where(t>=b,v,random)
        wave=torch.where(c==0,step,torch.where(c==1,ramp,torch.where(c==2,multi,torch.where(c==3,sine,random))))
        return case['Vc']-case['amp']*wave

    def interface_loss(self,rng,n_per_iface,scale=1.):
        specs=[(mode,a,b,y) for mode in ['steady','transient'] for a,b,y in INTERFACES];total=0.
        for mode,a,b,yval in specs:
            x,y,t,case,_=self._samples(mode,n_per_iface,rng,a,'horizontal',yval);xt,yt,tt=self._tensors(x,y,t)
            oa=self._forward(DOMAIN_ID[a],xt,yt,tt,case);ob=self._forward(DOMAIN_ID[b],xt,yt,tt,case)
            _,ga=self._d1(oa,xt,yt,[I[v] for v in ['T','phil','phis']]);_,gb=self._d1(ob,xt,yt,[I[v] for v in ['T','phil','phis']])
            items=[(oa[:,0]-ob[:,0])/DT_REF,(k_eff(a)*ga[0]-k_eff(b)*gb[0])/REF['flux_T']]
            if (a in ELECTRODES and b=='electrolyte') or (a=='electrolyte' and b in ELECTRODES):
                items.extend([(oa[:,8]-ob[:,8])/PHIL_REF,(sigma_l_eff(a)*ga[8]-sigma_l_eff(b)*gb[8])/REF['flux_phil']])
                items.append(sigma_s_eff(a)*ga[7]/REF['flux_phis'] if a in ELECTRODES else sigma_s_eff(b)*gb[7]/REF['flux_phis'])
            else:items.append((self._rho(oa)[0]*oa[:,2]-self._rho(ob)[0]*ob[:,2])/REF['mass_flux'])
            loss=smooth_mean(items);total+=float(loss.detach());(scale/len(specs)*loss).backward()
        return total/len(specs)

    def _inlet_values(self,cids,dom,y):
        result=np.empty((len(cids),5))
        for cid in set(cids):
            mask=np.array([c==cid for c in cids]);g=self._inlets[(cid,dom)]
            for j,col in enumerate(['u','v','w1','w2','w3']):result[mask,j]=np.interp(y[mask],g.index.to_numpy(),g[col].to_numpy())
        return torch.tensor(result,dtype=torch.float32,device=self.device)

    def boundary_loss(self,rng,n_per_edge,scale=1.,do_backward=True,include_transient=True):
        modes=['steady','transient'] if include_transient else ['steady'];specs=[]
        for mode in modes:
            for dom in ['air_channel','fuel_channel']:
                specs.extend([(mode,dom,'inlet',None),(mode,dom,'outlet',None),(mode,dom,'wall',0. if dom=='air_channel' else .00262)])
            specs.extend([(mode,'air_electrode','collector',.0008),(mode,'fuel_electrode','collector',.00182)])
            for dom in ['air_electrode','electrolyte','fuel_electrode']:specs.extend([(mode,dom,'solid_inlet',None),(mode,dom,'solid_outlet',None)])
        total=0.
        for mode,dom,kind,yval in specs:
            axis='inlet' if kind in ['inlet','solid_inlet'] else 'outlet' if kind in ['outlet','solid_outlet'] else 'horizontal'
            x,y,t,case,cids=self._samples(mode,n_per_edge,rng,dom,axis,yval);xt,yt,tt=self._tensors(x,y,t)
            o=self._forward(DOMAIN_ID[dom],xt,yt,tt,case);T=o[:,0];items=[]
            if kind=='inlet':
                target=self._inlet_values(cids,dom,y);Tx=self._grad(T,xt);rho=self._rho(o)[0]
                # Danckwerts inflow for constant Cp: k Tx = rho Cp u (T-T_upstream).
                items=[danckwerts_inflow(T,Tx,rho,P.CP_G,o[:,1],case['Tin'],P.K_G)/REF['flux_T'],(o[:,1]-target[:,0])/U_REF,(o[:,2]-target[:,1])/U_REF]
                items.extend([(o[:,j+4]-target[:,j+2])/W_REF for j in range(3)])
            elif kind=='outlet':
                ux=self._grad(o[:,1],xt);uy=self._grad(o[:,1],yt);vx=self._grad(o[:,2],xt);vy=self._grad(o[:,2],yt)
                normal_stress=outlet_normal_stress(o[:,3],ux,vy,P.MU_G)
                items=[normal_stress/200.,P.MU_G*(uy+vx)/200.,P.K_G*self._grad(T,xt)/REF['flux_T'],torch.minimum(o[:,1],torch.zeros_like(T))/U_REF]
            elif kind=='wall':
                normal=-1. if dom=='air_channel' else 1.
                items=[newton_wall(T,self._grad(T,yt)*normal,None,case['Tin'],P.K_G,P.H_CONV)/REF['flux_T'],o[:,1]/U_REF,o[:,2]/U_REF]
            elif kind=='collector':items=[(o[:,7]-(self._voltage(case,tt) if dom=='air_electrode' else 0.))/PHIS_REF]
            else:items=[P.K_S*self._grad(T,xt)/REF['flux_T']]
            loss=smooth_mean(items);total+=float(loss.detach())
            if do_backward:(scale/len(specs)*loss).backward()
        return total/len(specs)

    def _initial_loss(self,rng,n_per_domain,scale,pre=False):
        ids=[cid for cid in self._tcase_ids if not pre or (self._tcases[cid][6]==0 and self._tcases[cid][5]>0)]
        if not ids:return 0.
        total=0.
        for dom in DOMAIN_ORDER:
            n=int(n_per_domain[dom]) if isinstance(n_per_domain,dict) else int(n_per_domain)
            cells=[(cid,q) for cid in ids for q in range(4)]
            if n<len(cells):raise ValueError('Initial reference budget cannot cover all case/space cells')
            indices=np.tile(np.arange(len(cells)),n//len(cells));rem=n%len(cells)
            if rem:indices=np.concatenate([indices,rng.permutation(len(cells))[:rem]])
            pieces=[];cids=[]
            for i in indices:
                cid,q=cells[i];g=self.initial_bank[(cid,dom)];xx=g.x.to_numpy();mask=(xx>=q*.025)&(xx<=.1 if q==3 else xx<(q+1)*.025);available=np.flatnonzero(mask)
                k=int(rng.choice(available));pieces.append(g.iloc[k]);cids.append(cid)
            g=pd.DataFrame(pieces);times=np.array([rng.uniform(0.,.9*self._tcases[cid][5]) for cid in cids]) if pre else np.zeros(len(g))
            x,y,t=self._tensors(g.x.to_numpy(),g.y.to_numpy(),times);o=self._forward(DOMAIN_ID[dom],x,y,t,self._case_dict(cids,'transient'))
            target=torch.tensor(g['T'].to_numpy(),dtype=torch.float32,device=self.device);loss=(((o[:,0]-target)/DT_REF)**2).mean()
            total+=float(loss.detach());(scale/len(DOMAIN_ORDER)*loss).backward()
        return total/len(DOMAIN_ORDER)

    def ic_loss(self,rng,n_per_domain,use_bv=True,scale=1.,t_fixed=0.,n_hotspot=0):
        if t_fixed!=0:raise ValueError('No artificial equilibrium constraint at final time')
        return self._initial_loss(rng,n_per_domain,scale)

    def pre_step_loss(self,rng,n_per_domain,use_bv=True,scale=1.):
        return self._initial_loss(rng,n_per_domain,scale,pre=True)
