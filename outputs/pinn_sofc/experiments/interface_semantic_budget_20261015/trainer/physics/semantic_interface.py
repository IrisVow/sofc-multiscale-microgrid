"""Semantic interface partitions with the original loss coefficients and forward order."""
import math
import torch

TERMS=('temperature','heat_flux','ionic_potential','ionic_flux','electronic_insulation','normal_mass_flux')

def validate_shares(shares):
    if set(shares)!=set(TERMS):raise ValueError('Every semantic interface term requires an explicit share')
    if any(not math.isfinite(float(v)) or v<0 for v in shares.values()) or not math.isclose(sum(shares.values()),1.,abs_tol=1e-12,rel_tol=0):
        raise ValueError('Semantic interface shares must be finite, nonnegative and sum to1')

def collect(pde,rng,n_per_iface,scale=1.):
    from .domains import DOMAIN_ID,INTERFACES
    from .pde import I,DT_REF,PHIL_REF,REF,k_eff,sigma_l_eff,sigma_s_eff,ELECTRODES
    params=list(pde.net.parameters())
    gradients={name:[torch.zeros_like(p) for p in params] for name in TERMS}
    losses={name:0. for name in TERMS}
    specs=[(mode,a,b,y) for mode in ['steady','transient'] for a,b,y in INTERFACES]
    for mode,a,b,yval in specs:
        x,y,t,case,_=pde._samples(mode,n_per_iface,rng,a,'horizontal',yval);xt,yt,tt=pde._tensors(x,y,t)
        oa=pde._forward(DOMAIN_ID[a],xt,yt,tt,case);ob=pde._forward(DOMAIN_ID[b],xt,yt,tt,case)
        _,ga=pde._d1(oa,xt,yt,[I[v] for v in ['T','phil','phis']]);_,gb=pde._d1(ob,xt,yt,[I[v] for v in ['T','phil','phis']])
        items=[('temperature',(oa[:,0]-ob[:,0])/DT_REF),('heat_flux',(k_eff(a)*ga[0]-k_eff(b)*gb[0])/REF['flux_T'])]
        if (a in ELECTRODES and b=='electrolyte') or (a=='electrolyte' and b in ELECTRODES):
            items.extend([('ionic_potential',(oa[:,8]-ob[:,8])/PHIL_REF),
                ('ionic_flux',(sigma_l_eff(a,oa[:,I['T']])*ga[8]-sigma_l_eff(b,ob[:,I['T']])*gb[8])/REF['flux_phil'])])
            items.append(('electronic_insulation',sigma_s_eff(a)*ga[7]/REF['flux_phis'] if a in ELECTRODES else sigma_s_eff(b)*gb[7]/REF['flux_phis']))
        else:items.append(('normal_mass_flux',(pde._rho(oa)[0]*oa[:,2]-pde._rho(ob)[0]*ob[:,2])/REF['mass_flux']))
        for index,(name,value) in enumerate(items):
            term=torch.mean(torch.sqrt(value*value+1e-8))/(len(items)*len(specs))
            losses[name]+=float(term.detach())
            grads=torch.autograd.grad(scale*term,params,retain_graph=index<len(items)-1,allow_unused=True)
            for acc,g in zip(gradients[name],grads):
                if g is not None:acc.add_(g.detach())
    return {'unscaled_mean_loss':sum(losses.values()),'term_mean_losses':losses,'gradients':gradients}

def vector_norm(gradients):
    return math.sqrt(sum(float(torch.sum(g.double()**2)) for g in gradients))

def apply(budget,name,action,shares):
    validate_shares(shares)
    for p in budget.params:p.grad=None
    payload=action()
    if set(payload['gradients'])!=set(TERMS) or set(payload['term_mean_losses'])!=set(TERMS):raise ValueError('Incomplete semantic interface result')
    combined_raw=[torch.zeros_like(p) for p in budget.params]
    combined_used=[torch.zeros_like(p) for p in budget.params]
    cap=budget.ratio*budget.reference_norm*budget.budgets[name];logs=[]
    if not math.isfinite(cap) or cap<0:raise FloatingPointError('Nonfinite semantic interface cap')
    for term in TERMS:
        gs=payload['gradients'][term]
        if len(gs)!=len(budget.params) or any(g.shape!=p.shape or g.dtype!=p.dtype or g.device!=p.device for g,p in zip(gs,budget.params)):
            raise ValueError('Semantic interface gradient layout changed')
        raw=vector_norm(gs);value=float(payload['term_mean_losses'][term])
        if not math.isfinite(raw) or not math.isfinite(value):raise FloatingPointError('Nonfinite semantic interface loss or gradient')
        termcap=cap*shares[term];factor=min(1.,termcap/max(raw,1e-30))
        for whole,used,g in zip(combined_raw,combined_used,gs):whole.add_(g);used.add_(g,alpha=factor)
        logs.append({'component':name+'/'+term,'share':shares[term],'raw_grad_norm':raw,'cap':termcap,'scale':factor,
                     'used_grad_norm':raw*factor,'unscaled_mean_loss_contribution':value})
    used=vector_norm(combined_used)
    if used>cap*(1+2e-5)+1e-10:raise AssertionError('Semantic interface family exceeded original total cap')
    for acc,g in zip(budget.physics,combined_used):acc.add_(g)
    value=float(payload['unscaled_mean_loss'])
    if not math.isfinite(value):raise FloatingPointError('Nonfinite semantic interface family loss')
    if not math.isclose(value,sum(payload['term_mean_losses'].values()),rel_tol=1e-6,abs_tol=1e-8):raise ValueError('Semantic interface loss sum changed')
    budget.logs.append({'component':name,'returned_unscaled_mean_loss':value,'raw_grad_norm':vector_norm(combined_raw),
        'cap':cap,'scale':None,'used_grad_norm':used,'allocation':'semantic_equal','subcomponents':logs})
    return value
