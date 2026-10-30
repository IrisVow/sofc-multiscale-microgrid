import sys,random
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import train
import physics.pde
class Data:
 def __init__(self,cfg,device):
  self.cfg=cfg;self.device=device;self.target_vars=['T','u'];self.n_out=2;self.n_cont=7
  self.x_stats={'x':{'mean':0.,'std':1.}};self.y_stats={v:{'mean':0.,'std':1.} for v in self.target_vars};self.masks={'all':[1,1]};self.pools={'steady_val':1}
  x=torch.linspace(-1,1,168).reshape(24,7);dom=torch.nn.functional.one_hot(torch.arange(24)%5,5).float()
  self.X=torch.cat([x,dom],1).to(device);self.Y=torch.stack([x[:,0]*.2,x[:,1]*.3],1).to(device);self.M=torch.ones_like(self.Y)
 def save_stats(self,path): pass
 def pool_tensors(self,pool):
  assert pool=='steady_val','Training unexpectedly read a test pool'
  return self.X,self.Y,self.M,None
 def iter_epoch(self,batch,rng):
  for i in range(3):
   idx=rng.choice(24,8,replace=False);yield self.X[idx],self.Y[idx],self.M[idx]
class PDE:
 def __init__(self,model,dm,device,**kw):self.model=model;self.dm=dm;self._ema={}
 def domain_loss(self,dom,rng,n,use_bv=True,**kw):
  idx=rng.choice(24,4);z=self.model(self.dm.X[idx]);m=(z*z).mean()+z.mean()*0
  key=dom+str(kw.get('transient',False))+str(kw.get('t_fixed',0));old=self._ema.get(key,float(m.detach()));self._ema[key]=.9*old+.1*float(m.detach())
  _=random.random();_=np.random.random()
  return m/(1+self._ema[key])
 def side(self,rng,scale,key):
  loss=self.domain_loss(key,rng,4);(loss*scale).backward();return float(loss.detach())
 def interface_loss(self,rng,n,scale=1):return self.side(rng,scale,'iface')
 def boundary_loss(self,rng,n,scale=1):return self.side(rng,scale,'bc')
 def ic_loss(self,rng,n,scale=1,**kw):return self.side(rng,scale,'ic'+str(kw.get('t_fixed',0)))
 def pre_step_loss(self,rng,n,scale=1,**kw):return self.side(rng,scale,'pre')
train.SOFCDataModule=Data;physics.pde.PDEResiduals=PDE
train.main()
