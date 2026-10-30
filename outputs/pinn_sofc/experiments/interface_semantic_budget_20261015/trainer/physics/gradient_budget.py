"""Stateless same-step gradient budgets; data and physics share one optimizer step."""
import math
import torch

class GradientBudget:
    def __init__(self,model,ratio,budgets):
        self.params=list(model.parameters());self.ratio=float(ratio);self.budgets=dict(budgets)
        if self.ratio<0 or any(v<0 for v in self.budgets.values()) or sum(self.budgets.values())>1+1e-12:raise ValueError('Invalid gradient budget')
        self.data=[torch.zeros_like(p) if p.grad is None else p.grad.detach().clone() for p in self.params]
        self.reference_norm=math.sqrt(sum(float(torch.sum(g.double()**2)) for g in self.data))
        self.physics=[torch.zeros_like(p) for p in self.params];self.logs=[]

    def run(self,name,action):
        for p in self.params:p.grad=None
        value=action()
        norm=math.sqrt(sum(float(torch.sum(p.grad.detach().double()**2)) for p in self.params if p.grad is not None))
        if not math.isfinite(norm):raise FloatingPointError('Nonfinite physics gradient')
        cap=self.ratio*self.reference_norm*self.budgets[name];factor=min(1.,cap/max(norm,1e-30))
        for acc,p in zip(self.physics,self.params):
            if p.grad is not None:acc.add_(p.grad.detach(),alpha=factor)
        self.logs.append({'component':name,'returned_unscaled_mean_loss':float(value),'raw_grad_norm':norm,'cap':cap,'scale':factor,'used_grad_norm':norm*factor})
        return float(value)

    def run_partitioned(self,name,action,shares):
        from .semantic_interface import apply
        return apply(self,name,action,shares)

    def finish(self):
        norm=math.sqrt(sum(float(torch.sum(g.double()**2)) for g in self.physics))
        if norm>self.ratio*self.reference_norm*(1+2e-5)+1e-10:raise AssertionError('Physics gradient budget exceeded')
        for p,d,g in zip(self.params,self.data,self.physics):p.grad=d+g
        return {'data_grad_norm':self.reference_norm,'combined_physics_grad_norm':norm,'physics_to_data_ratio':norm/max(self.reference_norm,1e-30),'components':self.logs}
