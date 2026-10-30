"""Dimensional boundary operators: positive outward normal, gauge traction."""
def danckwerts_inflow(T,Tx,rho,cp,u,T_upstream,k):
    return k*Tx-rho*cp*u*(T-T_upstream)

def newton_wall(T,Tnormal,rho_unused,T_external,k,h):
    return -k*Tnormal-h*(T-T_external)

def outlet_normal_stress(p,ux,vy,mu,p0=0.):
    # The saved Free and Porous Media Flow interface is explicitly Incompressible.
    return p-p0-2*mu*ux
