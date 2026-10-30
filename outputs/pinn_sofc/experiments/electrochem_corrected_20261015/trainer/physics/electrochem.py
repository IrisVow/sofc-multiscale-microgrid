"""Saved COMSOL 6.3 constitutive settings; NoCorr conductivities and signed currents.

Sources: echem_ug_electrochem.06.094/.095 and the preserved DOE source model.
No constants are fitted against validation or independent cases.
"""
import torch
from . import params as P

ACTIVATION_T_K = 10300.0
DIFFUSION_LAYER_M = 0.5e-3

def ionic_conductivity(T):
    """S/m. The temperature floor only protects nonphysical network iterates."""
    return P.SIGMA_L_YSZ * torch.exp(-ACTIVATION_T_K / T.clamp_min(1.0))

def limited_butler_volmer(eta, T, i0, limiting_current):
    """i_expr/(1+abs(i_expr)/ilim), algebraically scaled to avoid overflow.

    The expression is equivalent to the documented COMSOL expression for ilim>0.
    Nonpositive predicted reactant concentration implies ilim=0 and zero current.
    """
    z = P.F_CONST * eta / (P.R_CONST * T.clamp_min(1.0))
    a, c = P.ALPHA_A * z, -P.ALPHA_C * z
    m = torch.maximum(a, c)
    difference = torch.exp(a - m) - torch.exp(c - m)
    limit = limiting_current.clamp_min(0.0)
    denominator = limit * torch.exp(-m) + i0 * difference.abs()
    return limit * i0 * difference / denominator.clamp_min(1e-30)

def reaction_current(dom, phis, phil, T, rho, w_H2, w_O2):
    """Signed reaction current, A/m2, prior to multiplication by specific area."""
    if dom == 'fuel_electrode':
        eta = phis - phil - P.EEQ_AN
        concentration = (rho * w_H2 / P.M_H2).clamp_min(0.0)
        limit = 2 * P.F_CONST * P.D_EFF_GAS * concentration / DIFFUSION_LAYER_M
        return limited_butler_volmer(eta, T, P.I0_AN, limit)
    if dom == 'air_electrode':
        # dEeq/dT is a reaction-heat setting in the source; Eeq itself is 1 V.
        eta = phis - phil - P.EEQ_CA
        concentration = (rho * w_O2 / P.M_O2).clamp_min(0.0)
        limit = 4 * P.F_CONST * P.D_EFF_GAS * concentration / DIFFUSION_LAYER_M
        return limited_butler_volmer(eta, T, P.I0_CA, limit)
    raise ValueError(f'No electrode reaction for {dom}')
