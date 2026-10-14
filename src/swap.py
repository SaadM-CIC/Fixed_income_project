import numpy as np
from .curve import discount_factor


def swap_schedule(maturity, freq=1):
    """Dates de paiement (en années) : 1/freq, 2/freq, ..., maturity."""
    n = int(round(maturity * freq))
    return np.arange(1, n + 1) / freq


def annuity(pay_times, maturities, zeros, freq=1):
    """A = somme des tau * DF(T_i) : valeur actuelle de 1 payé à chaque date."""
    tau = 1 / freq
    return np.sum(tau * discount_factor(pay_times, maturities, zeros))


def par_swap_rate(pay_times, maturities, zeros, freq=1):
    """Taux fixe qui rend la valeur du swap nulle : (1 - DF(T_n)) / A."""
    df_n = discount_factor(pay_times[-1], maturities, zeros)
    return (1 - df_n) / annuity(pay_times, maturities, zeros, freq)


def swap_legs(notional, fixed_rate, pay_times, maturities, zeros, freq=1):
    """Valeur actuelle des deux jambes (jambe variable obtenue via les forwards)."""
    tau = 1 / freq
    df_end = discount_factor(pay_times, maturities, zeros)
    df_start = discount_factor(pay_times - tau, maturities, zeros)   # DF(0) = 1
    fwd = (df_start / df_end - 1) / tau                              # forward simple par période
    fixed_leg = notional * fixed_rate * tau * np.sum(df_end)
    float_leg = notional * np.sum(fwd * tau * df_end)
    return fixed_leg, float_leg


def swap_value(notional, fixed_rate, pay_times, maturities, zeros, freq=1, payer=True):
    """Valeur du swap : payeur = variable - fixe, receveur = l'opposé."""
    fixed_leg, float_leg = swap_legs(notional, fixed_rate, pay_times, maturities, zeros, freq)
    v = float_leg - fixed_leg
    return v if payer else -v


def swap_dv01(notional, fixed_rate, pay_times, maturities, zeros, freq=1, payer=True, bump=1e-4):
    """P&L pour +1 pb parallèle sur les taux zéro (signé : > 0 pour le payeur)."""
    up = swap_value(notional, fixed_rate, pay_times, maturities, zeros + bump, freq, payer)
    dn = swap_value(notional, fixed_rate, pay_times, maturities, zeros - bump, freq, payer)
    return (up - dn) / 2


def swap_key_rate_dv01(notional, fixed_rate, pay_times, maturities, zeros,
                       freq=1, payer=True, bump=1e-4):
    """P&L pour +1 pb sur un seul pilier de la courbe, pilier par pilier."""
    res = []
    for k in range(len(zeros)):
        z_up, z_dn = zeros.copy(), zeros.copy()
        z_up[k] += bump
        z_dn[k] -= bump
        v_up = swap_value(notional, fixed_rate, pay_times, maturities, z_up, freq, payer)
        v_dn = swap_value(notional, fixed_rate, pay_times, maturities, z_dn, freq, payer)
        res.append((v_up - v_dn) / 2)
    return np.array(res)