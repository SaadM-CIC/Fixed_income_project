import numpy as np
from .bonds import bond_price
from .curve import price_with_curve


def macaulay_duration(times, flows, y, freq=1):
    """Maturité moyenne des flux, pondérée par leur valeur actualisée."""
    pv = flows / (1 + y / freq) ** (times * freq)
    return np.sum(times * pv) / np.sum(pv)


def modified_duration(times, flows, y, freq=1):
    """Sensibilité relative du prix : -(1/P) dP/dy."""
    return macaulay_duration(times, flows, y, freq) / (1 + y / freq)


def convexity(times, flows, y, freq=1):
    """(1/P) d²P/dy², en années²."""
    p = bond_price(times, flows, y, freq)
    terms = flows * times * (times + 1 / freq) * (1 + y / freq) ** (-(times * freq + 2))
    return np.sum(terms) / p


def dv01(times, flows, y, freq=1):
    """Variation de prix pour 1 point de base (analytique)."""
    return modified_duration(times, flows, y, freq) * bond_price(times, flows, y, freq) * 1e-4


def price_approx(times, flows, y0, dy, freq=1, order=2):
    """Prix approché après un choc dy : ordre 1 (duration) ou 2 (duration + convexité)."""
    p0 = bond_price(times, flows, y0, freq)
    rel = -modified_duration(times, flows, y0, freq) * dy
    if order == 2:
        rel += 0.5 * convexity(times, flows, y0, freq) * dy ** 2
    return p0 * (1 + rel)


def dv01_bump(times, flows, y, freq=1, bump=1e-4):
    """DV01 par différence finie centrée (bump and reprice)."""
    return (bond_price(times, flows, y - bump, freq) - bond_price(times, flows, y + bump, freq)) / 2


def dv01_curve(times, flows, maturities, zeros, bump=1e-4):
    """DV01 sur la courbe zéro-coupon : translation parallèle de tous les taux zéro."""
    up = price_with_curve(times, flows, maturities, zeros + bump)
    dn = price_with_curve(times, flows, maturities, zeros - bump)
    return (dn - up) / 2


def key_rate_dv01(times, flows, maturities, zeros, bump=1e-4):
    """DV01 par pilier : on bouge un seul taux zéro à la fois."""
    res = []
    for k in range(len(zeros)):
        z_up, z_dn = zeros.copy(), zeros.copy()
        z_up[k] += bump
        z_dn[k] -= bump
        res.append((price_with_curve(times, flows, maturities, z_dn)
                    - price_with_curve(times, flows, maturities, z_up)) / 2)
    return np.array(res)