import sys
sys.path.append("src")
import numpy as np
from bonds import cashflows, bond_price
from curve import bootstrap_par_curve, discount_factor, price_with_curve, forward_rate

mats = np.arange(1, 11)
par = np.array([0.025, 0.027, 0.029, 0.031, 0.0325, 0.034, 0.035, 0.036, 0.0365, 0.037])


def test_df_decreasing_and_below_one():
    df, _ = bootstrap_par_curve(mats, par)
    assert np.all(df < 1) and np.all(np.diff(df) < 0)


def test_par_bonds_reprice_at_100():
    """Une obligation au pair, valorisée avec la courbe bootstrappée, vaut 100."""
    _, zeros = bootstrap_par_curve(mats, par)
    for n, c in zip(mats, par):
        t, cf = cashflows(100, c, n)
        assert abs(price_with_curve(t, cf, mats, zeros) - 100) < 1e-8


def test_flat_curve():
    """Courbe plate : zéro-coupon = taux au pair, et prix identique à l'étape 1."""
    flat = np.full(10, 0.04)
    _, zeros = bootstrap_par_curve(mats, flat)
    assert np.allclose(zeros, 0.04)
    t, cf = cashflows(100, 0.05, 7)
    assert abs(price_with_curve(t, cf, mats, zeros) - bond_price(t, cf, 0.04)) < 1e-8


def test_forward_consistency():
    """(1+z2)^2 = (1+z1)(1+f(1,2))."""
    _, zeros = bootstrap_par_curve(mats, par)
    f12 = forward_rate(1, 2, mats, zeros)
    assert abs((1 + zeros[1]) ** 2 - (1 + zeros[0]) * (1 + f12)) < 1e-10