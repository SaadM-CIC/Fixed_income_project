import numpy as np
from src.bonds import cashflows, bond_price
from src.curve import bootstrap_par_curve
from src.risk import (macaulay_duration, modified_duration, convexity, dv01,
                      price_approx, dv01_bump, dv01_curve, key_rate_dv01)


def test_zero_coupon_duration_equals_maturity():
    t, cf = cashflows(100, 0.0, 7)
    assert abs(macaulay_duration(t, cf, 0.04) - 7) < 1e-10
    assert abs(modified_duration(t, cf, 0.04) - 7 / 1.04) < 1e-10


def test_coupon_bond_duration_below_maturity():
    t, cf = cashflows(100, 0.05, 10)
    assert 0 < macaulay_duration(t, cf, 0.05) < 10


def test_duration_increases_with_maturity_decreases_with_coupon():
    d = lambda c, T: macaulay_duration(*cashflows(100, c, T), 0.05)
    assert d(0.05, 5) < d(0.05, 10) < d(0.05, 20)
    assert d(0.08, 10) < d(0.03, 10)


def test_dv01_analytic_vs_bump():
    t, cf = cashflows(100, 0.05, 10)
    assert np.isclose(dv01(t, cf, 0.04), dv01_bump(t, cf, 0.04), rtol=1e-5)


def test_convexity_vs_numerical_second_derivative():
    t, cf = cashflows(100, 0.05, 10)
    y, h = 0.04, 1e-3
    p = bond_price(t, cf, y)
    num = (bond_price(t, cf, y + h) + bond_price(t, cf, y - h) - 2 * p) / h**2 / p
    assert np.isclose(convexity(t, cf, y), num, rtol=1e-3)


def test_semiannual():
    t, cf = cashflows(100, 0.06, 5, freq=2)
    assert np.isclose(dv01(t, cf, 0.05, 2), dv01_bump(t, cf, 0.05, 2), rtol=1e-5)


def test_second_order_better_than_first():
    t, cf = cashflows(100, 0.05, 10)
    exact = bond_price(t, cf, 0.05 + 0.02)
    e1 = abs(price_approx(t, cf, 0.05, 0.02, order=1) - exact)
    e2 = abs(price_approx(t, cf, 0.05, 0.02, order=2) - exact)
    assert e2 < e1


def test_key_rates_sum_to_parallel_dv01():
    mats = np.arange(1, 11)
    par = np.array([0.025, 0.027, 0.029, 0.031, 0.0325, 0.034, 0.035, 0.036, 0.0365, 0.037])
    _, zeros = bootstrap_par_curve(mats, par)
    t, cf = cashflows(100, 0.05, 7)
    assert np.isclose(key_rate_dv01(t, cf, mats, zeros).sum(),
                      dv01_curve(t, cf, mats, zeros), rtol=1e-4)