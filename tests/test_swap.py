import numpy as np
from src.bonds import cashflows
from src.curve import bootstrap_par_curve, discount_factor, price_with_curve
from src.swap import (swap_schedule, annuity, par_swap_rate, swap_legs, swap_value,
                      swap_dv01, swap_key_rate_dv01)

mats = np.arange(1, 11)
par = np.array([0.025, 0.027, 0.029, 0.031, 0.0325, 0.034, 0.035, 0.036, 0.0365, 0.037])
_, zeros = bootstrap_par_curve(mats, par)
N = 1_000_000


def test_par_swap_rate_equals_par_yield():
    """Courbe unique : taux swap au pair = taux au pair bootstrappé."""
    for n, c in zip(mats, par):
        assert abs(par_swap_rate(swap_schedule(n), mats, zeros) - c) < 1e-10


def test_value_zero_at_par_rate():
    pay = swap_schedule(5)
    s = par_swap_rate(pay, mats, zeros)
    assert abs(swap_value(N, s, pay, mats, zeros)) < 1e-6


def test_payer_plus_receiver_is_zero():
    pay = swap_schedule(5)
    assert abs(swap_value(N, 0.035, pay, mats, zeros)
               + swap_value(N, 0.035, pay, mats, zeros, payer=False)) < 1e-9


def test_floating_leg_telescopes():
    """Jambe variable = N (1 - DF(T_n)), y compris avec des paiements semestriels."""
    for freq in [1, 2]:
        pay = swap_schedule(5, freq)
        _, float_leg = swap_legs(N, 0.03, pay, mats, zeros, freq)
        assert abs(float_leg - N * (1 - discount_factor(5, mats, zeros))) < 1e-6


def test_value_formula_N_A_S_minus_K():
    pay = swap_schedule(7)
    s, a = par_swap_rate(pay, mats, zeros), annuity(pay, mats, zeros)
    assert abs(swap_value(N, 0.04, pay, mats, zeros) - N * a * (s - 0.04)) < 1e-6


def test_swap_is_floater_minus_fixed_bond():
    """Payeur = N - obligation à coupon K (lien avec les étapes 1 et 2)."""
    pay = swap_schedule(6)
    t, cf = cashflows(100, 0.04, 6)
    bond_value = N / 100 * price_with_curve(t, cf, mats, zeros)
    assert abs(swap_value(N, 0.04, pay, mats, zeros) - (N - bond_value)) < 1e-6


def test_flat_curve():
    _, z = bootstrap_par_curve(mats, np.full(10, 0.04))
    assert abs(par_swap_rate(swap_schedule(7), mats, z) - 0.04) < 1e-10


def test_dv01_sign_and_size():
    pay = swap_schedule(5)
    k = par_swap_rate(pay, mats, zeros)
    dv = swap_dv01(N, k, pay, mats, zeros)
    approx = N * annuity(pay, mats, zeros) * 1e-4        # ~ N * A * 1 pb
    assert dv > 0 and abs(dv / approx - 1) < 0.05
    assert abs(swap_dv01(N, k, pay, mats, zeros, payer=False) + dv) < 1e-9


def test_key_rates_sum_to_parallel():
    pay = swap_schedule(7)
    kr = swap_key_rate_dv01(N, 0.04, pay, mats, zeros)
    assert np.isclose(kr.sum(), swap_dv01(N, 0.04, pay, mats, zeros), rtol=1e-4)