import sys
sys.path.append("src")
import numpy as np
from src.vasicek import (vasicek_mean, vasicek_var, simulate_vasicek, vasicek_B, vasicek_zc_price,
                     vasicek_zero_rate, mc_zero_price, mc_zc_call, calibrate_vasicek_curve,
                     fit_vasicek_ols)

r0, a, b, sigma = 0.03, 0.5, 0.04, 0.01


def test_moments_of_simulation():
    t, r = simulate_vasicek(r0, a, b, sigma, 5, 50, 40000, seed=0)
    rT = r[:, -1]
    se = rT.std() / np.sqrt(len(rT))
    assert abs(rT.mean() - vasicek_mean(5, r0, a, b)) < 4 * se
    assert abs(rT.var() / vasicek_var(5, a, sigma) - 1) < 0.04


def test_stationary_variance():
    assert abs(vasicek_var(100, a, sigma) - sigma**2 / (2 * a)) < 1e-12


def test_euler_close_to_exact_for_small_dt():
    t, r = simulate_vasicek(r0, a, b, sigma, 3, 300, 40000, seed=1, scheme="euler")
    assert abs(r[:, -1].var() / vasicek_var(3, a, sigma) - 1) < 0.04


def test_zc_price_decreasing_and_below_one():
    taus = np.arange(1, 31)
    p = vasicek_zc_price(r0, a, b, sigma, taus)
    assert p[0] < 1 and np.all(np.diff(p) < 0)


def test_zc_price_deterministic_limit():
    """sigma -> 0 : P = exp(-integrale de la moyenne deterministe)."""
    tau = 5.0
    det = np.exp(-(b * tau + (r0 - b) * vasicek_B(tau, a)))
    assert abs(vasicek_zc_price(r0, a, b, 1e-8, tau) - det) < 1e-9


def test_long_yield_limit():
    r_inf = b - sigma**2 / (2 * a**2)
    assert abs(vasicek_zero_rate(r0, a, b, sigma, 500.0) - r_inf) < 1e-4


def test_sensitivity_is_minus_B():
    tau, h = 7.0, 1e-6
    d = (np.log(vasicek_zc_price(r0 + h, a, b, sigma, tau))
         - np.log(vasicek_zc_price(r0 - h, a, b, sigma, tau))) / (2 * h)
    assert abs(d + vasicek_B(tau, a)) < 1e-6


def test_monte_carlo_matches_closed_form():
    for T in [1, 5]:
        price, se = mc_zero_price(r0, a, b, sigma, T, n_steps=50 * T, n_paths=20000, seed=0)
        assert abs(price - vasicek_zc_price(r0, a, b, sigma, T)) < 4 * se + 1e-4


def test_calibration_recovers_parameters():
    mats = np.arange(1, 11, dtype=float)
    curve = vasicek_zero_rate(0.02, 0.3, 0.05, 0.012, mats)
    res = calibrate_vasicek_curve(mats, curve, 0.012, np.arange(0.05, 2.0, 0.01))
    assert abs(res["a"] - 0.3) < 0.02
    assert abs(res["r0"] - 0.02) < 1e-4 and abs(res["b"] - 0.05) < 1e-4


def test_ols_recovers_parameters():
    t, r = simulate_vasicek(0.04, 0.5, 0.04, 0.01, 5000 / 12, 5000, 1, seed=1)
    a_hat, b_hat, s_hat = fit_vasicek_ols(r[0], 1 / 12)
    assert abs(a_hat - 0.5) < 0.2 and abs(b_hat - 0.04) < 0.005 and abs(s_hat / 0.01 - 1) < 0.1


def test_call_on_zc_bounds_and_vol_monotonicity():
    P5, P1 = vasicek_zc_price(r0, a, b, sigma, 5), vasicek_zc_price(r0, a, b, sigma, 1)
    K = 0.8 * P5 / P1
    price, se = mc_zc_call(r0, a, b, sigma, 1, 5, K)
    assert price >= (P5 - K * P1) - 4 * se                       # borne basse : valeur intrinsèque forward
    low, _ = mc_zc_call(r0, a, b, 0.005, 1, 5, P5 / P1)
    high, _ = mc_zc_call(r0, a, b, 0.02, 1, 5, P5 / P1)
    assert high > low