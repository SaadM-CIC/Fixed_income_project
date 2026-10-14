import sys
sys.path.append("src")      # à lancer depuis la racine du projet : pytest

from bonds import cashflows, bond_price, ytm


def test_par_bond():
    """Si y = coupon, le prix vaut le nominal."""
    t, cf = cashflows(100, 0.05, 5)
    assert abs(bond_price(t, cf, 0.05) - 100) < 1e-8


def test_zero_coupon():
    """Zéro-coupon : P = N / (1+y)^T."""
    t, cf = cashflows(100, 0.0, 5)
    assert abs(bond_price(t, cf, 0.04) - 100 / 1.04**5) < 1e-8


def test_price_decreasing_in_yield():
    t, cf = cashflows(100, 0.05, 10)
    assert bond_price(t, cf, 0.03) > bond_price(t, cf, 0.05) > bond_price(t, cf, 0.07)


def test_premium_and_discount():
    """y < coupon : au-dessus du pair. y > coupon : en dessous."""
    t, cf = cashflows(100, 0.05, 5)
    assert bond_price(t, cf, 0.03) > 100
    assert bond_price(t, cf, 0.07) < 100


def test_ytm_roundtrip():
    """On prend un prix à y = 3 %, le YTM doit retrouver 3 %."""
    t, cf = cashflows(100, 0.05, 5)
    p = bond_price(t, cf, 0.03)
    assert abs(ytm(t, cf, p) - 0.03) < 1e-8


def test_semiannual():
    """Au pair aussi avec des coupons semestriels."""
    t, cf = cashflows(100, 0.06, 4, freq=2)
    assert abs(bond_price(t, cf, 0.06, freq=2) - 100) < 1e-8