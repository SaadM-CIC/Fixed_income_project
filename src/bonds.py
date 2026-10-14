import numpy as np


def cashflows(nominal, coupon_rate, maturity, freq=1):
    """Return coupon dates and cash flow amounts for a bond."""
    n = int(round(maturity * freq))
    times = np.arange(1, n + 1) / freq
    flows = np.full(n, nominal * coupon_rate / freq)
    flows[-1] += nominal
    return times, flows


def bond_price(times, flows, y, freq=1):
    """Discounted bond price at annual yield y."""
    return np.sum(flows / (1 + y / freq) ** (times * freq))


def ytm(times, flows, market_price, freq=1, lo=-0.05, hi=1.0, tol=1e-10):
    """Estimate yield to maturity by bisection."""
    for _ in range(200):
        mid = (lo + hi) / 2
        if bond_price(times, flows, mid, freq) > market_price:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol:
            break
    return (lo + hi) / 2


__all__ = ["cashflows", "bond_price", "ytm"]
