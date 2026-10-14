import numpy as np


def bootstrap_par_curve(maturities, par_yields):
    """Bootstrap from par yields to discount factors and zero-coupon rates."""
    maturities = np.asarray(maturities, dtype=float)
    n_max = len(par_yields)
    df = np.zeros(n_max)
    for n in range(n_max):
        c = par_yields[n]
        suma_df = df[:n].sum()
        df[n] = (1 - c * suma_df) / (1 + c)
    zeros = df ** (-1 / maturities) - 1
    return df, zeros


def interp_zero(t, maturities, zeros):
    """Linear interpolation of zero rates, flat outside the range."""
    return np.interp(t, maturities, zeros)


def discount_factor(t, maturities, zeros):
    z = interp_zero(t, maturities, zeros)
    return (1 + z) ** (-t)


def price_with_curve(times, flows, maturities, zeros):
    """Discount each cash flow with its own zero-coupon rate."""
    return np.sum(flows * discount_factor(times, maturities, zeros))


def forward_rate(t1, t2, maturities, zeros):
    """Annual forward rate from t1 to t2."""
    df1 = discount_factor(t1, maturities, zeros)
    df2 = discount_factor(t2, maturities, zeros)
    return (df1 / df2) ** (1 / (t2 - t1)) - 1

def par_on_annual_grid(known_mats, known_yields, n_max=10):
    """Interpole les taux au pair sur 1..n_max et convertit semestriel -> annuel."""
    grid = np.arange(1, n_max + 1)
    y_sa = np.interp(grid, known_mats, known_yields)
    return grid, (1 + y_sa / 2) ** 2 - 1