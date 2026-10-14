import numpy as np


def vasicek_mean(t, r0, a, b):
    """E[r_t] = b + (r0 - b) e^{-a t}."""
    return b + (r0 - b) * np.exp(-a * t)


def vasicek_var(t, a, sigma):
    """Var[r_t] = sigma^2 / (2a) (1 - e^{-2 a t})."""
    return sigma**2 / (2 * a) * (1 - np.exp(-2 * a * t))


def simulate_vasicek(r0, a, b, sigma, T, n_steps, n_paths, seed=None, scheme="exact"):
    """Trajectoires du taux court. Retourne les dates et un tableau (n_paths, n_steps+1).

    scheme = "exact" : on tire dans la loi gaussienne exacte entre deux dates
    scheme = "euler" : discrétisation d'Euler (approchée), à comparer
    """
    rng = np.random.default_rng(seed)
    dt = T / n_steps
    t = np.linspace(0, T, n_steps + 1)
    r = np.empty((n_paths, n_steps + 1))
    r[:, 0] = r0

    decay = np.exp(-a * dt)
    std_exact = sigma * np.sqrt((1 - np.exp(-2 * a * dt)) / (2 * a))

    for i in range(n_steps):
        z = rng.standard_normal(n_paths)
        if scheme == "exact":
            r[:, i + 1] = b + (r[:, i] - b) * decay + std_exact * z
        else:
            r[:, i + 1] = r[:, i] + a * (b - r[:, i]) * dt + sigma * np.sqrt(dt) * z
    return t, r


def vasicek_B(tau, a):
    return (1 - np.exp(-a * tau)) / a


def vasicek_zc_price(r, a, b, sigma, tau):
    """Prix P(t, t+tau) d'un zéro-coupon quand le taux court vaut r (r peut être un tableau)."""
    B = vasicek_B(tau, a)
    ln_A = (b - sigma**2 / (2 * a**2)) * (B - tau) - sigma**2 * B**2 / (4 * a)
    return np.exp(ln_A - B * r)


def vasicek_zero_rate(r, a, b, sigma, tau):
    """Taux zéro-coupon en composition continue."""
    return -np.log(vasicek_zc_price(r, a, b, sigma, tau)) / tau


def vasicek_zero_annual(r, a, b, sigma, tau):
    """Même taux, en composition annuelle (convention de nos étapes 1 à 4)."""
    return np.exp(vasicek_zero_rate(r, a, b, sigma, tau)) - 1


def mc_zero_price(r0, a, b, sigma, T, n_steps=250, n_paths=20000, seed=0):
    """Prix zéro-coupon par Monte Carlo : E[exp(-intégrale de r)], intégrale par trapèzes."""
    t, r = simulate_vasicek(r0, a, b, sigma, T, n_steps, n_paths, seed)
    dt = t[1] - t[0]
    integral = np.sum((r[:, :-1] + r[:, 1:]) / 2, axis=1) * dt
    disc = np.exp(-integral)
    return disc.mean(), disc.std(ddof=1) / np.sqrt(n_paths)


def mc_zc_call(r0, a, b, sigma, t_opt, T, K, n_steps=100, n_paths=50000, seed=0):
    """Call européen d'échéance t_opt, strike K, sur un zéro-coupon de maturité T.

    Le prix du zéro-coupon à t_opt est connu en formule fermée, conditionnellement à r(t_opt).
    """
    t, r = simulate_vasicek(r0, a, b, sigma, t_opt, n_steps, n_paths, seed)
    dt = t[1] - t[0]
    disc = np.exp(-np.sum((r[:, :-1] + r[:, 1:]) / 2, axis=1) * dt)
    p_T = vasicek_zc_price(r[:, -1], a, b, sigma, T - t_opt)
    values = disc * np.maximum(p_T - K, 0)
    return values.mean(), values.std(ddof=1) / np.sqrt(n_paths)


def calibrate_vasicek_curve(maturities, zero_rates_cc, sigma, a_grid):
    """Calibre (r0, a, b) sur une courbe de taux zéro continus, sigma étant fixé.

    Pour (a, sigma) fixés : R = R_inf (1 - B/tau) + r0 B/tau + sigma^2 B^2 / (4 a tau),
    donc (R_inf, r0) s'obtiennent par moindres carrés linéaires. On balaie a sur une grille.
    """
    maturities = np.asarray(maturities, dtype=float)
    best = None
    for a in a_grid:
        B = vasicek_B(maturities, a)
        X = np.column_stack([1 - B / maturities, B / maturities])
        y = zero_rates_cc - sigma**2 * B**2 / (4 * a * maturities)
        coef = np.linalg.lstsq(X, y, rcond=None)[0]
        sse = np.sum((y - X @ coef) ** 2)
        if best is None or sse < best[0]:
            best = (sse, a, coef[0], coef[1])
    sse, a, r_inf, r0 = best
    b = r_inf + sigma**2 / (2 * a**2)
    return {"a": a, "b": b, "sigma": sigma, "r0": r0,
            "rmse": np.sqrt(sse / len(maturities))}


def fit_vasicek_ols(r, dt):
    """Estimation historique (mesure P) par régression r_{t+1} = alpha + phi r_t + eps.

    phi = e^{-a dt}, alpha = b (1 - phi), Var(eps) = sigma^2 (1 - phi^2) / (2a).
    """
    x, y = r[:-1], r[1:]
    phi, alpha = np.polyfit(x, y, 1)
    a = -np.log(phi) / dt
    b = alpha / (1 - phi)
    resid = y - (alpha + phi * x)
    sigma = np.sqrt(resid.var(ddof=2) * 2 * a / (1 - phi**2))
    return a, b, sigma