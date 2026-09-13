from gwpopulation.utils import xp
from gwpopulation.models.mass import truncnorm
# baseline chi_eff, chi_p model Skew_G*G


def _erf(x):
    if "jax" in xp.__name__:
        from jax.scipy.special import erf
    else:
        from scipy.special import erf
    return erf(x)


def _trapz(y, x):
    return xp.trapz(y, x)


def _gaussian_chi_p(chi_p, mu_chi_p, sigma_chi_p):
    return truncnorm(
        chi_p,
        mu=mu_chi_p,
        sigma=sigma_chi_p,
        low=0.0,
        high=1.0,
    )


def _skewnormal_raw(chi_eff, mu_chi_eff, sigma_chi_eff, eta_chi_eff):
    """
    taken from https://www.jstor.org/stable/4615982
    A Class of Distributions Which Includes the Normal Ones
    A. Azzalini
    """
    sigma = xp.maximum(sigma_chi_eff, 1e-6)
    t = (chi_eff - mu_chi_eff) / sigma

    normal = xp.exp(-0.5 * t**2) / (xp.sqrt(2.0 * xp.pi) * sigma)
    skew_factor = 1.0 + _erf(eta_chi_eff * t / xp.sqrt(2.0))

    return normal * skew_factor


def _truncated_skewnormal_chi_eff(
    chi_eff,
    mu_chi_eff,
    sigma_chi_eff,
    eta_chi_eff,
    low=-1.0,
    high=1.0,
    n_norm=1000,
):
    """
    skewnormal 
    """
    grid = xp.linspace(low, high, n_norm)

    raw_grid = _skewnormal_raw(
        grid,
        mu_chi_eff=mu_chi_eff,
        sigma_chi_eff=sigma_chi_eff,
        eta_chi_eff=eta_chi_eff,
    )

    norm = _trapz(raw_grid, grid)

    raw = _skewnormal_raw(
        chi_eff,
        mu_chi_eff=mu_chi_eff,
        sigma_chi_eff=sigma_chi_eff,
        eta_chi_eff=eta_chi_eff,
    )


    return raw / norm


class EffectiveSpinSkewNormalChiEffChiP:
    """
    p(chi_eff, chi_p) = SkewNormal_truncated*TruncNormal
    eta_chi_eff > 0 gives a broader positive chi_eff tail.
    eta_chi_eff = 0 reduces to a truncated Gaussian chi_eff model.
    """

    variable_names = [
        "mu_chi_eff",
        "sigma_chi_eff",
        "eta_chi_eff",
        "mu_chi_p",
        "sigma_chi_p",
    ]

    def __init__(self, n_norm=1000):
        self.n_norm = n_norm

    def __call__(self, dataset, **params):
        chi_eff = dataset["chi_eff"]
        chi_p = dataset["chi_p"]

        p_chi_eff = _truncated_skewnormal_chi_eff(
            chi_eff,
            mu_chi_eff=params["mu_chi_eff"],
            sigma_chi_eff=params["sigma_chi_eff"],
            eta_chi_eff=params["eta_chi_eff"],
            low=-1.0,
            high=1.0,
            n_norm=self.n_norm,
        )

        p_chi_p = _gaussian_chi_p(
            chi_p,
            mu_chi_p=params["mu_chi_p"],
            sigma_chi_p=params["sigma_chi_p"],
        )

        prob = p_chi_eff * p_chi_p

       

        return prob
