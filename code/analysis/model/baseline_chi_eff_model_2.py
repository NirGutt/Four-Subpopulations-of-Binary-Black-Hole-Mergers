from gwpopulation.utils import xp
from gwpopulation.models.mass import truncnorm


def _safe_where(mask, x, y=0.0):
    return xp.where(mask, x, y)


def _uniform_chi_eff(chi_eff, chi_uniform_min, chi_uniform_max):
    """
    Uniform density on [chi_uniform_min, chi_uniform_max],
    zero outside.
    """
    width = chi_uniform_max - chi_uniform_min

    inside = (
        (chi_eff >= chi_uniform_min)
        & (chi_eff <= chi_uniform_max)
    )


    return _safe_where(inside, 1.0 / width, 0.0)


def _gaussian_chi_eff(chi_eff, mu_chi_eff, sigma_chi_eff):
    """
    Gaussian chi_eff distribution truncated to [-1, 1].
    """
    return truncnorm(
        chi_eff,
        mu=mu_chi_eff,
        sigma=sigma_chi_eff,
        low=-1.0,
        high=1.0,
    )


def _gaussian_chi_p(chi_p, mu_chi_p, sigma_chi_p):
    """
    Gaussian chi_p distribution truncated to [0, 1].
    """
    return truncnorm(
        chi_p,
        mu=mu_chi_p,
        sigma=sigma_chi_p,
        low=0.0,
        high=1.0,
    )


class EffectiveSpinGaussianUniformChiEffChiP:
    """
    Baseline spin model:
        p(chi_eff, chi_p)=[(1 - xi_eff) G(chi_eff) + xi_eff U(chi_eff)]*G(chi_p)
    """

    variable_names = [
        "mu_chi_eff",
        "sigma_chi_eff",
        "xi_eff",
        "chi_uniform_min",
        "chi_uniform_max",
        "mu_chi_p",
        "sigma_chi_p",
    ]

    def __call__(self, dataset, **params):
        chi_eff = dataset["chi_eff"]
        chi_p = dataset["chi_p"]

        p_gaussian_eff = _gaussian_chi_eff(
            chi_eff,
            mu_chi_eff=params["mu_chi_eff"],
            sigma_chi_eff=params["sigma_chi_eff"],
        )

        p_uniform_eff = _uniform_chi_eff(
            chi_eff,
            chi_uniform_min=params["chi_uniform_min"],
            chi_uniform_max=params["chi_uniform_max"],
        )

        xi_eff = params["xi_eff"]

        p_chi_eff = (
            (1.0 - xi_eff) * p_gaussian_eff
            + xi_eff * p_uniform_eff
        )

        p_chi_p = _gaussian_chi_p(
            chi_p,
            mu_chi_p=params["mu_chi_p"],
            sigma_chi_p=params["sigma_chi_p"],
        )

        prob = p_chi_eff * p_chi_p

        return prob



