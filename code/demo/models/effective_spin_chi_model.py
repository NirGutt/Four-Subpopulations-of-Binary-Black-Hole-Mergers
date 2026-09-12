from gwpopulation.utils import xp
from gwpopulation.models.mass import truncnorm


def _gaussian_chi_eff(chi_eff, mu_chi_eff, sigma_chi_eff):
    return truncnorm(
        chi_eff,
        mu=mu_chi_eff,
        sigma=sigma_chi_eff,
        low=-1.0,
        high=1.0,
    )


def _gaussian_chi_p(chi_p, mu_chi_p, sigma_chi_p):
    return truncnorm(
        chi_p,
        mu=mu_chi_p,
        sigma=sigma_chi_p,
        low=0.0,
        high=1.0,
    )


class EffectiveSpinChiEffChiP:
    """
    Factorized effective-spin model:
        p(chi_eff, chi_p)=G(chi_eff)*G(chi_p)    
    """

    variable_names = [
        "mu_chi_eff",
        "sigma_chi_eff",        
        "mu_chi_p",
        "sigma_chi_p",
    ]

    def __call__(self, dataset, **params):
        chi_eff = dataset["chi_eff"]
        chi_p = dataset["chi_p"]

        p_chi_eff = _gaussian_chi_eff(
            chi_eff,
            mu_chi_eff=params["mu_chi_eff"],
            sigma_chi_eff=params["sigma_chi_eff"],
        )

        
        p_chi_p = _gaussian_chi_p(
            chi_p,
            mu_chi_p=params["mu_chi_p"],
            sigma_chi_p=params["sigma_chi_p"],
        )

        prob = p_chi_eff * p_chi_p

        

        return prob
