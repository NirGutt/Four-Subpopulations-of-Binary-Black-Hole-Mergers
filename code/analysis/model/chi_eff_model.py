from gwpopulation.utils import xp
from gwpopulation.models.mass import truncnorm
from gwpopulation.models.redshift import PowerLawRedshift

def f_2_condition(reference_params, f_1):
    return dict(
        minimum=0.0,
        maximum=1.0 - f_1,
    )


def f_3_condition(reference_params, f_1, f_2):
    return dict(
        minimum=0.0,
        maximum=1.0 - f_1 - f_2,
    )

def planck_taper_low(m, mmin, delta_m):
    return planck_taper(m, mmin, delta_m)


def planck_taper_high(m, mmax, delta_m):
    return planck_taper(mmax - m, 0.0, delta_m)


def smooth_window(m, mmin, mmax, delta_low, delta_high):
    return (
        planck_taper_low(m, mmin, delta_low)
        * planck_taper_high(m, mmax, delta_high)
    )

def planck_taper(m, mmin, delta_m):
    m = xp.asarray(m)

    delta_m = xp.maximum(delta_m, 1e-12)
    x = (m - mmin) / delta_m

    middle = (x > 0.0) & (x < 1.0)
    high = x >= 1.0

    x_safe = xp.clip(x, 1e-12, 1.0 - 1e-12)

    exponent = 1.0 / x_safe + 1.0 / (x_safe - 1.0)
    taper_middle = 1.0 / (xp.exp(exponent) + 1.0)

    taper = xp.where(high, 1.0, 0.0)
    taper = xp.where(middle, taper_middle, taper)

    return taper


def _powerlaw_with_two_sided_smoothing(m, alpha, low, high, smooth_low, smooth_high):
    base = xp.power(m, -alpha)

    window = smooth_window(
        m,
        mmin=low,
        mmax=high,
        delta_low=smooth_low,
        delta_high=smooth_high,
    )

    return base * window


def _trapz(y, x):
    return xp.trapz(y, x=x)


def _safe_norm(y, x):
    norm = _trapz(y, x=x)
    return xp.maximum(norm, 1e-300)


def _normalized_powerlaw_with_two_sided_smoothing(
    m,
    norm_m,
    alpha,
    low,
    high,
    smooth_low,
    smooth_high,
):

    p = _powerlaw_with_two_sided_smoothing(
        m,
        alpha=alpha,
        low=low,
        high=high,
        smooth_low=smooth_low,
        smooth_high=smooth_high,
    )

    p_norm = _powerlaw_with_two_sided_smoothing(
        norm_m,
        alpha=alpha,
        low=low,
        high=high,
        smooth_low=smooth_low,
        smooth_high=smooth_high,
    )

    norm = _safe_norm(p_norm, norm_m)

    return p / norm


def _smoothed_truncnorm(
    x,
    mu,
    sigma,
    low,
    high,
    smooth,
):
    
    p = truncnorm(
        x,
        mu=mu,
        sigma=sigma,
        low=low,
        high=high,
    )

    p *= smooth_window(
        x,
        mmin=low,
        mmax=high,
        delta_low=smooth,
        delta_high=smooth,
    )

    return p


def _normalized_smoothed_truncnorm(
    x,
    norm_x,
    mu,
    sigma,
    low,
    high,
    smooth,
):
   
    p = _smoothed_truncnorm(
        x,
        mu=mu,
        sigma=sigma,
        low=low,
        high=high,
        smooth=smooth,
    )

    p_norm = _smoothed_truncnorm(
        norm_x,
        mu=mu,
        sigma=sigma,
        low=low,
        high=high,
        smooth=smooth,
    )

    norm = _safe_norm(p_norm, norm_x)

    return p / norm    



def _safe_where(mask, x, y=0.0):
    return xp.where(mask, x, y)


def _unnorm_powerlaw(x, alpha, low, high, smooth=0.0):
    x = xp.asarray(x)
    inside = (x >= low) & (x <= high)

    base = xp.power(x, -alpha)

    turn_on_arg = xp.maximum(1.0 - low / xp.maximum(x, 1e-300), 0.0)
    smooth_factor = xp.power(turn_on_arg, smooth)

    prob = base * smooth_factor
    return _safe_where(inside, prob, 0.0)



def _uniform_chi_eff(chi_eff, chi_min=-0.47, chi_max=0.47):
    inside = (chi_eff >= chi_min) & (chi_eff <= chi_max)
    width = chi_max - chi_min
    return _safe_where(inside & (width > 0), 1.0 / width, 0.0)


def _gaussian_chi_eff(chi_eff, mu_chi_eff, sigma_chi_eff):
    return truncnorm(
        chi_eff,
        mu=mu_chi_eff,
        sigma=sigma_chi_eff,
        low=-1.0,
        high=1.0,
    )


def _spin_origin_mixture(
    chi_eff,
    xi,
    mu_chi_eff,
    sigma_chi_eff,
    chi_uniform_min=-0.47,
    chi_uniform_max=0.47,
):
    """
    Mixture of ordinary-spin Gaussian and broad-spin uniform component.
    """
    g = _gaussian_chi_eff(chi_eff, mu_chi_eff, sigma_chi_eff)
    u = _uniform_chi_eff(chi_eff, chi_uniform_min, chi_uniform_max)
    return (1.0 - xi) * g + xi * u


def _gaussian_chi_p(chi_p, mu_chi_p, sigma_chi_p):
    return truncnorm(
        chi_p,
        mu=mu_chi_p,
        sigma=sigma_chi_p,
        low=0.0,
        high=1.0,
    )


class PiStrokeFourComponentMassSpin:
    """
    Spin model:
        Each mass component has its own broad-spin fraction xi_k:

            p_k(chi_eff) = (1 - xi_k) G(chi_eff) + xi_k U(chi_eff)
    """

    variable_names = [
        # Mixture fractions
        "f_1",
        "f_2",
        "f_3",

        # Shared spin Gaussian
        "mu_chi_eff",
        "sigma_chi_eff",

        # Broad-spin fractions
        "xi_1",
        "xi_2",
        "xi_3",
        "xi_4",

        # uniform chi_eff bounds
        "chi_uniform_min",
        "chi_uniform_max",

        # Component 1: low-mass peak
        "m1_min_1",
        "m1_max_1",
        "delta_m_1",  
        "mu_m1_1",     
        "sigma_m1_1",     
        "mu_q_1",      
        "sigma_q_1",

        # Component 2: m2 ~ 10 strip
        "alpha_m1_2",
        "smooth_m1_2",
        "m1_min_2",
        "m1_max_2",
        "mu_m2_2",
        "sigma_m2_2",

        # Component 3: diagonal branch
        "alpha_m1_3",
        "smooth_m1_3",
        "m1_min_3",
        "m1_max_3",
        
        # C3 Gaussian excess in m1
        "lambda_m1_peak_3",
        "mu_m1_peak_3",
        "sigma_m1_peak_3",
        
        # C3 q distribution
        "mu_q_3",
        "sigma_q_3",

        # Component 4: high-mass 
        "alpha_m1_4",
        "smooth_m1_4",
        "m1_min_4",
        "m1_max_4",
       
        
        
        "mu_q_4",
        "sigma_q_4",
        
        
        # Component redshift evolution
        "kappa",

        "mu_chi_p",
        "sigma_chi_p",
    ]
    
    def __init__(
        self,
        mmin=2.0,
        mmax=300.0,
        qmin=1e-3,
        normalization_shape=(1500, 1500),
        z_max=1.9,
    ):
        self.mmin = mmin
        self.mmax = mmax
        self.qmin = qmin
    
        self.m1_grid = xp.linspace(mmin, mmax, normalization_shape[0])
        self.q_grid = xp.linspace(qmin, 1.0, normalization_shape[1])
    
        self.dm1 = (mmax - mmin) / (normalization_shape[0] - 1)
        self.dq = (1.0 - qmin) / (normalization_shape[1] - 1)
    
        self.m1s_grid, self.qs_grid = xp.meshgrid(self.m1_grid, self.q_grid)
    
        self.grid_dataset = {
            "mass_1": self.m1s_grid,
            "mass_ratio": self.qs_grid,
        }
    
        # Use GWPopulation's own redshift model.
        self.redshift_model = PowerLawRedshift(z_max=z_max)

    def __call__(self, dataset, **params):
        m1 = dataset["mass_1"]
        q = dataset["mass_ratio"]
        chi_eff = dataset["chi_eff"]
        chi_p = dataset["chi_p"]
        redshift = dataset["redshift"]

        f1 = params["f_1"]
        f2 = params["f_2"]
        f3 = params["f_3"]
        f4 = 1.0 - f1 - f2 - f3

        valid_fractions = (f1 >= 0.0) & (f2 >= 0.0) & (f3 >= 0.0) & (f4 >= 0.0)

        p1 = self.component_1(dataset, **params)
        p2 = self.component_2(dataset, **params)
        p3 = self.component_3(dataset, **params)
        p4 = self.component_4(dataset, **params)

        z1 = self.component_1_norm(**params)
        z2 = self.component_2_norm(**params)
        z3 = self.component_3_norm(**params)
        z4 = self.component_4_norm(**params)

        p1 = p1 / z1
        p2 = p2 / z2
        p3 = p3 / z3
        p4 = p4 / z4

        chi_uniform_min = params.get("chi_uniform_min", -0.47)
        chi_uniform_max = params.get("chi_uniform_max", 0.47)

        s1 = _spin_origin_mixture(
            chi_eff,
            params["xi_1"],
            params["mu_chi_eff"],
            params["sigma_chi_eff"],
            chi_uniform_min,
            chi_uniform_max,
        )
        s2 = _spin_origin_mixture(
            chi_eff,
            params["xi_2"],
            params["mu_chi_eff"],
            params["sigma_chi_eff"],
            chi_uniform_min,
            chi_uniform_max,
        )
        s3 = _spin_origin_mixture(
            chi_eff,
            params["xi_3"],
            params["mu_chi_eff"],
            params["sigma_chi_eff"],
            chi_uniform_min,
            chi_uniform_max,
        )
        s4 = _spin_origin_mixture(
            chi_eff,
            params["xi_4"],
            params["mu_chi_eff"],
            params["sigma_chi_eff"],
            chi_uniform_min,
            chi_uniform_max,
        )
        
        s_p = _gaussian_chi_p(
            chi_p,
            params["mu_chi_p"],
            params["sigma_chi_p"],
        )
        
        
        z1 = self.p_z_component(redshift, params["kappa"])
        z2 = self.p_z_component(redshift, params["kappa"])
        z3 = self.p_z_component(redshift, params["kappa"])
        z4 = self.p_z_component(redshift, params["kappa"])

        prob = (
            f1 * p1 * s1 * z1
            + f2 * p2 * s2 * z2
            + f3 * p3 * s3 * z3
            + f4 * p4 * s4 * z4
        )
        
        prob = prob * s_p
        
        prob = _safe_where(valid_fractions, prob, 0.0)
        prob = _safe_where((m1 > self.mmin) & (m1 < self.mmax) & (q > self.qmin) & (q <= 1.0), prob, 0.0)

        return prob
    
    def _integrate_grid(self, prob):
        return xp.trapz(
            xp.trapz(
                prob,
                dx=self.dq,
                axis=0,
            ),
            dx=self.dm1,
            axis=0,
        )
    
    def p_z_component(self, redshift, kappa):
        return self.redshift_model(
            {"redshift": redshift},
            lamb=kappa,
        )

    def component_1_norm(self, **params):
        return self._integrate_grid(self.component_1(self.grid_dataset, **params))

    def component_2_norm(self, **params):
        return self._integrate_grid(self.component_2(self.grid_dataset, **params))

    def component_3_norm(self, **params):
        return self._integrate_grid(self.component_3(self.grid_dataset, **params))

    def component_4_norm(self, **params):
        return self._integrate_grid(self.component_4(self.grid_dataset, **params))

    def component_1(self, dataset, **params):
        """
        Low-mass 
        """
        m1 = dataset["mass_1"]
        q = dataset["mass_ratio"]


        p_m1 = truncnorm(
            m1,
            mu=params["mu_m1_1"],
            sigma=params["sigma_m1_1"],
            low=params["m1_min_1"],
            high=params["m1_max_1"],
        )
        
        p_m1 *= smooth_window(
            m1,
            mmin=params["m1_min_1"],
            mmax=params["m1_max_1"],
            delta_low=params["delta_m_1"],
            delta_high=params["delta_m_1"],
        )
                

        p_q = truncnorm(
            q,
            mu=params["mu_q_1"],
            sigma=params["sigma_q_1"],
            low=self.qmin,
            high=1.0,
        )

        return p_m1 * p_q

    def component_2(self, dataset, **params):
        """
        Horizontal 
        """
        m1 = dataset["mass_1"]
        q = dataset["mass_ratio"]
        m2 = m1 * q

        p_m1 = _powerlaw_with_two_sided_smoothing(
            m1,
            alpha=params["alpha_m1_2"],
            low=params["m1_min_2"],
            high=params["m1_max_2"],
            smooth_low=params["smooth_m1_2"],
            smooth_high=params["smooth_m1_2"],
        )

        p_m2 = truncnorm(
            m2,
            mu=params["mu_m2_2"],
            sigma=params["sigma_m2_2"],
            low=self.mmin,
            high=m1,
        )

        jac = m1
        return p_m1 * p_m2 * jac


    def component_3_m1_distribution(self, m1, **params):
        """
        Diagonal
        """
        norm_m1 = self.m1_grid

        p_pl = _normalized_powerlaw_with_two_sided_smoothing(
            m1,
            norm_m=norm_m1,
            alpha=params["alpha_m1_3"],
            low=params["m1_min_3"],
            high=params["m1_max_3"],
            smooth_low=params["smooth_m1_3"],
            smooth_high=params["smooth_m1_3"],
        )

        p_g = _normalized_smoothed_truncnorm(
            m1,
            norm_x=norm_m1,
            mu=params["mu_m1_peak_3"],
            sigma=params["sigma_m1_peak_3"],
            low=params["m1_min_3"],
            high=params["m1_max_3"],
            smooth=params["smooth_m1_3"],
        )

        lam = params["lambda_m1_peak_3"]

        return (1.0 - lam) * p_pl + lam * p_g

    def component_3(self, dataset, **params):
        """
        C3: diagonal 
        """
        m1 = dataset["mass_1"]
        q = dataset["mass_ratio"]

        p_m1 = self.component_3_m1_distribution(
            m1,
            **params,
        )

        p_q = truncnorm(
            q,
            mu=params["mu_q_3"],
            sigma=params["sigma_q_3"],
            low=self.qmin,
            high=1.0,
        )

        return p_m1 * p_q    
    
   

    def component_4(self, dataset, **params):
        """
            C4: high-mass 
        """
        m1 = dataset["mass_1"]
        q = dataset["mass_ratio"]
    
        p_m1 = _powerlaw_with_two_sided_smoothing(
            m1,
            alpha=params["alpha_m1_4"],
            low=params["m1_min_4"],
            high=params["m1_max_4"],
            smooth_low=params["smooth_m1_4"],
            smooth_high=params["smooth_m1_4"],
        )
    
        p_q = truncnorm(
            q,
            mu=params["mu_q_4"],
            sigma=params["sigma_q_4"],
            low=self.qmin,
            high=1.0,
        )
        
        
        return p_m1 * p_q 


