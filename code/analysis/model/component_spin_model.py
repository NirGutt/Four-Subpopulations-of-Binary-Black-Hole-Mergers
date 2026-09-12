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


def _safe_where(mask, x, y=0.0):
    return xp.where(mask, x, y)


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


def planck_taper_low(m, mmin, delta_m):
    return planck_taper(m, mmin, delta_m)


def planck_taper_high(m, mmax, delta_m):
    return planck_taper(mmax - m, 0.0, delta_m)


def smooth_window(m, mmin, mmax, delta_low, delta_high):
    return (
        planck_taper_low(m, mmin, delta_low)
        * planck_taper_high(m, mmax, delta_high)
    )


def _powerlaw_with_two_sided_smoothing(
    m,
    alpha,
    low,
    high,
    smooth_low,
    smooth_high,
):
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


def _gaussian_spin_magnitude(a, mu_a, sigma_a):
    return truncnorm(
        a,
        mu=mu_a,
        sigma=sigma_a,
        low=0.0,
        high=1.0,
    )


def _uniform_costilt(costilt):
    inside = (costilt >= -1.0) & (costilt <= 1.0)
    return _safe_where(inside, 0.5, 0.0)


def _aligned_costilt(costilt, sigma_tilt, mu_tilt=1.0):
    return truncnorm(
        costilt,
        mu=mu_tilt,
        sigma=sigma_tilt,
        low=-1.0,
        high=1.0,
    )


def _tilt_mixture(costilt, xi_tilt, sigma_tilt, mu_tilt=1.0):
    """
    Isotropic + aligned tilt mixture
    """
    p_iso = _uniform_costilt(costilt)

    p_aligned = _aligned_costilt(
        costilt,
        sigma_tilt=sigma_tilt,
        mu_tilt=mu_tilt,
    )

    return (1.0 - xi_tilt) * p_iso + xi_tilt * p_aligned


def _physical_spin_component(
    a1,
    a2,
    costilt1,
    costilt2,
    mu_a1,
    sigma_a1,
    mu_a2,
    sigma_a2,
    xi_tilt1,
    sigma_tilt1,
    xi_tilt2,
    sigma_tilt2,
    mu_tilt1=1.0,
    mu_tilt2=1.0,
):
    """
    Physical-spin distribution 
    """
    p_a1 = _gaussian_spin_magnitude(
        a1,
        mu_a=mu_a1,
        sigma_a=sigma_a1,
    )

    p_a2 = _gaussian_spin_magnitude(
        a2,
        mu_a=mu_a2,
        sigma_a=sigma_a2,
    )

    p_t1 = _tilt_mixture(
        costilt1,
        xi_tilt=xi_tilt1,
        sigma_tilt=sigma_tilt1,
        mu_tilt=mu_tilt1,
    )

    p_t2 = _tilt_mixture(
        costilt2,
        xi_tilt=xi_tilt2,
        sigma_tilt=sigma_tilt2,
        mu_tilt=mu_tilt2,
    )

    prob = p_a1 * p_a2 * p_t1 * p_t2

    valid = (
        (mu_a1 >= 0.0)
        & (mu_a1 <= 1.0)
        & (mu_a2 >= 0.0)
        & (mu_a2 <= 1.0)
        & (sigma_a1 > 0.0)
        & (sigma_a2 > 0.0)
        & (xi_tilt1 >= 0.0)
        & (xi_tilt1 <= 1.0)
        & (xi_tilt2 >= 0.0)
        & (xi_tilt2 <= 1.0)
        & (sigma_tilt1 > 0.0)
        & (sigma_tilt2 > 0.0)
    )

    return _safe_where(valid, prob, 0.0)



class PiStrokeFourComponentPhysicalSpin:
    """
    Physical spin:
    a ~ G0,1](mu_a1_k, sigma_a_k)
    p(cos theta)=(1 - xi_tilt_k) U(-1,1)+ xi_tilt_k G[-1,1](1, sigma_tilt)
    """

    variable_names = [
        # Mixture fractions
        "f_1",
        "f_2",
        "f_3",

        # Component 1: low-mass 
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

        # Component 3: diagonal 
        "alpha_m1_3",
        "smooth_m1_3",
        "m1_min_3",
        "m1_max_3",

        # Component 3 Gaussian excess in m1
        "lambda_m1_peak_3",
        "mu_m1_peak_3",
        "sigma_m1_peak_3",

        # Component 3 q distribution
        "mu_q_3",
        "sigma_q_3",

        # Component 4: high-mass 
        "alpha_m1_4",
        "smooth_m1_4",
        "m1_min_4",
        "m1_max_4",
        "mu_q_4",
        "sigma_q_4",


        # Shared redshift evolution
        "kappa",

        # Physical spin parameters: component 1
        "mu_a1_1",
        "sigma_a1_1",
        "xi_tilt1_1",
        "sigma_tilt",
       

        # Physical spin parameters: component 2
        "mu_a1_2",
        "sigma_a1_2",
        "xi_tilt1_2",
        
       

        # Physical spin parameters: component 3
        "mu_a1_3",
        "sigma_a1_3",
        "xi_tilt1_3",
        
       

        # Physical spin parameters: component 4
        "mu_a1_4",
        "sigma_a1_4",
        "xi_tilt1_4",
       
        
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

        self.m1s_grid, self.qs_grid = xp.meshgrid(
            self.m1_grid,
            self.q_grid,
        )

        self.grid_dataset = {
            "mass_1": self.m1s_grid,
            "mass_ratio": self.qs_grid,
        }

        self.redshift_model = PowerLawRedshift(z_max=z_max)

    def __call__(self, dataset, **params):
        m1 = dataset["mass_1"]
        q = dataset["mass_ratio"]
        redshift = dataset["redshift"]

        a1 = dataset["a_1"]
        a2 = dataset["a_2"]
        costilt1 = dataset["cos_tilt_1"]
        costilt2 = dataset["cos_tilt_2"]

        f1 = params["f_1"]
        f2 = params["f_2"]
        f3 = params["f_3"]
        f4 = 1.0 - f1 - f2 - f3

        valid_fractions = (
            (f1 >= 0.0)
            & (f2 >= 0.0)
            & (f3 >= 0.0)
            & (f4 >= 0.0)
        )

        p1 = self.component_1(dataset, **params)
        p2 = self.component_2(dataset, **params)
        p3 = self.component_3(dataset, **params)
        p4 = self.component_4(dataset, **params)

        n1 = self.component_1_norm(**params)
        n2 = self.component_2_norm(**params)
        n3 = self.component_3_norm(**params)
        n4 = self.component_4_norm(**params)

        p1 = p1 / n1
        p2 = p2 / n2
        p3 = p3 / n3
        p4 = p4 / n4

        s1 = _physical_spin_component(
            a1,
            a2,
            costilt1,
            costilt2,
            mu_a1=params["mu_a1_1"],
            sigma_a1=params["sigma_a1_1"],
            mu_a2=params["mu_a1_1"],
            sigma_a2=params["sigma_a1_1"],
            xi_tilt1=params["xi_tilt1_1"],
            sigma_tilt1=params["sigma_tilt"],
            xi_tilt2=params["xi_tilt1_1"],
            sigma_tilt2=params["sigma_tilt"],
        )

        s2 = _physical_spin_component(
            a1,
            a2,
            costilt1,
            costilt2,
            mu_a1=params["mu_a1_2"],
            sigma_a1=params["sigma_a1_2"],
            mu_a2=params["mu_a1_2"],
            sigma_a2=params["sigma_a1_2"],
            xi_tilt1=params["xi_tilt1_2"],
            sigma_tilt1=params["sigma_tilt"],
            xi_tilt2=params["xi_tilt1_2"],
            sigma_tilt2=params["sigma_tilt"],
        )

        s3 = _physical_spin_component(
            a1,
            a2,
            costilt1,
            costilt2,
            mu_a1=params["mu_a1_3"],
            sigma_a1=params["sigma_a1_3"],
            mu_a2=params["mu_a1_3"],
            sigma_a2=params["sigma_a1_3"],
            xi_tilt1=params["xi_tilt1_3"],
            sigma_tilt1=params["sigma_tilt"],
            xi_tilt2=params["xi_tilt1_3"],
            sigma_tilt2=params["sigma_tilt"],
        )

        s4 = _physical_spin_component(
            a1,
            a2,
            costilt1,
            costilt2,
            mu_a1=params["mu_a1_4"],
            sigma_a1=params["sigma_a1_4"],
            mu_a2=params["mu_a1_4"],
            sigma_a2=params["sigma_a1_4"],
            xi_tilt1=params["xi_tilt1_4"],
            sigma_tilt1=params["sigma_tilt"],
            xi_tilt2=params["xi_tilt1_4"],
            sigma_tilt2=params["sigma_tilt"],
        )

        pz = self.p_z_component(redshift, params["kappa"])

        prob = (
              f1 * p1 * s1
            + f2 * p2 * s2
            + f3 * p3 * s3
            + f4 * p4 * s4
        )

        prob = prob * pz

        valid_domain = (
            (m1 > self.mmin)
            & (m1 < self.mmax)
            & (q > self.qmin)
            & (q <= 1.0)
            & (a1 >= 0.0)
            & (a1 <= 1.0)
            & (a2 >= 0.0)
            & (a2 <= 1.0)
            & (costilt1 >= -1.0)
            & (costilt1 <= 1.0)
            & (costilt2 >= -1.0)
            & (costilt2 <= 1.0)
        )

        prob = _safe_where(valid_fractions, prob, 0.0)
        prob = _safe_where(valid_domain, prob, 0.0)

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
        return self._integrate_grid(
            self.component_1(self.grid_dataset, **params)
        )

    def component_2_norm(self, **params):
        return self._integrate_grid(
            self.component_2(self.grid_dataset, **params)
        )

    def component_3_norm(self, **params):
        return self._integrate_grid(
            self.component_3(self.grid_dataset, **params)
        )

    def component_4_norm(self, **params):
        return self._integrate_grid(
            self.component_4(self.grid_dataset, **params)
        )

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
        diagonal 
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
