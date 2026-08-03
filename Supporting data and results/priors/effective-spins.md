# Prior ranges: effective-spin model

This file lists the prior ranges used for the four-component effective-spin population model.

## Mixing fractions

| Parameter | Prior | Minimum | Maximum | Notes |
|---|---:|---:|---:|---|
| `f_1` | Uniform | 0.0 | 1.0 | Mixing fraction for the low-mass component |
| `f_2` | ConditionalUniform | 0.0 | 1.0 | Mixing fraction for the horizontal component |
| `f_3` | ConditionalUniform | 0.0 | 1.0 | Mixing fraction for the diagonal component |

The fourth mixing fraction is determined by normalization,
`f_4 = 1 - f_1 - f_2 - f_3`, and corresponds to the high-mass component.

## Effective-spin model

| Parameter | Prior | Minimum | Maximum | Notes |
|---|---:|---:|---:|---|
| `mu_chi_eff` | Uniform | -0.3 | 0.3 | Mean of the Gaussian component in `chi_eff` |
| `sigma_chi_eff` | Uniform | 0.02 | 0.4 | Width of the Gaussian component in `chi_eff` |
| `xi_1` | Uniform | 0.0 | 1.0 | Gaussian-mixture fraction in `chi_eff` for the low-mass component |
| `xi_2` | Uniform | 0.0 | 1.0 | Gaussian-mixture fraction in `chi_eff` for the horizontal component |
| `xi_3` | Uniform | 0.0 | 1.0 | Gaussian-mixture fraction in `chi_eff` for the diagonal component |
| `xi_4` | Uniform | 0.0 | 1.0 | Gaussian-mixture fraction in `chi_eff` for the high-mass component |
| `mu_chi_p` | Uniform | 0.1 | 0.7 | Mean of the `chi_p` distribution |
| `sigma_chi_p` | Uniform | 0.1 | 0.6 | Width of the `chi_p` distribution |

The restricted-uniform component in `chi_eff` has fixed support
`chi_uniform_min = -0.47` and `chi_uniform_max = 0.47`.

## Mass model: low-mass component

| Parameter | Prior | Minimum | Maximum | Notes |
|---|---:|---:|---:|---|
| `m1_min_1` | Uniform | 3.0 | 7.0 | Lower primary-mass bound |
| `m1_max_1` | Uniform | 10.0 | 22.0 | Upper primary-mass bound |
| `delta_m_1` | Uniform | 0.3 | 4.0 | Low-mass smoothing scale |
| `mu_m1_1` | Uniform | 8.0 | 12.0 | Mean of the `m_1` truncated normal |
| `sigma_m1_1` | Uniform | 0.3 | 4.0 | Width of the `m_1` truncated normal |
| `mu_q_1` | Uniform | 0.5 | 1.0 | Mean of the mass-ratio truncated normal |
| `sigma_q_1` | Uniform | 0.02 | 0.4 | Width of the mass-ratio truncated normal |

## Mass model: horizontal component

| Parameter | Prior | Minimum | Maximum | Notes |
|---|---:|---:|---:|---|
| `m1_min_2` | Uniform | 3.0 | 20.0 | Lower primary-mass bound |
| `m1_max_2` | Uniform | 20.0 | 60.0 | Upper primary-mass bound |
| `delta_m1_2` | Uniform | 0.3 | 5.0 | Primary-mass smoothing scale |
| `alpha_m1_2` | Uniform | -2.0 | 10.0 | Primary-mass power-law slope |
| `mu_m2_2` | Uniform | 6.0 | 14.0 | Mean of the `m_2` truncated normal |
| `sigma_m2_2` | Uniform | 0.5 | 6.0 | Width of the `m_2` truncated normal |

## Mass model: diagonal component

| Parameter | Prior | Minimum | Maximum | Notes |
|---|---:|---:|---:|---|
| `m1_min_3` | Uniform | 3.0 | 28.0 | Lower primary-mass bound |
| `m1_max_3` | Uniform | 30.0 | 70.0 | Upper primary-mass bound |
| `delta_m1_3` | Uniform | 0.3 | 6.0 | Primary-mass smoothing scale |
| `alpha_m1_3` | Uniform | -2.0 | 10.0 | Primary-mass power-law slope |
| `lambda_m1_peak_3` | Uniform | 0.0 | 1.0 | Mixture fraction of the primary-mass peak |
| `mu_m1_peak_3` | Uniform | 22.0 | 42.0 | Mean of the primary-mass peak |
| `sigma_m1_peak_3` | Uniform | 1.0 | 12.0 | Width of the primary-mass peak |
| `mu_q_3` | Uniform | 0.5 | 1.0 | Mean of the mass-ratio truncated normal |
| `sigma_q_3` | Uniform | 0.02 | 0.4 | Width of the mass-ratio truncated normal |

## Mass model: high-mass component

| Parameter | Prior | Minimum | Maximum | Notes |
|---|---:|---:|---:|---|
| `m1_min_4` | Uniform | 30.0 | 75.0 | Lower primary-mass bound |
| `m1_max_4` | Uniform | 160.0 | 300.0 | Upper primary-mass bound |
| `delta_m1_4` | Uniform | 0.1 | 3.0 | Primary-mass smoothing scale |
| `alpha_m1_4` | Uniform | 0.0 | 12.0 | Primary-mass power-law slope |
| `mu_q_4` | Uniform | 0.2 | 0.9 | Mean of the mass-ratio truncated normal |
| `sigma_q_4` | Uniform | 0.05 | 0.5 | Width of the mass-ratio truncated normal |

## Redshift evolution

| Parameter | Prior | Minimum | Maximum | Notes |
|---|---:|---:|---:|---|
| `kappa` | Uniform | -1.0 | 6.0 | Redshift-evolution parameter |
