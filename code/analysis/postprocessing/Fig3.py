#!/usr/bin/env python

import os
import argparse

import numpy as np
import matplotlib.pyplot as plt
import bilby
import tqdm
from gwpopulation.models.mass import truncnorm
from scipy.stats import gaussian_kde




BASE_FONTSIZE = 13
PLOT_STYLE = {
    "font.size": BASE_FONTSIZE,
    "axes.labelsize": BASE_FONTSIZE + 2,
    "axes.titlesize": BASE_FONTSIZE,
    "legend.fontsize": BASE_FONTSIZE,
    "xtick.labelsize": BASE_FONTSIZE,
    "ytick.labelsize": BASE_FONTSIZE,
    "axes.linewidth": 1.1,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
}



COMPONENTS = ["C1", "C2", "C3", "C4"]


COMPONENT_INDEX = {
    "C1": 1,
    "C2": 2,
    "C3": 3,
    "C4": 4,
}

COMPONENT_LABELS = {
    "C1": r"$\mathrm{low\!-\!mass}$",
    "C2": r"$\mathrm{horizontal}$",
    "C3": r"$\mathrm{diagonal}$",
    "C4": r"$\mathrm{high\!-\!mass}$",
}

STYLE = {
    "C1": {"color": "#0072B2", "ls": "-",  "lw": 2.4},
    "C2": {"color": "#D55E00", "ls": "--", "lw": 2.4},
    "C3": {"color": "#009E73", "ls": "-.", "lw": 2.4},
    "C4": {"color": "#CC79A7", "ls": ":",  "lw": 3.0},
}




def load_bilby_posterior(result_file):
    result = bilby.result.read_in_result(str(result_file))
    posterior = result.posterior

    return posterior, result


def posterior_values(posterior, key):
    if key not in posterior.columns:
        raise KeyError(f"Missing posterior column: {key}")

    vals = np.asarray(posterior[key], dtype=float)
    vals = vals[np.isfinite(vals)]

    return vals


def params_from_row(row):
    params = {}
    for key in row.index:
        value = row[key]

        if np.isscalar(value):
            try:
                value = float(value)
            except Exception:
                continue

            if np.isfinite(value):
                params[key] = value
    return params



def summarize_curves(curves):
    curves = np.asarray(curves, dtype=float)

    if len(curves) == 0:
        raise ValueError("No curves to summarize.")

    return {
        "median": np.nanpercentile(curves, 50, axis=0),
        "low": np.nanpercentile(curves, 5, axis=0),
        "high": np.nanpercentile(curves, 95, axis=0),
    }


def renormalize_curve(y, x):
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)

    y = np.where(np.isfinite(y) & (y > 0.0), y, 0.0)

    norm = np.trapz(y, x=x)

    if np.isfinite(norm) and norm > 0.0:
        return y / norm

    return y

def bounded_reflected_kde(vals, x_grid, low=0.0, high=1.0, bw_method=0.15):
    vals = np.asarray(vals, dtype=float)
    vals = vals[np.isfinite(vals)]
    vals = vals[(vals >= low) & (vals <= high)]

    
    vals_reflected = np.concatenate(
        [
            vals,
            2.0 * low - vals,
            2.0 * high - vals,
        ]
    )

    kde = gaussian_kde(vals_reflected, bw_method=bw_method)
    y = kde(x_grid)

    return renormalize_curve(y, x_grid)




def truncated_gaussian_pdf(x, mu, sigma, low, high):
    x = np.asarray(x, dtype=float)

    mu = float(mu)
    sigma = float(sigma)
    low = float(low)
    high = float(high)


    p = truncnorm(
        x,
        mu=mu,
        sigma=sigma,
        low=low,
        high=high,
    )

    p = np.asarray(p, dtype=float)


    return p


def spin_magnitude_pdf(a, params, component):
    
    i = COMPONENT_INDEX[component]

    return truncated_gaussian_pdf(
            a,
            mu=params[f"mu_a1_{i}"],
            sigma=params[f"sigma_a1_{i}"],
            low=0.0,
            high=1.0,
        )
       


def tilt_costilt_pdf(x, params, component):

    i = COMPONENT_INDEX[component]

    xi = params[f"xi_tilt1_{i}"]
    sigma = params[f"sigma_tilt"]

    p_iso = np.where((x >= -1.0) & (x <= 1.0), 0.5, 0.0)

    p_aligned = truncated_gaussian_pdf(
        x,
        mu=1.0,
        sigma=sigma,
        low=-1.0,
        high=1.0,
    )

    p = (1.0 - xi) * p_iso + xi * p_aligned

    return p






def build_spin_and_tilt_curves(
    posterior,
    n_grid=600,
):
    rng = np.random.default_rng(0)
    n_draws = len(posterior)

    indices = rng.choice(n_draws, size=n_draws, replace=False)

    a_grid = np.linspace(0.0, 1.0, n_grid)
    xi_grid = np.linspace(0.0, 1.0, n_grid)
    costilt_grid = np.linspace(-1.0, 1.0, n_grid)
    theta_grid = np.linspace(0.0, np.pi, n_grid)

    curves = {
        c: {
            "a": [],
            "costilt": [],
            "theta": [],
        }
        for c in COMPONENTS
    }

    skipped = 0

    for idx in tqdm.tqdm(indices, desc="Building spin/tilt curves"):
        try:
            params = params_from_row(posterior.iloc[idx])

            for c in COMPONENTS:
                curves[c]["a"].append(
                    spin_magnitude_pdf(
                        a=a_grid,
                        params=params,
                        component=c,
                    )
                )

                curves[c]["costilt"].append(
                    tilt_costilt_pdf(
                        x=costilt_grid,
                        params=params,
                        component=c,
                    )
                )

                

        except Exception as exc:
            skipped += 1
            print(f"Skipping sample {idx}: {exc}")

    print("Skipped samples:", skipped)

    if skipped == n_draws:
        raise RuntimeError("All samples were skipped.")

    summaries = {
        c: {
            "a": summarize_curves(curves[c]["a"]),
            "costilt": summarize_curves(curves[c]["costilt"]),            
        }
        for c in COMPONENTS
    }

    return a_grid, xi_grid, costilt_grid, theta_grid, summaries




def plot_spin_tilt_publication_1x2(
    posterior,
    outdir,
    n_grid=600,
    filename="shared_spin_tilt_publication_1x2",
):
    
    os.makedirs(outdir, exist_ok=True)


    a_grid, xi_grid, costilt_grid, theta_grid, summaries = build_spin_and_tilt_curves(
        posterior=posterior,      
        n_grid=n_grid,     
    )

    with plt.rc_context(PLOT_STYLE):
        fig, (ax_a, ax_xi) = plt.subplots(
            1,
            2,
            figsize=(2*6.5, 4.2),
            constrained_layout=True,
        )
                
       
        # Left panel: 
       
        ymax = 0.0
        for c in COMPONENTS:
            st = STYLE[c]
            s = summaries[c]["a"]

            ax_a.fill_between(
                a_grid,
                s["low"],
                s["high"],
                color=st["color"],
                alpha=0.16,
                linewidth=0.0,
                zorder=1,
            )

            ax_a.plot(
                a_grid,
                s["median"],
                color=st["color"],
                linestyle=st["ls"],
                lw=st["lw"],
                label=COMPONENT_LABELS[c],
                zorder=5,
            )

        ax_a.set_xlim(0.0, 1.0)
        ax_a.set_ylim(0.0, None)
        ax_a.set_xlabel(r"$\chi$")
        ax_a.set_ylabel(r"$p(\chi)$")
        ax_a.set_xticks([0.0, 0.25, 0.5, 0.75, 1.0])
        ax_a.set_yticks([0.0, 1.0, 2.0, 3.0])
        ax_a.grid(True, alpha=0.16)

        ax_a.legend(
            frameon=False,
            loc="upper right",
            handlelength=2.8,
            labelspacing=0.35,
            borderaxespad=0.3,
        )

        
        #Right panel: 
       

        ax_xi.axvspan(
            0.0,
            0.5,
            color="0.92",
            alpha=0.35,
            linewidth=0.0,
            zorder=0,
        )

        ax_xi.axvspan(
            0.5,
            1.0,
            color="#F0E6C8",
            alpha=0.25,
            linewidth=0.0,
            zorder=0,
        )

        ax_xi.axvline(
            0.5,
            color="0.45",
            lw=1.0,
            ls=":",
            zorder=1,
        )

        ymax = 0.0

        for c in COMPONENTS:
            i = COMPONENT_INDEX[c]
            st = STYLE[c]

            xi = posterior_values(posterior, f"xi_tilt1_{i}")

            y = bounded_reflected_kde(
                xi,
                xi_grid,
                low=0.0,
                high=1.0,
                bw_method=0.12,
            )

            ymax = max(ymax, np.nanmax(y))

            ax_xi.plot(
                xi_grid,
                y,
                color=st["color"],
                linestyle=st["ls"],
                lw=st["lw"],
                label=COMPONENT_LABELS[c],
                zorder=5,
            )

        ax_xi.text(
            0.25,
            0.90,
            "isotropic",
            ha="center",
            va="bottom",
            fontsize=BASE_FONTSIZE - 1,
            color="0.25",
            transform=ax_xi.transAxes,
        )

        ax_xi.text(
            0.75,
            0.90,
            "aligned",
            ha="center",
            va="bottom",
            fontsize=BASE_FONTSIZE - 1,
            color="0.25",
            transform=ax_xi.transAxes,
        )

        ax_xi.set_xlim(0.0, 1.0)
        ax_xi.set_ylim(0.0, 1.12 * ymax)
        ax_xi.set_xlabel(r"$\xi_{\rm tilt}$")
        ax_xi.set_ylabel(r"$p(\xi_{\rm tilt})$")
        ax_xi.set_xticks([0.0, 0.25, 0.5, 0.75, 1.0])
        ax_xi.grid(True, alpha=0.16)

       
        from matplotlib.ticker import FormatStrFormatter
        ax_a.xaxis.set_major_formatter(FormatStrFormatter("%.1f"))
        ax_xi.xaxis.set_major_formatter(FormatStrFormatter("%.1f"))
        
        ax_a.set_yticks([0.0, 1.0, 2.0, 3.0])
        ax_xi.set_yticks([0.0, 1.0, 2.0])
      
        # fix the ticks
        for ax in [ax_a, ax_xi]:
            ax.tick_params(axis="both", which="major", pad=7)
            for spine in ax.spines.values():
                spine.set_linewidth(1.1)
                
        ax_a.set_xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ax_xi.set_xticks([0.0, 0.5, 1.0])
        
        

        png_path = os.path.join(outdir, f"{filename}.png")
        pdf_path = os.path.join(outdir, f"{filename}.pdf")

        fig.savefig(png_path, dpi=350)
        fig.savefig(pdf_path)
        plt.close(fig)

    print("Saved:", png_path)
    print("Saved:", pdf_path)






def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--result",
        default="gwtc5_result.hdf5",
        help="Bilby result file.",
    )

    parser.add_argument(
        "--outdir",
        default="publication_figs_shared_spin_tilt",
        help="Output directory.",
    )

    
    parser.add_argument(
        "--n-grid",
        type=int,
        default=600,
    )

  
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    posterior, result = load_bilby_posterior(args.result)
    
    plot_spin_tilt_publication_1x2(
        posterior=posterior,
        outdir=args.outdir,
        n_grid=args.n_grid,  
        filename="shared_spin_tilt_publication_1x2",
    )


  
if __name__ == "__main__":
    main()
