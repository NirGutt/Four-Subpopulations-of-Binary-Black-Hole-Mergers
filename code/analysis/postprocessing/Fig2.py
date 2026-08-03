#!/usr/bin/env python

import os
import numpy as np
import matplotlib.pyplot as plt
import bilby
from scipy.stats import gaussian_kde

BASE_FONTSIZE = 13

RESULT_FILE = "gwtc5_result.hdf5"
OUTDIR = "xi_chi_publication"

# True:  xi_chi = Gaussian fraction, 1-xi_chi = uniform fraction.
# False: xi_chi = uniform fraction, 1-xi_chi = Gaussian fraction.
XI_IS_GAUSSIAN_FRACTION = False



COMPONENTS = ["C1", "C2", "C3", "C4"]


COMPONENT_LABELS = {
    "C1": r"$\mathrm{low-mass}$",
    "C2": r"$\mathrm{horizontal}$",
    "C3": r"$\mathrm{diagonal}$",
    "C4": r"$\mathrm{high-mass}$",
}

STYLE = {
    "C1": {"color": "#0072B2", "ls": "-",  "lw": 2.4},
    "C2": {"color": "#D55E00", "ls": "--", "lw": 2.4},
    "C3": {"color": "#009E73", "ls": "-.", "lw": 2.4},
    "C4": {"color": "#CC79A7", "ls": ":",  "lw": 3.0},
}


def summarize(samples):
    
    return {
        "median": np.percentile(samples, 50),
        "low": np.percentile(samples, 5),
        "high": np.percentile(samples, 95),
        "mean": np.mean(samples),
    }


def load_posterior(result_file):
    result = bilby.result.read_in_result(result_file)
    posterior = result.posterior

    print("Loaded:", result_file)
    print("Number of posterior samples:", len(posterior))

    return posterior



def kde_1d(samples, x, low=0.0, high=1.0, bw_method=0.25):
    """
    KDE for a bounded variable on [low, high], using boundary reflection.
    This avoids the artificial KDE drop near 0 and 1.
    """
    
    x = np.asarray(x, dtype=float)

    samples = samples[
        np.isfinite(samples)
        & (samples >= low)
        & (samples <= high)
    ]
  
    # Reflect samples around both boundaries.
    reflected = np.concatenate(
        [
            samples,
            2.0 * low - samples,
            2.0 * high - samples,
        ]
    )

    
    kde = gaussian_kde(reflected, bw_method=bw_method)
    y = kde(x)
   
    y = np.where(np.isfinite(y) & (y > 0.0), y, 0.0)

    # Renormalize only on the physical interval.
    norm = np.trapz(y, x=x)
    if np.isfinite(norm) and norm > 0.0:
        y = y / norm

    return y


def xi_columns():
    
    base= 'xi'
    cols = {
        "C1": f"{base}_1",
        "C2": f"{base}_2",
        "C3": f"{base}_3",
        "C4": f"{base}_4",
    }
    return cols
    
    

def plot_xi_density(posterior, params, outdir):
    os.makedirs(outdir, exist_ok=True)

    x = np.linspace(0.0, 1.0, 600)

    
    if XI_IS_GAUSSIAN_FRACTION:
        left_label = "uniform dominated"
        right_label = "Gaussian dominated"
    else:
        left_label = "Gaussian dominated"
        right_label = "uniform dominated"

    summaries = {
        c: summarize(posterior[params[c]])
        for c in COMPONENTS
    }

    with plt.rc_context(
        {
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
    ):
        fig, ax = plt.subplots(figsize=(6.5, 4.2))

        ax.axvspan(
            0.0,
            0.5,
            color="0.94",
            zorder=-10,
        )



        for c in COMPONENTS:
            samples = posterior[params[c]]
            y = kde_1d(samples, x, low=0.0, high=1.0, bw_method=0.03)

            st = STYLE[c]
            q = summaries[c]

            label = (
                rf"{COMPONENT_LABELS[c]}: "
                rf"${q['median']:.2f}^{{+{q['high'] - q['median']:.2f}}}"
                rf"_{{-{q['median'] - q['low']:.2f}}}$"
            )

            ax.plot(
                x,
                y,
                color=st["color"],
                ls=st["ls"],
                lw=st["lw"],
                label=label,
            )



        ax.text(
            0.25,
            0.96,
            left_label,
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize=BASE_FONTSIZE,
            color="0.35",
        )

        ax.text(
            0.75,
            0.96,
            right_label,
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize=BASE_FONTSIZE,
            color="0.35",
        )

        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(bottom=0.0)

        ax.set_xlabel(r"$\xi_{\chi}$")
        ax.set_ylabel(r"$p(\xi_\chi)$")

        ax.set_xticks([0.0, 0.25, 0.5, 0.75, 1.0])
        ax.set_xticklabels([r"$0$", r"$0.25$", r"$0.5$", r"$0.75$", r"$1$"])

        ax.grid(True, alpha=0.18)

        ax.legend(
            loc="center right",
            ncol=1,
            frameon=False,
            fontsize=13,
            handlelength=3.0,
            labelspacing=0.55,
        )

        fig.tight_layout()

        png_path = os.path.join(outdir, "xi_chi_density.png")
        pdf_path = os.path.join(outdir, "xi_chi_density.pdf")

        fig.savefig(png_path, dpi=350)
        fig.savefig(pdf_path)
        plt.close(fig)

    print("Saved:", png_path)
    print("Saved:", pdf_path)

    return summaries




def main():
    os.makedirs(OUTDIR, exist_ok=True)

    posterior = load_posterior(RESULT_FILE)
    params = xi_columns()

    plot_xi_density(
            posterior=posterior,
            params=params,
            outdir=OUTDIR,
        )

    
  
if __name__ == "__main__":
    main()
