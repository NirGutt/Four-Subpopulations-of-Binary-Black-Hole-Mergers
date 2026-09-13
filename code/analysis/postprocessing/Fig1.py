#!/usr/bin/env python

#this script creates the fig. 1 of the paper
import os
import argparse
from matplotlib.colors import LogNorm
import numpy as np
import matplotlib.pyplot as plt
import tqdm
import bilby
import pickle
from pathlib import Path
from phys_spin_model import PiStrokeFourComponentPhysicalSpin
from matplotlib.colors import LinearSegmentedColormap, LogNorm, to_rgba
from matplotlib.patches import Patch
from matplotlib.lines import Line2D


BASE_FONTSIZE = 13
PLOT_STYLE = {
    "font.size": BASE_FONTSIZE,
    "axes.labelsize": BASE_FONTSIZE + 1,
    "axes.titlesize": BASE_FONTSIZE,
    "legend.fontsize": BASE_FONTSIZE,
    "xtick.labelsize": BASE_FONTSIZE,
    "ytick.labelsize": BASE_FONTSIZE,
    "axes.linewidth": 1.0,
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
    "C1": r"$\mathrm{low-mass}$",
    "C2": r"$\mathrm{horizontal}$",
    "C3": r"$\mathrm{diagonal}$",
    "C4": r"$\mathrm{high-mass}$",
}

STYLE = {
    "C1": {"color": "#0072B2", "ls": "-",  "lw": 2.2},
    "C2": {"color": "#D55E00", "ls": "--", "lw": 2.2},
    "C3": {"color": "#009E73", "ls": "-.", "lw": 2.2},
    "C4": {"color": "#CC79A7", "ls": ":",  "lw": 2.8},
}




def weighted_histogram_density(x, weights, bins):
    x = np.asarray(x, dtype=float)
    weights = np.asarray(weights, dtype=float)

    ok = np.isfinite(x) & np.isfinite(weights) & (weights > 0.0)

    x = x[ok]
    weights = weights[ok]

    hist, edges = np.histogram(
        x,
        bins=bins,
        weights=weights,
    )

    widths = np.diff(edges)
    density = hist / widths
    centers = 0.5 * (edges[:-1] + edges[1:])

    return centers, density, edges


def assign_pistroke_clusters(m1, m2):
    """
    Manual cluster assignment for pi-stroke points, as there is on ground truth.
    """
    m1 = np.asarray(m1)
    m2 = np.asarray(m2)

    q = m2 / np.maximum(m1, 1e-300)

    cluster = np.full(len(m1), "C4", dtype=object)

    c1 = (
        (m1 < 16.0)
        & (m2 < 16.0)
    )

    c2 = (
        ~c1
        & (m2 < 18.0)
        & (m1 >= 18.0)
        & (m1 < 45.0)
    )

    c3 = (
        ~c1
        & ~c2
        & (q > 0.75)
        & (m1 < 50.0)
    )

    c4 = ~(c1 | c2 | c3)

    cluster[c1] = "C1"
    cluster[c2] = "C2"
    cluster[c3] = "C3"
    cluster[c4] = "C4"

    return cluster


def make_pistroke_dots_from_results(pts_xy, w_opt):
    pts_xy = np.asarray(pts_xy, dtype=float)
    w_opt = np.asarray(w_opt, dtype=float)

    if pts_xy.ndim != 2 or pts_xy.shape[1] < 2:
        raise ValueError(f"Expected pts_xy shape (N,2). Got {pts_xy.shape}")

    m1 = pts_xy[:, 0]
    m2 = pts_xy[:, 1]
    w = w_opt.copy()

    ok = (
        np.isfinite(m1)
        & np.isfinite(m2)
        & np.isfinite(w)
        & (m1 > 0.0)
        & (m2 > 0.0)
    )

    m1 = m1[ok]
    m2 = m2[ok]
    w = w[ok]

    w = np.maximum(w, 0.0)

    if len(m1) == 0:
        raise ValueError("No valid PiStroke points after filtering.")

    if np.nanmax(w) > 0.0:
        w_plot = w / np.nanmax(w)
    else:
        w_plot = np.ones_like(w)

    cluster = assign_pistroke_clusters(m1, m2)

    for c in COMPONENTS:
        print(f"PiStroke {c}: {np.sum(cluster == c)} points")

    return {
        "m1": m1,
        "m2": m2,
        "w": w,
        "w_plot": w_plot,
        "cluster": cluster,
    }


def load_results(result_file):
    result_file = Path(result_file)

    if not result_file.exists():
        raise FileNotFoundError(f"Could not find PiStroke results file: {result_file}")

    with result_file.open("rb") as f:
        results = pickle.load(f)

    x_ml = results["means"] if "means" in results else results["x_ml"]
    var_ml = results["vars"] if "vars" in results else -np.ones(np.shape(x_ml))

    return results["x_opt"], results["w_opt"], x_ml, results["w_ml"], var_ml


def to_numpy(x):
    if hasattr(x, "get"):
        return x.get()
    return np.asarray(x)


def clean_density(p):
    p = to_numpy(p).astype(float)
    return np.where(np.isfinite(p) & (p > 0.0), p, 0.0)


def load_bilby_posterior(result_file):
    result = bilby.result.read_in_result(str(result_file))
    posterior = result.posterior

    print("Loaded:", result_file)
    print("Number of posterior samples:", len(posterior))

    if hasattr(result, "log_evidence"):
        print("log_evidence:", result.log_evidence)
    if hasattr(result, "log_evidence_err"):
        print("log_evidence_err:", result.log_evidence_err)
    if hasattr(result, "log_bayes_factor"):
        print("log_bayes_factor:", result.log_bayes_factor)

    return posterior


def summarize(curves):
    curves = np.asarray(curves, dtype=float)

    if len(curves) == 0:
        raise ValueError("No curves to summarize.")

    return {
        "median": np.nanpercentile(curves, 50, axis=0),
        "low": np.nanpercentile(curves, 5, axis=0),
        "high": np.nanpercentile(curves, 95, axis=0),
    }


def positive_floor(*arrays, factor=1e-5, minimum=1e-12):
    vals = []

    for arr in arrays:
        arr = np.asarray(arr, dtype=float)
        arr = arr[np.isfinite(arr) & (arr > 0.0)]
        if arr.size > 0:
            vals.append(arr)

    if len(vals) == 0:
        return minimum

    vals = np.concatenate(vals)
    return max(np.nanmax(vals) * factor, minimum)


def relative_levels(p, fractions=(0.25,)):
    p = np.asarray(p, dtype=float)
    p = np.where(np.isfinite(p) & (p > 0.0), p, 0.0)

    pmax = np.nanmax(p)

    if not np.isfinite(pmax) or pmax <= 0.0:
        return None

    levels = np.asarray(fractions) * pmax
    levels = levels[(levels > 0.0) & (levels < pmax)]
    levels = np.unique(np.sort(levels))

    if levels.size == 0:
        return None

    return levels



def params_from_row(row):
    params = dict()

    for key in row.index:
        try:
            value = float(row[key])
        except Exception:
            continue

        if np.isfinite(value):
            params[key] = value

    f1 = params["f_1"]
    f2 = params["f_2"]
    f3 = params["f_3"]
    f4 = 1.0 - f1 - f2 - f3

    if f1 < 0.0 or f2 < 0.0 or f3 < 0.0 or f4 < 0.0:
        return None

    return params


def mixture_weights(params):
    f1 = params["f_1"]
    f2 = params["f_2"]
    f3 = params["f_3"]
    f4 = 1.0 - f1 - f2 - f3

    return {
        "C1": f1,
        "C2": f2,
        "C3": f3,
        "C4": f4,
    }



# model probability helpers
def component_mass_probability_m1q(model, component, dataset, params):
    """
    normalized component density in m1, q coordinates:
        p_k(m1, q) = raw_component_k(m1, q) / n_k
    """
    i = COMPONENT_INDEX[component]

    component_func = getattr(model, f"component_{i}")
    norm_func = getattr(model, f"component_{i}_norm")

    p = component_func(dataset, **params)
    n = norm_func(**params)

    p = clean_density(p)
    n = float(n)

    if not np.isfinite(n) or n <= 0.0:
        raise ValueError(f"Bad norm for {component}: {n}")

    return p / n


def total_mass_probability_m1q(model, dataset, params):
    """
    total mass density in m1, q coordinates:
        p(m1, q) = sum_k f_k p_k(m1, q)
    where each p_k is normalized exactly as in the model.

    """
    fracs = mixture_weights(params)

    total = None

    for c in COMPONENTS:
        pc = component_mass_probability_m1q(
            model=model,
            component=c,
            dataset=dataset,
            params=params,
        )

        if total is None:
            total = fracs[c] * pc
        else:
            total = total + fracs[c] * pc

    return clean_density(total)


def weighted_component_mass_probability_m1q(model, component, dataset, params):
    """
    Weighted component contribution:
        f_k p_k(m1, q)
    density wrt dm1 dq.
    """
    fracs = mixture_weights(params)

    p = component_mass_probability_m1q(
        model=model,
        component=component,
        dataset=dataset,
        params=params,
    )

    return fracs[component] * p


def m1q_to_m1m2_density(p_m1q, m1_grid, valid):
    """
    Convert density wrt dm1 dq to density wrt dm1 dm2.
    q = m2 / m1
    dq / dm2 = 1 / m1
    Therefore:
        p(m1, m2) = p(m1, q) / m1
    """
    p_m1m2 = p_m1q / np.maximum(m1_grid, 1e-300)

    p_m1m2 = np.where(
        valid & np.isfinite(p_m1m2) & (p_m1m2 > 0.0),
        p_m1m2,
        0.0,
    )

    return p_m1m2



# Evaluate a posterior draw
def evaluate_2d_density_on_m1m2_grid(model, params, m1_vals, m2_vals):
    """
    2D density p(m1, m2) on a plotting grid.
    """
    m1_grid, m2_grid = np.meshgrid(m1_vals, m2_vals, indexing="xy")    
    q_grid = m2_grid / m1_grid

    dataset = {
        "mass_1": m1_grid,
        "mass_ratio": q_grid,
    }

    valid = (
        (m1_grid > model.mmin)
        & (m1_grid < model.mmax)
        & (m2_grid > model.mmin)
        & (m2_grid <= m1_grid)
        & (q_grid > model.qmin)
        & (q_grid <= 1.0)
    )

    p_total_m1q = total_mass_probability_m1q(
        model=model,
        dataset=dataset,
        params=params,
    )

    total_m1m2 = m1q_to_m1m2_density(
        p_m1q=p_total_m1q,
        m1_grid=m1_grid,
        valid=valid,
    )

    components_m1m2 = {}

    for c in COMPONENTS:
        p_c_m1q = weighted_component_mass_probability_m1q(
            model=model,
            component=c,
            dataset=dataset,
            params=params,
        )

        components_m1m2[c] = m1q_to_m1m2_density(
            p_m1q=p_c_m1q,
            m1_grid=m1_grid,
            valid=valid,
        )

    return total_m1m2, components_m1m2


def evaluate_m1_marginals(model, params, m1_vals, n_q):
    """
    top marginal:
    p(m1) = int p(m1, q) dq
    """
    q_vals = np.linspace(model.qmin, 1.0, n_q)

    m1_grid, q_grid = np.meshgrid(m1_vals, q_vals, indexing="xy")

    dataset = {
        "mass_1": m1_grid,
        "mass_ratio": q_grid,
    }

    total_m1q = total_mass_probability_m1q(
        model=model,
        dataset=dataset,
        params=params,
    )

    total_m1 = np.trapz(total_m1q, x=q_vals, axis=0)

    component_m1 = {}

    for c in COMPONENTS:
        p_c_m1q = weighted_component_mass_probability_m1q(
            model=model,
            component=c,
            dataset=dataset,
            params=params,
        )

        component_m1[c] = np.trapz(p_c_m1q, x=q_vals, axis=0)

    return total_m1, component_m1


def evaluate_m2_marginals(model, params, m2_vals, m1_full_vals):
    """
    right marginal:
    p(m2) = int p(m1, m2) dm1
    """
    m1_grid, m2_grid = np.meshgrid(m1_full_vals, m2_vals, indexing="xy")
    q_grid = m2_grid / np.maximum(m1_grid, 1e-300)

    dataset = {
        "mass_1": m1_grid,
        "mass_ratio": q_grid,
    }

    valid = (
        (m1_grid > model.mmin)
        & (m1_grid < model.mmax)
        & (m2_grid > model.mmin)
        & (m2_grid <= m1_grid)
        & (q_grid > model.qmin)
        & (q_grid <= 1.0)
    )

    total_m1q = total_mass_probability_m1q(
        model=model,
        dataset=dataset,
        params=params,
    )

    total_m1m2 = m1q_to_m1m2_density(
        p_m1q=total_m1q,
        m1_grid=m1_grid,
        valid=valid,
    )

    total_m2 = np.trapz(total_m1m2, x=m1_full_vals, axis=1)

    component_m2 = {}

    for c in COMPONENTS:
        p_c_m1q = weighted_component_mass_probability_m1q(
            model=model,
            component=c,
            dataset=dataset,
            params=params,
        )

        p_c_m1m2 = m1q_to_m1m2_density(
            p_m1q=p_c_m1q,
            m1_grid=m1_grid,
            valid=valid,
        )

        component_m2[c] = np.trapz(p_c_m1m2, x=m1_full_vals, axis=1)

    return total_m2, component_m2



# Build distributions from the posterior samples
def build_distributions(
    posterior,
    model,
    m1_vals,
    m2_vals,
    m1_full_vals,
    n_q_marginal,
    n_draws,
    seed,
):
    rng = np.random.default_rng(seed)

    n_post = len(posterior)
    n_draws = min(n_draws, n_post)

    indices = rng.choice(n_post, size=n_draws, replace=False)

    curves = {
        "total_2d": [],
        "total_m1": [],
        "total_m2": [],
        "component_2d": {c: [] for c in COMPONENTS},
        "component_m1": {c: [] for c in COMPONENTS},
        "component_m2": {c: [] for c in COMPONENTS},
    }

    skipped = 0

    for idx in tqdm.tqdm(indices, desc="Building correct m1-m2 figure"):
        try:
            params = params_from_row(posterior.iloc[idx])

          
            total_2d, component_2d = evaluate_2d_density_on_m1m2_grid(
                model=model,
                params=params,
                m1_vals=m1_vals,
                m2_vals=m2_vals,
            )

            total_m1, component_m1 = evaluate_m1_marginals(
                model=model,
                params=params,
                m1_vals=m1_vals,
                n_q=n_q_marginal,
            )

            total_m2, component_m2 = evaluate_m2_marginals(
                model=model,
                params=params,
                m2_vals=m2_vals,
                m1_full_vals=m1_full_vals,
            )
            curves["total_2d"].append(np.asarray(total_2d, dtype=np.float32))
  
            curves["total_m1"].append(total_m1)
            curves["total_m2"].append(total_m2)

            for c in COMPONENTS:
              
                curves["component_2d"][c].append(np.asarray(component_2d[c], dtype=np.float32))
                curves["component_m1"][c].append(component_m1[c])
                curves["component_m2"][c].append(component_m2[c])

        except Exception as exc:
            skipped += 1
            print(f"Skipping sample {idx}: {exc}")

    print("Skipped samples:", skipped)

    if skipped == n_draws:
        raise RuntimeError("All samples skipped.")


    # compute median ,5% and 95% of the distribution
    summaries = {
        "total_2d": summarize(curves["total_2d"]),
        "total_m1": summarize(curves["total_m1"]),
        "total_m2": summarize(curves["total_m2"]),
        "component_2d": {
            c: summarize(curves["component_2d"][c])
            for c in COMPONENTS
        },
        "component_m1": {
            c: summarize(curves["component_m1"][c])
            for c in COMPONENTS
        },
        "component_m2": {
            c: summarize(curves["component_m2"][c])
            for c in COMPONENTS
        },
    }

    return summaries




def m1m2_figure(distributions_dict_summary, m1_vals, m2_vals, outdir, dots=None):
    os.makedirs(outdir, exist_ok=True)

    p2d = distributions_dict_summary["total_2d"]["median"]
    # for plotting, take 5 order of magnitude in the plots 
    floor_2d = positive_floor(p2d, factor=1e-5)
    z = np.log10(np.maximum(p2d, floor_2d))

    with plt.rc_context(PLOT_STYLE):
        fig = plt.figure(figsize=(10, 10))

        gs = fig.add_gridspec(
            2,
            2,
            width_ratios=(4.0, 1.32),
            height_ratios=(1.32, 4.0),
            left=0.10,
            right=0.98,
            bottom=0.09,
            top=0.96,
            wspace=0.06,
            hspace=0.06,
        )

        ax_top = fig.add_subplot(gs[0, 0])
        ax_main = fig.add_subplot(gs[1, 0])
        ax_right = fig.add_subplot(gs[1, 1], sharey=ax_main)
        
        ax_legend = fig.add_subplot(gs[0, 1])
        ax_legend.axis("off")

       
        # Main 2D density       
        im = ax_main.pcolormesh(
            m1_vals,
            m2_vals,
            z,
            shading="auto",
            cmap="Greys",
            rasterized=True,
        )


        for c in COMPONENTS:
            pc =distributions_dict_summary["component_2d"][c]["median"]
            # making the line to emphsis the subpopulation location 
            levels_c = relative_levels(pc, fractions=(0.25,))

            if levels_c is None:
                continue

            st = STYLE[c]

            ax_main.contour(
                m1_vals,
                m2_vals,
                pc,
                levels=levels_c,
                colors=st["color"],
                linestyles=st["ls"],
                linewidths=1.6,
                alpha=0.95,
            )
        # just a diagonal line to make sure m1>m2 visually      
        ax_main.plot(
            [m1_vals[0], m1_vals[-1]],
            [m1_vals[0], m1_vals[-1]],
            color="0.35",
            lw=1.0,
            ls=":",
        )
        
        # add the pi-stroke if file is there         
        if dots is not None:
            for c in COMPONENTS:
                mask = dots["cluster"] == c
        
                if np.sum(mask) == 0:
                    continue
        
                st = STYLE[c]        
                ax_main.scatter(
                    dots["m1"][mask],
                    dots["m2"][mask],
                    s=10.0 + 45.0 * np.sqrt(dots["w_plot"][mask]),
                    c=st["color"],
                    marker="o",
                    edgecolors="k",
                    linewidths=0.25,
                    alpha=0.55,
                    label="_nolegend_",
                    zorder=70,
                )

        ax_main.set_xlim(m1_vals[0], m1_vals[-1])
        ax_main.set_ylim(m2_vals[0], m2_vals[-1])
        ax_main.set_xlabel(r"$m_1\,[M_\odot]$")
        ax_main.set_ylabel(r"$m_2\,[M_\odot]$")
        ax_main.grid(True, alpha=0.12)

     
        # Top marginal: p(m1) = int p(m1,q)dq        
        total_m1 = distributions_dict_summary["total_m1"]
        # take 4 0rder of magnitude 
        floor_m1 = positive_floor(
            total_m1["median"],
            *[distributions_dict_summary["component_m1"][c]["median"] for c in COMPONENTS],
            factor=1e-4,
        )
        
        
        
        ymax_m1 = np.nanmax(total_m1["high"])
        
        # plot the total C.I. band 

        ax_top.fill_between(
            m1_vals,
            np.maximum(total_m1["low"], floor_m1),
            np.maximum(total_m1["high"], floor_m1),
            color="0.80",
            alpha=0.80,
            linewidth=0.0,
        )

        ax_top.plot(
            m1_vals,
            np.maximum(total_m1["median"], floor_m1),
            color="k",
            lw=2.2,
            label="Total",
        )
        
        # plot the median not to clutter the plot too much, its already challanging

        for c in COMPONENTS:
            st = STYLE[c]
            s = distributions_dict_summary["component_m1"][c]

            ax_top.plot(
                m1_vals,
                np.maximum(s["median"], floor_m1),
                color=st["color"],
                ls=st["ls"],
                lw=st["lw"],
                label=COMPONENT_LABELS[c],
            )

            ymax_m1 = max(ymax_m1, np.nanmax(s["median"]))
            
        if dots is not None:
            bins_m1 = np.linspace(m1_vals[0], m1_vals[-1], 45)
            total_w = np.sum(dots["w"])
    
            for c in COMPONENTS:
                mask = dots["cluster"] == c
    
                if np.sum(mask) == 0 or total_w <= 0.0:
                    continue
    
                st = STYLE[c]
    
                _, hist_m1, edges_m1 = weighted_histogram_density(
                    dots["m1"][mask],
                    dots["w"][mask] / total_w,
                    bins=bins_m1,
                )
    
                hist_m1 = np.maximum(hist_m1, floor_m1)
    
                ax_top.stairs(
                    hist_m1,
                    edges_m1,
                    baseline=floor_m1,
                    fill=True,
                    color=st["color"],
                    alpha=0.16,
                    linewidth=0.0,
                    zorder=40, # this just place the order of teh elemnts, was trying to make it visually pleasing, not sure i did a good job   
                )
    
                ax_top.stairs(
                    hist_m1,
                    edges_m1,
                    baseline=floor_m1,
                    fill=False,
                    color=st["color"],
                    alpha=0.75,
                    linewidth=1.0,
                    linestyle=st["ls"],
                    zorder=65,
                )
    
                ymax_m1 = max(ymax_m1, np.nanmax(hist_m1))    

        ax_top.set_yscale("log")
        ax_top.set_ylim(floor_m1, 1.5 * ymax_m1)
        ax_top.set_xlim(m1_vals[0], m1_vals[-1])
        ax_top.set_ylabel(r"$p(m_1)$")
        ax_top.tick_params(labelbottom=False)
        ax_top.grid(True, alpha=0.12, which="both")

        
        # same as above just the right marginal: p(m2) = int p(m1,m2)dm1
       

        total_m2 = distributions_dict_summary["total_m2"]

        floor_m2 = positive_floor(
            total_m2["median"],
            *[distributions_dict_summary["component_m2"][c]["median"] for c in COMPONENTS],
            factor=1e-4,
        )

        xmax_m2 = np.nanmax(total_m2["high"])

        ax_right.fill_betweenx(
            m2_vals,
            np.maximum(total_m2["low"], floor_m2),
            np.maximum(total_m2["high"], floor_m2),
            color="0.80",
            alpha=0.80,
            linewidth=0.0,
        )

        ax_right.plot(
            np.maximum(total_m2["median"], floor_m2),
            m2_vals,
            color="k",
            lw=2.2,
            label="Total",
        )

        for c in COMPONENTS:
            st = STYLE[c]
            s = distributions_dict_summary["component_m2"][c]

            ax_right.plot(
                np.maximum(s["median"], floor_m2),
                m2_vals,
                color=st["color"],
                ls=st["ls"],
                lw=st["lw"],
                label=COMPONENT_LABELS[c],
            )

            xmax_m2 = max(xmax_m2, np.nanmax(s["median"]))
            
        if dots is not None:
            bins_m2 = np.linspace(m2_vals[0], m2_vals[-1], 45)
            total_w = np.sum(dots["w"])
    
            for c in COMPONENTS:
                mask = dots["cluster"] == c
    
                if np.sum(mask) == 0 or total_w <= 0.0:
                    continue
    
                st = STYLE[c]
    
                _, hist_m2, edges_m2 = weighted_histogram_density(
                    dots["m2"][mask],
                    dots["w"][mask] / total_w,
                    bins=bins_m2,
                )
    
                hist_m2 = np.maximum(hist_m2, floor_m2)
    
                y_step = np.repeat(edges_m2, 2)[1:-1]
                x_step = np.repeat(hist_m2, 2)
    
                ax_right.fill_betweenx(
                    y_step,
                    floor_m2,
                    x_step,
                    color=st["color"],
                    alpha=0.16,
                    linewidth=0.0,
                    zorder=40,
                )
    
                ax_right.plot(
                    x_step,
                    y_step,
                    color=st["color"],
                    alpha=0.75,
                    linewidth=1.0,
                    linestyle=st["ls"],
                    zorder=65,
                )
    
                xmax_m2 = max(xmax_m2, np.nanmax(hist_m2))    
                
        ax_right.set_xscale("log")
        ax_right.set_xlim(floor_m2, 1.5 * xmax_m2)
        ax_right.set_xlabel(r"$p(m_2)$")
        ax_right.tick_params(labelleft=False)
        ax_right.grid(True, alpha=0.12, which="both")

       
        # Legend
       

        handles, labels = ax_top.get_legend_handles_labels()

        if dots is not None:
            pistroke_handle = Line2D(
                [0],
                [0],
                marker="o",
                linestyle="none",
                markerfacecolor="0.55",
                markeredgecolor="k",
                markeredgewidth=0.4,
                markersize=7,
                alpha=0.65,
                label="π\u0336",
            )
        
            handles.append(pistroke_handle)
            labels.append("π\u0336")
        
        ax_legend.legend(
            handles,
            labels,
            loc="center",
            frameon=False,
            ncol=1,
            handlelength=2.4,
            handletextpad=0.7,
            borderaxespad=0.0,
            labelspacing=0.55,
            fontsize=12,
        )
        # so its not too heavy for a quick look 
        png_path = os.path.join(outdir, "m1_m2_density.png")
        # paper image 
        pdf_path = os.path.join(outdir, "mass_corner_simple.pdf")

        fig.savefig(png_path, dpi=350)
        fig.savefig(pdf_path, dpi=350)
        plt.close(fig)

    print("Saved:", png_path)
    print("Saved:", pdf_path)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--result",
        default="gwtc5_result.hdf5",
    )
    
    parser.add_argument("--outdir", default="figures")
    parser.add_argument("--pistroke-file",default="pi_stroke_results_mass_1_mass_2.pkl", help="pi-stroke pickle result file. Use 'none' to skip.",
    )

    parser.add_argument("--n-draws", type=int, default=100)
    parser.add_argument("--seed", type=int, default=1234)

    parser.add_argument("--plot-m1-min", type=float, default=5.0)
    parser.add_argument("--plot-m1-max", type=float, default=70.0)
    parser.add_argument("--plot-m2-min", type=float, default=2.0)
    parser.add_argument("--plot-m2-max", type=float, default=70.0)

    parser.add_argument("--n-m1", type=int, default=360)
    parser.add_argument("--n-m2", type=int, default=320)

    parser.add_argument("--n-q-marginal", type=int, default=600)
    parser.add_argument("--n-m1-full", type=int, default=700)
  

    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    posterior = load_bilby_posterior(args.result)
    dots = None

    if args.pistroke_file.lower() != "none":
        pts_xy, w_opt, x_ml, w_ml, var_ml = load_results(args.pistroke_file)
        # do some pre processing to teh pi-storke results 
        dots = make_pistroke_dots_from_results(
            pts_xy=pts_xy,
            w_opt=w_opt,
        )

    model = PiStrokeFourComponentPhysicalSpin()

    m1_vals = np.linspace(args.plot_m1_min, args.plot_m1_max, args.n_m1)
    m2_vals = np.linspace(args.plot_m2_min, args.plot_m2_max, args.n_m2)

    # Full m1 grid for p(m2), so right marginal is not truncated by x-axis.
    m1_full_vals = np.linspace(model.mmin, model.mmax, args.n_m1_full)

 
    distributions_summary_dict = build_distributions(
            posterior=posterior,
            model=model,
            m1_vals=m1_vals,
            m2_vals=m2_vals,
            m1_full_vals=m1_full_vals,
            n_q_marginal=args.n_q_marginal,
            n_draws=args.n_draws,
            seed=args.seed,
    )
    
     
    m1m2_figure(
        distributions_dict_summary=distributions_summary_dict,
        m1_vals=m1_vals,
        m2_vals=m2_vals,
        outdir=args.outdir,
        dots=dots,
    )
    
    
   

if __name__ == "__main__":
    main()
