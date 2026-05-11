import numpy as onp
import pandas as pd
import torch
import yaml
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from argparse import ArgumentParser
from pathlib import Path
from scipy.special import logit as scipy_logit
from FlowStorm.flow_surface_torch import FlowSurface


TRANSFORMS = {
    "log":    onp.log,
    "log10":  onp.log10,
    "log1p":  onp.log1p,
    "cos":    onp.cos,
    "sin":    onp.sin,
    "sqrt":   onp.sqrt,
    "abs":    onp.abs,
    "logit":  scipy_logit,
    "neg":    onp.negative,
}


def make_parser():
    parser = ArgumentParser()
    parser.add_argument("--config",   required=True, type=str)
    parser.add_argument("--save_dir", default=None,  type=str,
                        help="Override save_dir from config")
    parser.add_argument("--n_samples", default=500_000, type=int,
                        help="Number of samples to draw from the flow")
    return parser


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def load_dataframe(data_cfg):
    path = data_cfg["path"]
    fmt  = data_cfg.get("format", Path(path).suffix.lstrip("."))
    if fmt in ("parquet", "pq"):
        return pd.read_parquet(path)
    elif fmt in ("hdf", "h5", "hdf5"):
        return pd.read_hdf(path, key=data_cfg.get("hdf_key", "data"))
    elif fmt == "csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported data format: {fmt!r}")


def parse_input_spec(spec):
    result = []
    for item in spec:
        if isinstance(item, str):
            result.append((item, None, item))
        else:
            col       = item["col"]
            transform = item.get("transform", None)
            name      = item.get("name", f"{transform}_{col}" if transform else col)
            result.append((col, transform, name))
    return result


def apply_transforms(df, input_spec):
    columns = []
    for raw_col, transform, _ in input_spec:
        values = df[raw_col].to_numpy(dtype=onp.float64)
        if transform is not None:
            values = TRANSFORMS[transform](values)
        columns.append(values)
    return onp.stack(columns, axis=1).astype(onp.float32)


def compute_weights(df, weight_cfg):
    baseline = df[weight_cfg["baseline_col"]].to_numpy(dtype=onp.float64)
    scheme   = weight_cfg.get("scheme", "baseline")
    if scheme == "baseline":
        return baseline
    elif scheme == "e2":
        energy = df[weight_cfg["energy_col"]].to_numpy(dtype=onp.float64)
        return baseline * (energy / 1e5) ** (-2)
    raise ValueError(f"Unknown weighting scheme: {scheme!r}")


# ── Sampling helpers ───────────────────────────────────────────────────────────

def sample_alphas(flow, n):
    """Draw n samples from the yield flow in raw alpha space."""
    with torch.no_grad():
        a_model = flow.sample_alpha(n).cpu().numpy()

    # if all(hasattr(flow, k) for k in ["alpha_min", "alpha_max", "_eps"]):
    #     eps = float(flow._eps)
    #     u   = (a_model - eps) / (1.0 - 2.0 * eps)
    #     lo  = onp.asarray(flow.alpha_min.cpu())
    #     hi  = onp.asarray(flow.alpha_max.cpu())
    #     return lo + u * (hi - lo)

    # if hasattr(flow, "alpha_std") and hasattr(flow, "alpha_mean"):
    #     a_t = a_model * onp.asarray(flow.alpha_std) + onp.asarray(flow.alpha_mean)
    # else:
    #     a_t = a_model

    # if hasattr(flow, "alpha_lo") and flow.alpha_lo is not None:
    #     lo = onp.asarray(flow.alpha_lo)
    #     hi = onp.asarray(flow.alpha_hi)
    #     u  = 1.0 / (1.0 + onp.exp(-a_t))
    #     return lo + (hi - lo) * u

    return a_model


def sample_x(flow, alphas_raw):
    """Draw one x sample per row in alphas_raw (raw space), return in transformed space."""
    alpha_t = torch.tensor(alphas_raw, dtype=torch.float32)
    with torch.no_grad():
        x_model = flow.sample_x(len(alphas_raw), enforce_bounds=False).cpu().numpy()
    return x_model


# ── 1-D marginal plot ──────────────────────────────────────────────────────────

def plot_marginals_1d(data_vals, flow_vals, labels, weights, n_cols, plot_path, title, log=False):
    n = data_vals.shape[1]
    n_rows = (n + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3.5 * n_rows))
    axes = onp.array(axes).flatten()

    for i in range(n):
        ax   = axes[i]

        if log:
            ax.set_yscale("log")

        dv   = data_vals[:, i]
        fv   = flow_vals[:, i]
        vmin = onp.nanpercentile(onp.concatenate([dv, fv]), 0.1)
        vmax = onp.nanpercentile(onp.concatenate([dv, fv]), 99.9)
        bins = onp.linspace(vmin, vmax, 51)

        d_counts, edges = onp.histogram(dv, bins=bins, weights=weights, density=False)
        binw    = onp.diff(edges)
        N_d     = d_counts.sum()
        d_dens  = d_counts / (N_d * binw)

        f_counts, _ = onp.histogram(fv, bins=bins, density=False)
        N_f     = f_counts.sum()
        f_dens  = f_counts / (N_f * binw)

        centers = 0.5 * (edges[1:] + edges[:-1])
        ax.step(centers, f_dens, where="mid", label="flow")
        ax.step(centers, d_dens, where="mid", label="data", color="C1")

        # pull panel
        mu    = f_dens * N_d * binw
        sigma = onp.sqrt(onp.maximum(d_counts, 1.0))
        pull  = (d_counts - mu) / sigma

        ax2 = ax.twinx()
        ax2.errorbar(centers, pull, yerr=onp.ones_like(pull),
                     fmt=".", ms=2, capsize=1, color="grey", alpha=0.6)
        ax2.axhline(0.0, linewidth=0.8, color="grey", linestyle="--")
        ax2.set_ylim(-5, 5)
        ax2.set_ylabel("pull", fontsize=7, color="grey")
        ax2.tick_params(labelsize=6, colors="grey")

        ax.set_xlabel(labels[i], fontsize=8)
        ax.set_ylabel("density",  fontsize=8)
        if i == 0:
            ax.legend(fontsize=7)

    for j in range(n, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {plot_path}")


# ── Corner plot ────────────────────────────────────────────────────────────────

def plot_corner(data_vals, flow_vals, labels, weights, plot_path):
    n    = data_vals.shape[1]
    fig, axes = plt.subplots(n, n, figsize=(2.8 * n, 2.8 * n))

    bins_1d = 40
    bins_2d = 30

    # percentile-based ranges per dimension
    ranges = []
    for i in range(n):
        combined = onp.concatenate([data_vals[:, i], flow_vals[:, i]])
        ranges.append((
            onp.nanpercentile(combined, 0.1),
            onp.nanpercentile(combined, 99.9),
        ))

    for row in range(n):
        for col in range(n):
            ax = axes[row, col]

            if col > row:
                ax.set_visible(False)
                continue
            


            xr = ranges[col]
            yr = ranges[row]

            if row == col:
                bins = onp.linspace(*xr, bins_1d + 1)
                binw = onp.diff(bins)
                centers = 0.5 * (bins[1:] + bins[:-1])

                d_c, _ = onp.histogram(data_vals[:, col], bins=bins,
                                       weights=weights, density=False)
                d_c    = d_c / (d_c.sum() * binw)
                f_c, _ = onp.histogram(flow_vals[:, col], bins=bins, density=False)
                f_c    = f_c / (f_c.sum() * binw)

                ax.step(centers, f_c, where="mid", color="C0", linewidth=1.0)
                ax.step(centers, d_c, where="mid", color="C1", linewidth=1.0)
                ax.set_xlim(*xr)

            else:
                xbins = onp.linspace(*xr, bins_2d + 1)
                ybins = onp.linspace(*yr, bins_2d + 1)

                # data: filled contours
                d_h, _, _ = onp.histogram2d(
                    data_vals[:, col], data_vals[:, row],
                    bins=[xbins, ybins], weights=weights, density=True,
                )
                d_h = d_h.T
                xc  = 0.5 * (xbins[1:] + xbins[:-1])
                yc  = 0.5 * (ybins[1:] + ybins[:-1])
                lvls_d = _contour_levels(d_h, [0.68, 0.95])
                ax.contourf(xc, yc, d_h, levels=[lvls_d[1], lvls_d[0], d_h.max() * 1.01],
                            colors=["C1"], alpha=[0.15, 0.30])
                ax.contour(xc, yc, d_h, levels=lvls_d, colors=["C1"], linewidths=0.8)

                # flow: contour lines only
                f_h, _, _ = onp.histogram2d(
                    flow_vals[:, col], flow_vals[:, row],
                    bins=[xbins, ybins], density=True,
                )
                f_h    = f_h.T
                lvls_f = _contour_levels(f_h, [0.68, 0.95])
                ax.contour(xc, yc, f_h, levels=lvls_f, colors=["C0"], linewidths=0.8)

                ax.set_xlim(*xr)
                ax.set_ylim(*yr)

            # axis labels only on edges
            if row == n - 1:
                ax.set_xlabel(labels[col], fontsize=7)
            else:
                ax.set_xticklabels([])
            if col == 0 and row != 0:
                ax.set_ylabel(labels[row], fontsize=7)
            else:
                ax.set_yticklabels([])

            ax.tick_params(labelsize=6)

    # legend patches
    import matplotlib.patches as mpatches
    legend_elements = [
        mpatches.Patch(color="C1", label="data"),
        mpatches.Patch(color="C0", label="flow"),
    ]
    fig.legend(handles=legend_elements, loc="upper right", fontsize=8)
    fig.suptitle("Input corner plot  (68% / 95% contours)", fontsize=10, y=1.01)
    fig.tight_layout()
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {plot_path}")


def _contour_levels(h, fractions):
    """Return histogram thresholds enclosing `fractions` of the probability mass."""
    h_flat = onp.sort(h.ravel())[::-1]
    cumsum  = onp.cumsum(h_flat)
    cumsum /= cumsum[-1]
    levels  = []
    for f in fractions:
        idx = onp.searchsorted(cumsum, f)
        levels.append(float(h_flat[min(idx, len(h_flat) - 1)]))
    return sorted(levels)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = make_parser()
    args   = parser.parse_args()

    cfg        = load_config(args.config)
    save_dir   = Path(args.save_dir or cfg["save_dir"])
    plot_dir   = save_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    input_spec   = parse_input_spec(cfg["inputs"])
    sys_params   = cfg["sys_params"]
    weight_cfg   = cfg["weighting"]
    input_labels = [name for _, _, name in input_spec]

    df = load_dataframe(cfg["data"])
    print(f"Loaded {len(df):,} rows from {cfg['data']['path']}")

    data_x       = apply_transforms(df, input_spec)
    data_alphas  = df[sys_params].to_numpy(dtype=onp.float32)
    data_weights = compute_weights(df, weight_cfg).astype(onp.float32)

    # finite mask
    mask       = onp.isfinite(data_x).all(axis=1)
    data_x     = data_x[mask]
    data_alphas  = data_alphas[mask]
    data_weights = data_weights[mask]

    flow = FlowSurface.load(str(save_dir))
    print(f"Loaded FlowSurface from {save_dir}")

    print(f"Sampling {args.n_samples:,} events from flow...")
    flow_alphas = sample_alphas(flow, args.n_samples)
    flow_x      = sample_x(flow, flow_alphas)

    print("Plotting alpha marginals...")
    plot_marginals_1d(
        data_alphas, flow_alphas,
        labels=sys_params,
        weights=data_weights/data_weights,
        n_cols=3,
        plot_path=plot_dir / "alpha_marginals.png",
        title="Systematic parameter marginals",
    )

    print("Plotting input marginals...")
    plot_marginals_1d(
        data_x, flow_x,
        labels=input_labels,
        weights=data_weights,
        n_cols=3,
        plot_path=plot_dir / "input_marginals.png",
        title="Input variable marginals",
        log=True,
    )

    # print("Plotting input corner plot...")
    # plot_corner(
    #     data_x, flow_x,
    #     labels=input_labels,
    #     weights=data_weights,
    #     plot_path=plot_dir / "input_corner.png",
    # )


if __name__ == "__main__":
    main()
    print("Done")