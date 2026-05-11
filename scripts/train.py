import numpy as np
import pandas as pd
import torch
import yaml

from argparse import ArgumentParser
from pathlib import Path
from scipy.special import logit as scipy_logit
from FlowStorm.flow_surface_torch import FlowSurface


TRANSFORMS = {
    "log":    np.log,
    "log10":  np.log10,
    "log1p":  np.log1p,
    "cos":    np.cos,
    "sin":    np.sin,
    "sqrt":   np.sqrt,
    "abs":    np.abs,
    "logit":  scipy_logit,
    "neg":    np.negative,
}


def make_parser():
    parser = ArgumentParser()
    parser.add_argument("--config",   required=True, type=str)
    parser.add_argument("--save_dir", default=None,  type=str,
                        help="Override save_dir from config")
    return parser


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def load_dataframe(data_cfg: dict) -> pd.DataFrame:
    path = data_cfg["path"]
    fmt  = data_cfg.get("format", Path(path).suffix.lstrip("."))
    if fmt in ("parquet", "pq"):
        return pd.read_parquet(path)
    elif fmt in ("hdf", "h5", "hdf5"):
        return pd.read_hdf(path, key=data_cfg.get("hdf_key", "data"))
    else:
        raise ValueError(f"Unsupported data format: {fmt!r}")


def parse_input_spec(spec: list) -> list[tuple[str, str | None, str]]:
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


def apply_transforms(df: pd.DataFrame, input_spec: list) -> np.ndarray:
    columns = []
    for raw_col, transform, name in input_spec:
        values = df[raw_col].to_numpy(dtype=np.float64)
        if transform is not None:
            if transform not in TRANSFORMS:
                raise ValueError(f"Unknown transform {transform!r} for column {raw_col!r}. "
                                 f"Available: {list(TRANSFORMS)}")
            values = TRANSFORMS[transform](values)
        columns.append(values)
    return np.stack(columns, axis=1).astype(np.float32)


def compute_weights(df: pd.DataFrame, weight_cfg: dict) -> np.ndarray:
    baseline = df[weight_cfg["baseline_col"]].to_numpy()
    scheme   = weight_cfg.get("scheme", "baseline")
    if scheme == "baseline":
        return baseline
    elif scheme == "e2":
        energy = df[weight_cfg["energy_col"]].to_numpy()
        return baseline * (energy / 1e5) ** (-2)
    else:
        raise ValueError(f"Unknown weighting scheme: {scheme!r}")


def build_tensors(df, input_spec, sys_params, weight_cfg):
    x_np    = apply_transforms(df, input_spec)
    x_input = torch.tensor(x_np, dtype=torch.float32)
    alphas  = torch.tensor(df[sys_params].to_numpy(), dtype=torch.float32)
    weights = torch.tensor(compute_weights(df, weight_cfg), dtype=torch.float32)

    mask = torch.isfinite(x_input).all(dim=1)
    return x_input[mask], alphas[mask], weights[mask]


def build_cartesian_dims(input_spec, az_cols, normalize_cols):
    raw_cols = [raw for raw, _, _ in input_spec]
    cartesian_az_dims = [i for i, col in enumerate(raw_cols) if col in az_cols]

    conv_idx = 0
    normalize_dims = []
    for raw_col in raw_cols:
        if raw_col in az_cols:
            conv_idx += 2
        else:
            if raw_col in normalize_cols:
                normalize_dims.append(conv_idx)
            conv_idx += 1

    return cartesian_az_dims, normalize_dims


def build_spherical_args(input_spec, az_cols, normalize_cols, spherical_cfg):
    raw_cols = [raw for raw, _, _ in input_spec]

    def _find(name):
        return raw_cols.index(name) if name in raw_cols else None

    s2_pairs = spherical_cfg.get("s2_pairs", [])
    s2_dim_pairs = [
        p for p in [(_find(a), _find(b)) for a, b in s2_pairs]
        if None not in p
    ]

    normalize_dims  = [i for i, col in enumerate(raw_cols) if col in normalize_cols]
    n_e, n_s        = spherical_cfg.get("n_e", 8), spherical_cfg.get("n_s", 8)
    n_e_sub         = len(normalize_dims)
    flow_manifold   = "+".join(["e1"] * n_e_sub + ["s2"] * len(s2_dim_pairs))
    flow_layers_str = "+".join([f"{'p'*n_e}"] * n_e_sub + [f"{'n'*n_s}"] * len(s2_dim_pairs))

    return s2_dim_pairs, normalize_dims, flow_manifold, flow_layers_str


def parse_model_config(model_cfg: dict) -> dict:
    kwargs = {}
    kwargs["seed"] = model_cfg.get("seed", 187)
    kwargs["eps"]  = float(model_cfg.get("eps", 1e-3))

    yw = model_cfg.get("yield_weights", None)
    kwargs["yield_weights"] = False if yw is False else None

    flow_cfg = model_cfg.get("flow", {})
    kwargs["flow_kwargs"] = {"flow_layers": flow_cfg.get("layers", 8)}
    if "options" in flow_cfg:
        kwargs["flow_options_overwrite"] = flow_cfg["options"]

    yield_cfg = model_cfg.get("yield_flow", {})
    kwargs["yield_kwargs"] = {"flow_layers": yield_cfg.get("layers", 16)}
    if "options" in yield_cfg:
        kwargs["yield_options_overwrite"] = yield_cfg["options"]

    return kwargs


def main():
    parser = make_parser()
    args   = parser.parse_args()

    cfg            = load_config(args.config)
    save_dir       = args.save_dir or cfg["save_dir"]
    mode           = cfg.get("mode", "cartesian")
    input_spec     = parse_input_spec(cfg["inputs"])
    sys_params     = cfg["sys_params"]
    weight_cfg     = cfg["weighting"]
    train_cfg      = cfg.get("training", {})
    az_cols        = set(cfg.get("az_cols",        []))
    normalize_cols = set(cfg.get("normalize_cols", []))
    model_kwargs   = parse_model_config(cfg.get("model", {}))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"PyTorch device: {device}")
    print(f"Mode: {mode}")
    print(f"Weighting scheme: {weight_cfg.get('scheme', 'baseline')}")
    print("Inputs:")
    for raw, transform, name in input_spec:
        t = f"  ({transform})" if transform else ""
        print(f"  {raw}{t}  ->  {name}")

    df = load_dataframe(cfg["data"])
    print(f"Loaded {len(df):,} rows from {cfg['data']['path']}")

    x_train, alphas_train, weights = build_tensors(df, input_spec, sys_params, weight_cfg)
    print(f"Training set: {x_train.shape[0]:,} events, {x_train.shape[1]} raw input dims")



    flow_manifold     = None
    flow_layers_str   = None
    normalize_dims    = None
    s2_dim_pairs      = None
    cartesian_az_dims = None

    if mode == "cartesian":
        cartesian_az_dims, normalize_dims = build_cartesian_dims(
            input_spec, az_cols, normalize_cols
        )
    elif mode == "spherical":
        s2_dim_pairs, normalize_dims, flow_manifold, flow_layers_str = build_spherical_args(
            input_spec, az_cols, normalize_cols, cfg.get("spherical", {})
        )


    flow = FlowSurface(
        x_train, alphas_train,
        weights=weights,
        yield_weights=model_kwargs.pop("yield_weights"),
        flow_manifold=flow_manifold,
        flow_layers_str=flow_layers_str,
        normalize_dims=normalize_dims,
        s2_dim_pairs=s2_dim_pairs,
        cartesian_az_dims=cartesian_az_dims,
        **model_kwargs,
    )

    print(f"Flow input dim: {flow._x_dim}  (raw: {flow._x_dim_raw})")
    print(f"Normalise dims: {flow._normalize_dims}")
    print(f"x_train — min: {flow.x_train.min():.3f}  max: {flow.x_train.max():.3f}"
          f"  nan: {flow.x_train.isnan().any()}  inf: {flow.x_train.isinf().any()}")
    print(f"alpha   — min: {flow.alpha_train.min():.3f}  max: {flow.alpha_train.max():.3f}")

    if mode == "cartesian":
        for raw_col, conv_cols in flow._raw_to_conv.items():
            if len(conv_cols) == 2:
                sin_col, cos_col = conv_cols
                _, _, name = input_spec[raw_col]
                for c, label in [(sin_col, f"sin_{name}"), (cos_col, f"cos_{name}")]:
                    v = flow.x_train[:, c]
                    print(f"  {label}: [{v.min():.3f}, {v.max():.3f}]  (expect [-1, 1])")

    flow.train_both(
        max_patience=train_cfg.get("max_patience",        10),
        max_epochs=train_cfg.get("max_epochs",           100),
        batch_size=train_cfg.get("batch_size",        131072),
        flow_learning_rate=train_cfg.get("flow_learning_rate",  3e-4),
        yield_learning_rate=train_cfg.get("yield_learning_rate", 3e-4),
        val_split=train_cfg.get("val_split", 0.1),
    )
    flow.save(save_dir)


if __name__ == "__main__":
    main()
    print("Done")