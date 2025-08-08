#!/usr/bin/env python3
"""
Confidence (uncertainty) estimation for cross-platform VAE imputation.

Two methods are supported (select via --method):
- mc (default): Run K imputations with stochastic sampling and compute per-cell
  CV = exp(sqrt(var)) - 1 from the repeated outputs (in original target scale).
- delta: Single-pass analytical approximation using the delta method to
  propagate latent variance through the decoder. For each sample, we compute
  output variance diag(J Σ_z J^T), where J is the Jacobian of the target
  decoder at z=μ, and Σ_z is diag(exp(logvar)). Optionally scaled to original
  target space if scaler supports scale_.

Notes:
- Project data is already in log space; we do not apply log-domain adjustments
  when mapping variance to original scale.
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, Tuple, Optional, List

import numpy as np
import pandas as pd
import torch
import yaml

# Ensure local src/ is importable
sys.path.append(str(Path(__file__).parent.parent / "src"))

from models import (  # noqa: E402
    JointVAELightning,
    JointVAEPlusLightning,
    DirectImputationLightning,
    GenerativeVAE,
    JointIWAELightning,
    JointVAEVampPriorLightning,
    JointVQLightning,
)
from utils import (  # noqa: E402
    load_scalers_and_features,
    preprocess_input_data,
    inverse_transform_output,
)
from utils.scaler_utils import (  # noqa: E402
    validate_scalers_for_inference,
)
from torch.autograd.functional import vjp, jvp  # noqa: E402


def find_best_checkpoint(exp_dir: str) -> str:
    """Find the best checkpoint file in the experiment directory."""
    checkpoint_dir = Path(exp_dir) / "checkpoints"

    # Look for best checkpoint first
    best_checkpoints = list(checkpoint_dir.glob("*best*.ckpt"))
    if best_checkpoints:
        return str(best_checkpoints[0])

    # Look for last checkpoint
    last_checkpoints = list(checkpoint_dir.glob("last.ckpt"))
    if last_checkpoints:
        return str(last_checkpoints[0])

    # Fallback to latest checkpoint
    all_checkpoints = sorted(checkpoint_dir.glob("*.ckpt"))
    if all_checkpoints:
        return str(all_checkpoints[-1])

    raise FileNotFoundError(f"No checkpoint files found in {checkpoint_dir}")


def load_config_file(config_path: str) -> Dict:
    """Load configuration from YAML file, handling nested PL hparams format."""
    with open(config_path, "r") as f:
        data = yaml.safe_load(f)

    if "config" in data and isinstance(data["config"], dict):
        print("Detected PyTorch Lightning hparams format, extracting config...")
        return data["config"]

    return data


def load_model(model_path: str, config: Dict) -> torch.nn.Module:
    """Load the trained model from checkpoint using model_type from config."""
    model_type = config["model"].get("model_type", "joint_vae")

    model_classes = {
        "joint_vae": JointVAELightning,
        "joint_vae_plus": JointVAEPlusLightning,
        "JointVAEVampPrior": JointVAEVampPriorLightning,
        "JointIWAE": JointIWAELightning,
        "JointVQ": JointVQLightning,
        "res_unet": DirectImputationLightning,
        "generative_vae": GenerativeVAE,
    }

    model_class = model_classes.get(model_type)
    if not model_class:
        raise ValueError(f"Unsupported model type: {model_type}")

    print(f"Loading {model_type} model from checkpoint...")
    model = model_class.load_from_checkpoint(model_path)
    model.eval()
    return model


@torch.no_grad()
def impute_once(
    model: torch.nn.Module,
    input_data: np.ndarray,
    source_platform: str,
    target_platform: str,
) -> np.ndarray:
    """Perform a single stochastic imputation pass using the trained model."""
    input_tensor = torch.as_tensor(input_data, dtype=torch.float32)

    model_device = next(model.parameters()).device
    input_tensor = input_tensor.to(model_device)

    # The underlying Lightning wrappers store the core model on .model
    core = model.model if hasattr(model, "model") else model

    # Prefer explicit imputation helpers when available
    if (
        hasattr(core, "impute_a_to_b")
        and source_platform == "a"
        and target_platform == "b"
    ):
        output_tensor = core.impute_a_to_b(input_tensor)
    elif (
        hasattr(core, "impute_b_to_a")
        and source_platform == "b"
        and target_platform == "a"
    ):
        output_tensor = core.impute_b_to_a(input_tensor)
    else:
        # Fallback: forward with dummy counterpart
        if source_platform == "a":
            input_dim_b = getattr(core, "input_dim_b", input_tensor.shape[1])
            dummy_b = torch.zeros(input_tensor.shape[0], input_dim_b, device=model_device)
            outputs = core(input_tensor, dummy_b)
            # Try common keys for cross-reconstruction
            for key in ("cross_recon_b", "recon_b", "x_recon_b"):
                if isinstance(outputs, dict) and key in outputs:
                    output_tensor = outputs[key]
                    break
            else:
                # As a last resort, try decoding a latent if present
                z = None
                for key in ("z", "latent", "mean_a"):
                    if isinstance(outputs, dict) and key in outputs:
                        z = outputs[key]
                        break
                if z is not None and hasattr(core, "decode_b"):
                    output_tensor = core.decode_b(z)
                else:
                    raise RuntimeError("Model does not expose expected imputation outputs")
        else:
            input_dim_a = getattr(core, "input_dim_a", input_tensor.shape[1])
            dummy_a = torch.zeros(input_tensor.shape[0], input_dim_a, device=model_device)
            outputs = core(dummy_a, input_tensor)
            for key in ("cross_recon_a", "recon_a", "x_recon_a"):
                if isinstance(outputs, dict) and key in outputs:
                    output_tensor = outputs[key]
                    break
            else:
                z = None
                for key in ("z", "latent", "mean_b"):
                    if isinstance(outputs, dict) and key in outputs:
                        z = outputs[key]
                        break
                if z is not None and hasattr(core, "decode_a"):
                    output_tensor = core.decode_a(z)
                else:
                    raise RuntimeError("Model does not expose expected imputation outputs")

    return output_tensor.detach().cpu().numpy()


def compute_cv_from_runs(imputed_runs: List[np.ndarray]) -> np.ndarray:
    """Compute per-cell CV = exp(sqrt(var)) - 1 across repeated imputations.

    Args:
        imputed_runs: list of arrays with shape (n_samples, n_features)

    Returns:
        cv_matrix: array with shape (n_samples, n_features)
    """
    stack = np.stack(imputed_runs, axis=0)  # (n_runs, n_samples, n_features)
    var = np.var(stack, axis=0, ddof=0)
    with np.errstate(over="ignore", invalid="ignore"):
        cv = np.exp(np.sqrt(var)) - 1.0
        cv[~np.isfinite(cv)] = 0.0
    return cv


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Estimate imputation confidence via repeated VAE sampling"
    )

    parser.add_argument(
        "--experiment_dir",
        type=str,
        required=True,
        help="Path to experiment directory containing config, checkpoint, and preprocessing artifacts",
    )
    parser.add_argument(
        "--input_data",
        type=str,
        required=True,
        help="Path to input data file to be imputed (CSV or TXT)",
    )
    parser.add_argument(
        "--source_platform",
        type=str,
        choices=["a", "b"],
        required=True,
        help="Source platform of input data (a or b)",
    )
    parser.add_argument(
        "--target_platform",
        type=str,
        choices=["a", "b"],
        required=True,
        help="Target platform for imputation (a or b)",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path for output CSV (per-cell confidence/CV)",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Specific checkpoint file to use (if not provided, will find best checkpoint)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Specific config file to use (if not provided, will search for config files in experiment_dir)",
    )
    parser.add_argument(
        "--id_column",
        type=str,
        default=None,
        help="Name of ID column in input data (auto-detected if not provided)",
    )
    parser.add_argument(
        "--n_runs",
        type=int,
        default=5,
        help="Number of repeated imputations to estimate variability",
    )
    parser.add_argument(
        "--method",
        type=str,
        choices=["mc", "delta", "latent"],
        default="mc",
        help="Confidence estimation method: 'mc' (repeated sampling), 'delta' (latent variance propagation), or 'latent' (direct latent variance)",
    )
    parser.add_argument(
        "--delta_backend",
        type=str,
        choices=["exact", "hutchinson"],
        default="hutchinson",
        help="Backend for delta method: 'exact' full Jacobian (very slow) or 'hutchinson' estimator (fast)",
    )
    parser.add_argument(
        "--delta_probes",
        type=int,
        default=16,
        help="Number of probe vectors for Hutchinson estimator (delta backend)",
    )

    return parser.parse_args()


def main():
    args = parse_arguments()

    if args.source_platform == args.target_platform:
        raise ValueError("Source and target platforms must be different")

    exp_dir = Path(args.experiment_dir)
    if not exp_dir.exists():
        raise FileNotFoundError(f"Experiment directory not found: {exp_dir}")

    # Locate config
    if args.config:
        config_path = Path(args.config)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
    else:
        candidates = (
            list(exp_dir.glob("config.yaml"))
            + list(exp_dir.glob("config.yml"))
            + list(exp_dir.glob("*_config.yaml"))
            + list(exp_dir.glob("*_config.yml"))
            + list(exp_dir.glob("hparams.yaml"))
            + list(exp_dir.glob("*.yaml"))
            + list(exp_dir.glob("*.yml"))
        )
        if not candidates:
            raise FileNotFoundError(f"No config file found in {exp_dir}")
        config_path = candidates[0]

    print(f"Loading configuration from {config_path}")
    config = load_config_file(str(config_path))

    # Load preprocessing artifacts
    print("Loading preprocessing artifacts...")
    scalers, feature_names, log_transform_params = load_scalers_and_features(
        experiment_dir=str(exp_dir)
    )
    if scalers is None or feature_names is None or log_transform_params is None:
        raise ValueError(
            "Could not load preprocessing artifacts (scalers, feature_names, log_transform_params)."
        )

    # Checkpoint
    if args.checkpoint:
        checkpoint_path = args.checkpoint
        if not Path(checkpoint_path).exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    else:
        checkpoint_path = find_best_checkpoint(str(exp_dir))
    print(f"Using checkpoint: {checkpoint_path}")

    # Load model
    model = load_model(checkpoint_path, config)
    model.eval()

    # Load input data
    print(f"Loading input data from {args.input_data}")
    sep_input = "\t" if Path(args.input_data).suffix.lower() == ".txt" else ","
    input_df = pd.read_csv(args.input_data, sep=sep_input)
    print(f"Input data shape: {input_df.shape}")

    # Preprocess
    source_platform_key = f"platform_{args.source_platform}"
    target_platform_key = f"platform_{args.target_platform}"
    preprocessed_data, id_column = preprocess_input_data(
        input_df, config, scalers, log_transform_params, source_platform_key
    )
    print(f"Preprocessed data shape: {preprocessed_data.shape}")

    method = args.method
    if method == "mc":
        # Run multiple imputations
        print(f"Running {args.n_runs} imputation passes ({args.source_platform} -> {args.target_platform})...")
        imputed_original_runs: List[np.ndarray] = []
        for run_idx in range(args.n_runs):
            imputed_norm = impute_once(
                model, preprocessed_data, args.source_platform, args.target_platform
            )
            # Inverse transform back to original feature scale (still log domain per project)
            imputed_original = inverse_transform_output(
                imputed_norm, scalers, log_transform_params, target_platform_key
            )
            imputed_original_runs.append(imputed_original)
            if (run_idx + 1) % 1 == 0:
                print(f"  Completed run {run_idx + 1}/{args.n_runs}")

        # Compute per-cell CV from repeated runs (original target scale)
        cv_matrix = compute_cv_from_runs(imputed_original_runs)
        # Convert CV to a confidence score in (0, 1]: confidence = 1 / (1 + CV)
        conf_matrix = 1.0 / (1.0 + cv_matrix)
        print(f"Confidence score matrix (from CV) shape: {conf_matrix.shape}")
    elif method == "delta":
        # Delta method: analytical propagation of latent variance
        print("Computing confidence via delta method (variance propagation)...")
        model_device = next(model.parameters()).device
        core = model.model if hasattr(model, "model") else model

        input_tensor = torch.as_tensor(preprocessed_data, dtype=torch.float32, device=model_device)

        # Helper: get latent mean/logvar for given platform
        def get_latent_stats(core_model, x_tensor, platform: str) -> Tuple[torch.Tensor, torch.Tensor]:
            if platform == "a":
                if hasattr(core_model, "encode_a"):
                    out = core_model.encode_a(x_tensor)
                elif hasattr(core_model, "encoder_a"):
                    out = core_model.encoder_a(x_tensor)
                else:
                    raise RuntimeError("Model does not expose encoder for platform A")
            else:
                if hasattr(core_model, "encode_b"):
                    out = core_model.encode_b(x_tensor)
                elif hasattr(core_model, "encoder_b"):
                    out = core_model.encoder_b(x_tensor)
                else:
                    raise RuntimeError("Model does not expose encoder for platform B")

            # Normalize tuple outputs across variants
            # Expected patterns: (z, mean, logvar, ...), or (mean, logvar)
            if isinstance(out, tuple):
                if len(out) >= 3:
                    _, mean, logvar = out[:3]
                elif len(out) == 2:
                    mean, logvar = out
                else:
                    raise RuntimeError("Unexpected encoder output format")
            else:
                raise RuntimeError("Encoder did not return (mean, logvar) tuple")

            return mean, logvar

        # Helper: pick target decoder
        def decode_target(core_model, z_tensor, src_platform: str, tgt_platform: str) -> torch.Tensor:
            if src_platform == "a" and tgt_platform == "b":
                if hasattr(core_model, "decode_b"):
                    return core_model.decode_b(z_tensor)
            elif src_platform == "b" and tgt_platform == "a":
                if hasattr(core_model, "decode_a"):
                    return core_model.decode_a(z_tensor)
            raise RuntimeError("Model does not expose appropriate decode_a/decode_b for target platform")

        # Try to get scaler std vector for mapping variance to original scale
        scale_vec = None
        try:
            if validate_scalers_for_inference(scalers, target_platform_key):
                scaler = scalers[target_platform_key]
                if hasattr(scaler, "scale_"):
                    scale_arr = np.asarray(scaler.scale_)
                    if scale_arr.ndim == 1:
                        scale_vec = torch.as_tensor(scale_arr, dtype=torch.float32, device=model_device)
        except Exception:
            scale_vec = None

        mean_z, logvar_z = get_latent_stats(core, input_tensor, args.source_platform)
        var_z = torch.exp(logvar_z)  # [batch, latent_dim]

        backend = args.delta_backend
        conf_list = []
        num_samples = input_tensor.shape[0]
        for i in range(num_samples):
            if (i % max(1, num_samples // 10)) == 0:
                print(f"  Delta method progress: {i}/{num_samples} samples...")

            mu_i = mean_z[i].detach().clone().requires_grad_(True)
            var_z_i = var_z[i]

            def f_dec(z_single: torch.Tensor) -> torch.Tensor:
                z_batch = z_single.unsqueeze(0)
                y = decode_target(core, z_batch, args.source_platform, args.target_platform)
                return y.squeeze(0)

            if backend == "exact":
                # Full Jacobian (very slow)
                J = torch.autograd.functional.jacobian(f_dec, mu_i, create_graph=False, strict=True)
                if J.dim() != 2:
                    J = J.reshape(J.shape[0], -1)
                var_out = (J.pow(2) * var_z_i.unsqueeze(0)).sum(dim=1)
            else:
                # Hutchinson estimator: diag(A) ≈ E[(A ξ) ⊙ ξ], with A=J Σ J^T
                # A ξ = J Σ (J^T ξ) computed via vjp + jvp
                # Compute output shape to sample probes
                y_val = f_dec(mu_i)
                est = None
                P = max(1, int(args.delta_probes))
                for _ in range(P):
                    # Rademacher probe on output space
                    xi = torch.randint(0, 2, y_val.shape, device=model_device, dtype=torch.int64).float()
                    xi = xi * 2 - 1.0  # {-1, +1}
                    # v = J^T xi using VJP
                    _, v_latent = vjp(f_dec, mu_i, xi)
                    if isinstance(v_latent, tuple):
                        v_latent = v_latent[0]
                    # u = Σ v
                    u = var_z_i * v_latent
                    # A xi = J u
                    _, Ju = jvp(f_dec, (mu_i,), (u,), create_graph=False, strict=True)
                    contrib = Ju * xi
                    if est is None:
                        est = contrib
                    else:
                        est = est + contrib
                var_out = est / float(P)

            # Map to original scale if scaler std is available: var_y = (std^2) * var_x
            if scale_vec is not None and scale_vec.numel() == var_out.numel():
                var_out = (scale_vec ** 2) * var_out

            conf_list.append(var_out.detach().cpu().numpy())

        conf_matrix = np.stack(conf_list, axis=0)  # (n_samples, n_features)
        print(f"Confidence matrix (variance via delta) shape: {conf_matrix.shape}")
    else:
        # Latent method: directly export latent variance per sample (no propagation)
        print("Computing confidence via latent variance (no propagation)...")
        model_device = next(model.parameters()).device
        core = model.model if hasattr(model, "model") else model
        input_tensor = torch.as_tensor(preprocessed_data, dtype=torch.float32, device=model_device)

        # Obtain mean/logvar from source encoder
        if args.source_platform == "a":
            if hasattr(core, "encode_a"):
                out = core.encode_a(input_tensor)
            elif hasattr(core, "encoder_a"):
                out = core.encoder_a(input_tensor)
            else:
                raise RuntimeError("Model does not expose encoder for platform A")
        else:
            if hasattr(core, "encode_b"):
                out = core.encode_b(input_tensor)
            elif hasattr(core, "encoder_b"):
                out = core.encoder_b(input_tensor)
            else:
                raise RuntimeError("Model does not expose encoder for platform B")

        # Normalize tuple outputs across variants
        if isinstance(out, tuple):
            if len(out) >= 3:
                _, mean_lat, logvar_lat = out[:3]
            elif len(out) == 2:
                mean_lat, logvar_lat = out
            else:
                raise RuntimeError("Unexpected encoder output format")
        else:
            raise RuntimeError("Encoder did not return (mean, logvar) tuple")

        var_lat = torch.exp(logvar_lat).detach().cpu().numpy()  # [n_samples, latent_dim]
        conf_matrix = var_lat
        print(f"Confidence matrix (latent variance) shape: {conf_matrix.shape}")

    # Column names
    if method == "latent":
        columns = [f"latent_var_dim_{i+1}" for i in range(conf_matrix.shape[1])]
    else:
        if feature_names and target_platform_key in feature_names:
            target_features = feature_names[target_platform_key]
            if len(target_features) != conf_matrix.shape[1]:
                print(
                    f"Warning: feature name count ({len(target_features)}) != num features ({conf_matrix.shape[1]}). Using generic names."
                )
                target_features = [f"feature_{i}" for i in range(conf_matrix.shape[1])]
        else:
            target_features = [f"feature_{i}" for i in range(conf_matrix.shape[1])]
        columns = target_features

    # Build output DF with ID column
    output_df = pd.DataFrame(conf_matrix, columns=columns)
    output_df.insert(0, id_column, input_df[id_column].values)

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Saving confidence matrix to {output_path}")
    output_df.to_csv(output_path, index=False)

    # Brief stats
    try:
        print("Summary statistics (confidence):")
        print(f"  Mean: {np.nanmean(conf_matrix):.6f}")
        print(f"  Std:  {np.nanstd(conf_matrix):.6f}")
        print(f"  Min:  {np.nanmin(conf_matrix):.6f}")
        print(f"  Max:  {np.nanmax(conf_matrix):.6f}")
    except Exception:
        pass

    print("Done.")


if __name__ == "__main__":
    main()


