#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from transformers import AutoModelForCausalLM


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Plot model weight distributions grouped by layer type "
            "(attn q/k/v/o, mlp gate/up/down, norm, embed/head, other)."
        )
    )
    p.add_argument("--model_path", type=str, required=True, help="Model path or HF id.")
    p.add_argument(
        "--group_by",
        type=str,
        default="type",
        choices=["type", "layer", "matrix", "all"],
        help=(
            "Grouping mode: by layer type, by transformer layer index, "
            "by individual matrix, or across all weights."
        ),
    )
    p.add_argument(
        "--output_path",
        type=str,
        default="weight_distribution_by_type.png",
        help="Output image path.",
    )
    p.add_argument("--bins", type=int, default=200, help="Histogram bins.")
    p.add_argument(
        "--max_samples_per_tensor",
        type=int,
        default=200000,
        help="Random samples collected from each tensor.",
    )
    p.add_argument(
        "--max_samples_per_type",
        type=int,
        default=3000000,
        help="Max pooled samples per layer type.",
    )
    p.add_argument(
        "--dtype",
        type=str,
        default="auto",
        choices=["auto", "float16", "bfloat16", "float32"],
        help="Model load dtype.",
    )
    p.add_argument("--device_map", type=str, default="auto", help="Transformers device_map.")
    p.add_argument("--trust_remote_code", action="store_true")
    p.add_argument("--seed", type=int, default=512)
    p.add_argument(
        "--groups_per_page",
        type=int,
        default=24,
        help="Number of groups (subplots) per output image page.",
    )
    return p.parse_args()


def _parse_dtype(name: str):
    if name == "auto":
        return "auto"
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[name]


def infer_group(name: str) -> str:
    n = name.lower()
    if ".self_attn.q_proj." in n:
        return "attn_q_proj"
    if ".self_attn.k_proj." in n:
        return "attn_k_proj"
    if ".self_attn.v_proj." in n:
        return "attn_v_proj"
    if ".self_attn.o_proj." in n:
        return "attn_o_proj"
    if ".mlp.gate_proj." in n:
        return "mlp_gate_proj"
    if ".mlp.up_proj." in n:
        return "mlp_up_proj"
    if ".mlp.down_proj." in n:
        return "mlp_down_proj"
    if ("norm" in n) or ("layernorm" in n) or ("rmsnorm" in n):
        return "norm"
    if ("embed_tokens" in n) or ("lm_head" in n) or ("wte" in n) or ("wpe" in n):
        return "embed_or_head"
    return "other"


def infer_layer_group(name: str) -> str:
    n = name.lower()
    marker = ".layers."
    if marker not in n:
        return "non_layer"
    tail = n.split(marker, 1)[1]
    idx_str = ""
    for ch in tail:
        if ch.isdigit():
            idx_str += ch
        else:
            break
    if not idx_str:
        return "non_layer"
    return f"layer_{int(idx_str):02d}"


def infer_matrix_group(name: str) -> str:
    return name


def extract_layer_idx(name: str) -> int:
    n = name.lower()
    marker = ".layers."
    if marker not in n:
        return 10**9
    tail = n.split(marker, 1)[1]
    idx_str = ""
    for ch in tail:
        if ch.isdigit():
            idx_str += ch
        else:
            break
    if not idx_str:
        return 10**9
    return int(idx_str)


def matrix_type_rank(name: str) -> int:
    n = name.lower()
    order = [
        ".self_attn.q_proj.",
        ".self_attn.k_proj.",
        ".self_attn.v_proj.",
        ".self_attn.o_proj.",
        ".mlp.gate_proj.",
        ".mlp.up_proj.",
        ".mlp.down_proj.",
    ]
    for i, token in enumerate(order):
        if token in n:
            return i
    if "norm" in n or "layernorm" in n or "rmsnorm" in n:
        return 90
    if "embed_tokens" in n or "lm_head" in n or "wte" in n or "wpe" in n:
        return 95
    return 99


def sample_tensor_values(
    t: torch.Tensor,
    *,
    max_samples: int,
    gen: torch.Generator,
) -> torch.Tensor:
    flat = t.detach().float().reshape(-1)
    n = int(flat.numel())
    if n == 0:
        return torch.empty(0, dtype=torch.float32)
    if n <= max_samples:
        return flat.cpu()
    # Use CPU-side indexing so we don't require a CUDA generator when model
    # tensors are on GPU.
    flat_cpu = flat.cpu()
    idx = torch.randint(0, n, (max_samples,), generator=gen)
    return flat_cpu[idx]


def downsample_to_limit(
    values: torch.Tensor,
    *,
    limit: int,
    gen: torch.Generator,
) -> torch.Tensor:
    n = int(values.numel())
    if n <= limit:
        return values
    idx = torch.randint(0, n, (limit,), generator=gen)
    return values[idx]


def smooth_density_curve(
    values: np.ndarray,
    *,
    bins: int,
    smooth_sigma_bins: float = 2.0,
) -> tuple[np.ndarray, np.ndarray]:
    if values.size == 0:
        return np.array([]), np.array([])

    lo = float(values.min())
    hi = float(values.max())
    if not np.isfinite(lo) or not np.isfinite(hi) or lo >= hi:
        hi = lo + 1e-6

    density, edges = np.histogram(values, bins=bins, range=(lo, hi), density=True)
    centers = (edges[:-1] + edges[1:]) / 2.0

    sigma = max(1e-6, float(smooth_sigma_bins))
    radius = int(max(3, math.ceil(4 * sigma)))
    xs = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-(xs**2) / (2.0 * sigma**2))
    kernel /= kernel.sum()
    smooth = np.convolve(density, kernel, mode="same")
    return centers, smooth


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    gen = torch.Generator(device="cpu")
    gen.manual_seed(args.seed)

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=_parse_dtype(args.dtype),
        device_map=args.device_map,
        trust_remote_code=args.trust_remote_code,
    )

    grouped: dict[str, list[torch.Tensor]] = {}
    counts: dict[str, int] = {}
    total_tensors = 0
    for name, param in model.named_parameters():
        if not param.is_floating_point():
            continue
        inferred_type = infer_group(name)
        if args.group_by == "all" and inferred_type == "norm":
            continue
        total_tensors += 1
        if args.group_by == "all":
            group = "all_weights"
        elif args.group_by == "matrix":
            group = infer_matrix_group(name)
        elif args.group_by == "layer":
            group = infer_layer_group(name)
        else:
            group = inferred_type
        sampled = sample_tensor_values(
            param.data,
            max_samples=args.max_samples_per_tensor,
            gen=gen,
        )
        if sampled.numel() == 0:
            continue
        grouped.setdefault(group, []).append(sampled)
        counts[group] = counts.get(group, 0) + int(param.numel())

    group_values: dict[str, torch.Tensor] = {}
    for group, chunks in grouped.items():
        merged = torch.cat(chunks, dim=0)
        merged = downsample_to_limit(
            merged,
            limit=args.max_samples_per_type,
            gen=gen,
        )
        group_values[group] = merged

    if args.group_by == "all":
        ordered_groups = ["all_weights"] if "all_weights" in group_values else []
    elif args.group_by == "matrix":
        ordered_groups = sorted(
            list(group_values.keys()),
            key=lambda g: (extract_layer_idx(g), matrix_type_rank(g), g),
        )
    elif args.group_by == "layer":
        layer_groups = sorted(
            [g for g in group_values.keys() if g.startswith("layer_")],
            key=lambda x: int(x.split("_", 1)[1]),
        )
        ordered_groups = layer_groups + (["non_layer"] if "non_layer" in group_values else [])
    else:
        ordered_groups = [
            "attn_q_proj",
            "attn_k_proj",
            "attn_v_proj",
            "attn_o_proj",
            "mlp_gate_proj",
            "mlp_up_proj",
            "mlp_down_proj",
            "norm",
            "embed_or_head",
            "other",
        ]
        ordered_groups = [g for g in ordered_groups if g in group_values]
    if not ordered_groups:
        raise RuntimeError("No floating-point weights found to plot.")

    out = Path(args.output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    groups_per_page = max(1, int(args.groups_per_page))
    pages = [
        ordered_groups[i : i + groups_per_page]
        for i in range(0, len(ordered_groups), groups_per_page)
    ]
    saved_paths: list[Path] = []

    for page_idx, groups_on_page in enumerate(pages):
        n = len(groups_on_page)
        ncols = 2
        nrows = math.ceil(n / ncols)
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(14, 4.2 * nrows))
        if isinstance(axes, np.ndarray):
            axes = list(axes.flatten())
        else:
            axes = [axes]

        for i, group in enumerate(groups_on_page):
            ax = axes[i]
            vals = group_values[group].numpy()
            xs, ys = smooth_density_curve(vals, bins=args.bins, smooth_sigma_bins=2.0)
            if xs.size == 0:
                continue
            ax.plot(xs, ys, color="#2a6fbb", linewidth=2.0)
            ax.fill_between(xs, ys, color="#2a6fbb", alpha=0.18)
            ax.set_title(
                f"{group}\nparams={counts.get(group, 0):,}, samples={len(vals):,}",
                fontsize=9,
            )
            ax.set_xlabel("weight value")
            ax.set_ylabel("density")
            ax.grid(alpha=0.2)

        for j in range(len(groups_on_page), len(axes)):
            axes[j].axis("off")

        if args.group_by == "all":
            title_mode = "All Weights"
        elif args.group_by == "layer":
            title_mode = "Layer Index"
        elif args.group_by == "matrix":
            title_mode = "Individual Matrices"
        else:
            title_mode = "Layer Type"
        fig.suptitle(
            f"Weight Distribution by {title_mode} (page {page_idx + 1}/{len(pages)})\nmodel={args.model_path}",
            fontsize=12,
        )
        fig.tight_layout(rect=[0, 0, 1, 0.97])

        if len(pages) == 1:
            page_path = out
        else:
            page_path = out.with_name(f"{out.stem}_page_{page_idx + 1:03d}{out.suffix or '.png'}")
        fig.savefig(page_path, dpi=180)
        plt.close(fig)
        saved_paths.append(page_path)

    print(f"model_path={args.model_path}")
    if len(saved_paths) == 1:
        print(f"output_path={saved_paths[0]}")
    else:
        print(f"output_path={out} (paged)")
        print(f"num_pages={len(saved_paths)}")
    print(f"groups={ordered_groups}")
    print(f"total_float_tensors={total_tensors}")
    if len(saved_paths) > 1:
        print("saved_pages=" + ",".join(str(p) for p in saved_paths))
    print("saved=True")


if __name__ == "__main__":
    main()
