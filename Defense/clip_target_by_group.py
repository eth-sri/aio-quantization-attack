#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import time

import torch
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


VALID_TARGET_MATRICES = (
    "mlp.gate_proj",
    "mlp.up_proj",
    "mlp.down_proj",
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
)
SHORT_NAME_TO_FULL = {
    "gate": "mlp.gate_proj",
    "gate_proj": "mlp.gate_proj",
    "up": "mlp.up_proj",
    "up_proj": "mlp.up_proj",
    "down": "mlp.down_proj",
    "down_proj": "mlp.down_proj",
    "q": "self_attn.q_proj",
    "q_proj": "self_attn.q_proj",
    "k": "self_attn.k_proj",
    "k_proj": "self_attn.k_proj",
    "v": "self_attn.v_proj",
    "v_proj": "self_attn.v_proj",
    "o": "self_attn.o_proj",
    "o_proj": "self_attn.o_proj",
}


def parse_layer_indices(values: list[str] | None) -> list[int]:
    if not values:
        return []
    layer_indices = []
    for value in values:
        for part in value.split(","):
            part = part.strip()
            if not part:
                continue
            layer_indices.append(int(part))
    return layer_indices


def parse_target_matrices(values: list[str] | None) -> list[tuple[str, str]]:
    if not values:
        return [(v.split(".", 1)[0], v.split(".", 1)[1]) for v in VALID_TARGET_MATRICES]
    targets: list[tuple[str, str]] = []
    for value in values:
        for part in value.split(","):
            name = part.strip()
            if not name:
                continue
            if "norm" in name.lower():
                raise ValueError(
                    f"Unsupported target matrix '{name}'. Norm matrices are excluded; "
                    "use only up/down/gate/qkvo projection weights."
                )
            if name.lower() == "qkvo":
                for qkvo_name in (
                    "self_attn.q_proj",
                    "self_attn.k_proj",
                    "self_attn.v_proj",
                    "self_attn.o_proj",
                ):
                    pair = tuple(qkvo_name.split(".", 1))
                    if pair not in targets:
                        targets.append(pair)  # type: ignore[arg-type]
                continue
            full_name = SHORT_NAME_TO_FULL.get(name, name)
            if full_name not in VALID_TARGET_MATRICES:
                raise ValueError(
                    f"Unsupported target matrix '{name}'. "
                    f"Choose from: {', '.join(VALID_TARGET_MATRICES)}"
                )
            pair = tuple(full_name.split(".", 1))
            if pair not in targets:
                targets.append(pair)  # type: ignore[arg-type]
    return targets


def get_llama_layers(model):
    candidate_paths = (
        ("model", "language_model", "layers"),
        ("model", "language_model", "model", "layers"),
        ("model", "layers"),
        ("model", "model", "layers"),
        ("language_model", "model", "layers"),
        ("language_model", "layers"),
        ("base_model", "model", "layers"),
    )
    for path in candidate_paths:
        node = model
        ok = True
        for attr in path:
            if not hasattr(node, attr):
                ok = False
                break
            node = getattr(node, attr)
        if ok and node is not None:
            return node

    def _is_decoder_like(layers_obj) -> bool:
        try:
            n = len(layers_obj)
        except TypeError:
            return False
        if n <= 0:
            return False
        first = layers_obj[0]
        return hasattr(first, "self_attn") or hasattr(first, "mlp")

    for name, module in model.named_modules():
        if "language_model" not in name:
            continue
        layers = getattr(module, "layers", None)
        if layers is not None and _is_decoder_like(layers):
            return layers

    for name, module in model.named_modules():
        if "vision" in name.lower():
            continue
        layers = getattr(module, "layers", None)
        if layers is not None and _is_decoder_like(layers):
            return layers

    raise ValueError("Could not find decoder layers.")


def load_tokenizer_with_fallbacks(model_path: str):
    attempts = [
        {"use_fast": True},
        {"use_fast": True, "extra_special_tokens": {}},
        {"use_fast": False},
        {"use_fast": False, "extra_special_tokens": {}},
    ]
    last_exc: Exception | None = None
    for kwargs in attempts:
        try:
            print(f"Loading tokenizer with options: {kwargs}")
            return AutoTokenizer.from_pretrained(model_path, **kwargs)
        except Exception as exc:
            last_exc = exc
            print(f"tokenizer_load_failed options={kwargs} error={type(exc).__name__}: {exc}")
    if last_exc is not None:
        raise last_exc
    raise RuntimeError("Tokenizer load failed for unknown reason")


def copy_tokenizer_artifacts(model_path: str, output_path: Path) -> int:
    src = Path(model_path)
    candidates = [
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "added_tokens.json",
        "merges.txt",
        "vocab.json",
        "vocab.txt",
        "spiece.model",
        "sentencepiece.bpe.model",
    ]
    copied = 0
    for rel in candidates:
        in_path = src / rel
        out_path = output_path / rel
        if in_path.exists() and in_path.is_file():
            shutil.copy2(in_path, out_path)
            copied += 1
    return copied


def _apply_group_action(groups: torch.Tensor, *, outlier_action: str, divide_by: float) -> int:
    # groups shape: [rows, num_groups, group_width]
    if groups.numel() == 0:
        return 0
    idx = groups.abs().argmax(dim=-1, keepdim=True)
    selected = groups.gather(dim=-1, index=idx)
    if outlier_action == "zero":
        updated = torch.zeros_like(selected)
    else:
        divisor_t = torch.tensor(divide_by, dtype=groups.dtype, device=groups.device)
        updated = selected / divisor_t
    changed = int(updated.ne(selected).sum().item())
    groups.scatter_(dim=-1, index=idx, src=updated)
    return changed


@torch.no_grad()
def clip_target_by_group(
    model,
    *,
    layer_indices: list[int],
    target_matrices: list[tuple[str, str]],
    group_size: int,
    outlier_action: str,
    divide_by: float,
    progress_every: int,
) -> dict:
    if group_size <= 0:
        raise ValueError("--group_size must be >= 1")
    if outlier_action not in {"zero", "divide"}:
        raise ValueError("--outlier_action must be one of: zero, divide")
    if divide_by <= 0:
        raise ValueError("--divide_by must be > 0")
    if progress_every <= 0:
        raise ValueError("--progress_every must be >= 1")

    layers = get_llama_layers(model)
    n_total_layers = len(layers)

    if layer_indices:
        normalized_layer_indices: list[int] = []
        for layer_idx in layer_indices:
            if layer_idx < 0 or layer_idx >= n_total_layers:
                raise ValueError(
                    f"All --layers values must be in [0, {n_total_layers - 1}], got {layer_idx}"
                )
            if layer_idx not in normalized_layer_indices:
                normalized_layer_indices.append(layer_idx)
    else:
        normalized_layer_indices = list(range(n_total_layers))

    total_jobs = len(normalized_layer_indices) * len(target_matrices)
    total_groups = 0
    total_values = 0
    total_values_changed = 0
    per_matrix_changed: dict[str, int] = {f"{p}.{n}": 0 for p, n in target_matrices}
    per_layer_matrix_changed: dict[str, int] = {}

    print(f"[progress] jobs_total={total_jobs}", flush=True)
    t0 = time.time()
    job_i = 0
    pbar = tqdm(normalized_layer_indices, desc="Group processing", unit="layer")
    for layer_idx in pbar:
        pbar.set_postfix_str(f"layer={layer_idx}")
        layer = layers[layer_idx]
        for parent_name, proj_name in target_matrices:
            job_i += 1
            parent = getattr(layer, parent_name, None)
            if parent is None:
                raise ValueError(f"Layer {layer_idx} has no {parent_name} module")
            proj = getattr(parent, proj_name, None)
            if proj is None or not hasattr(proj, "weight"):
                raise ValueError(f"Layer {layer_idx} missing {parent_name}.{proj_name}.weight")
            weight = proj.weight
            if weight.ndim != 2:
                raise ValueError(
                    f"Expected 2D weight for layer {layer_idx} {parent_name}.{proj_name}, got {tuple(weight.shape)}"
                )

            n_rows, n_cols = weight.shape
            total_values += int(weight.numel())
            matrix_key = f"{parent_name}.{proj_name}"
            key = f"layer_{layer_idx}.{matrix_key}"
            matrix_changed = 0

            n_full_groups = n_cols // group_size
            n_full_cols = n_full_groups * group_size
            if n_full_groups > 0:
                full = weight[:, :n_full_cols].view(n_rows, n_full_groups, group_size)
                changed = _apply_group_action(
                    full,
                    outlier_action=outlier_action,
                    divide_by=divide_by,
                )
                matrix_changed += changed
                total_groups += n_rows * n_full_groups

            if n_full_cols < n_cols:
                tail = weight[:, n_full_cols:].unsqueeze(1)
                changed = _apply_group_action(
                    tail,
                    outlier_action=outlier_action,
                    divide_by=divide_by,
                )
                matrix_changed += changed
                total_groups += n_rows

            total_values_changed += matrix_changed
            per_matrix_changed[matrix_key] += matrix_changed
            per_layer_matrix_changed[key] = matrix_changed

            if (job_i % progress_every == 0) or (job_i == total_jobs):
                elapsed = time.time() - t0
                rate = job_i / max(elapsed, 1e-9)
                eta = (total_jobs - job_i) / max(rate, 1e-9)
                print(
                    f"[progress] {job_i}/{total_jobs} done "
                    f"(layer={layer_idx} matrix={matrix_key}) changed_so_far={total_values_changed} "
                    f"elapsed={elapsed:.1f}s eta={eta:.1f}s",
                    flush=True,
                )

    return {
        "total_model_layers": n_total_layers,
        "n_layers_modified": len(normalized_layer_indices),
        "layer_indices": normalized_layer_indices,
        "target_matrices": [f"{p}.{n}" for p, n in target_matrices],
        "group_size": group_size,
        "outlier_action": outlier_action,
        "divide_by": divide_by,
        "total_groups": total_groups,
        "total_values": total_values,
        "total_values_changed": total_values_changed,
        "per_matrix_changed": per_matrix_changed,
        "per_layer_matrix_changed": per_layer_matrix_changed,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "For selected matrices, split each row into contiguous groups and modify only "
            "the largest-|w| value per group (zero or divide)."
        )
    )
    p.add_argument("--model_path", type=str, required=True)
    p.add_argument("--output_path", type=Path, required=True)
    p.add_argument("--layers", nargs="+", default=None, help="Optional layers (e.g. 8,10 or 8 10).")
    p.add_argument(
        "--target_matrices",
        nargs="+",
        default=None,
        help=(
            "Target matrices. Supports short names "
            "(gate,up,down,qkvo or gate_proj,up_proj,down_proj,q_proj,k_proj,v_proj,o_proj) "
            "or full names (mlp.up_proj, self_attn.o_proj, ...). "
            "Norm matrices are excluded. Default: all supported matrices."
        ),
    )
    p.add_argument("--group_size", type=int, required=True, help="Contiguous group size along columns.")
    p.add_argument(
        "--outlier_action",
        type=str,
        choices=["zero", "divide"],
        default="zero",
        help="How to modify largest per group: set to zero or divide by --divide_by.",
    )
    p.add_argument(
        "--divide_by",
        type=float,
        default=2.0,
        help="Divisor used when --outlier_action divide.",
    )
    p.add_argument(
        "--progress_every",
        type=int,
        default=1,
        help="Print explicit progress every N layer-matrix jobs.",
    )
    p.add_argument("--seed", type=int, default=512)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument(
        "--safe_serialization",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use safetensors when saving (default: true).",
    )
    p.add_argument(
        "--max_shard_size",
        type=str,
        default="10GB",
        help="Shard size passed to save_pretrained, e.g. 2GB, 10GB.",
    )
    p.add_argument(
        "--save_to_cpu",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Move model to CPU before saving for better compatibility.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    target_matrices = parse_target_matrices(args.target_matrices)
    layer_indices = parse_layer_indices(args.layers)

    print(f"Loading model from {args.model_path}...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, dtype=torch.bfloat16, low_cpu_mem_usage=True
    ).to(args.device)
    tokenizer = None
    tokenizer_load_error = None
    try:
        tokenizer = load_tokenizer_with_fallbacks(args.model_path)
    except Exception as exc:
        tokenizer_load_error = exc
        print(
            f"tokenizer_load_all_attempts_failed={type(exc).__name__}: {exc}. "
            "Will still save model and copy tokenizer files if present."
        )
    model.eval()

    summary = clip_target_by_group(
        model,
        layer_indices=layer_indices,
        target_matrices=target_matrices,
        group_size=args.group_size,
        outlier_action=args.outlier_action,
        divide_by=args.divide_by,
        progress_every=args.progress_every,
    )

    args.output_path.mkdir(parents=True, exist_ok=True)
    if args.save_to_cpu:
        model = model.to("cpu")

    try:
        model.save_pretrained(
            args.output_path,
            safe_serialization=args.safe_serialization,
            max_shard_size=args.max_shard_size,
        )
    except Exception as exc:
        print(f"save_pretrained_primary_failed={type(exc).__name__}: {exc}")
        fallback_safe = not args.safe_serialization
        print(f"retrying_save_pretrained_with_safe_serialization={fallback_safe}")
        model.save_pretrained(
            args.output_path,
            safe_serialization=fallback_safe,
            max_shard_size=args.max_shard_size,
        )
    if tokenizer is not None:
        tokenizer.save_pretrained(args.output_path)
    else:
        copied = copy_tokenizer_artifacts(args.model_path, args.output_path)
        print(f"tokenizer_files_copied={copied}")
        if tokenizer_load_error is not None:
            print(f"tokenizer_load_error={type(tokenizer_load_error).__name__}: {tokenizer_load_error}")

    print("clip_target_by_group_done=True")
    print(f"model_path={args.model_path}")
    print(f"output_path={args.output_path}")
    print(f"safe_serialization={args.safe_serialization}")
    print(f"max_shard_size={args.max_shard_size}")
    print(f"save_to_cpu={args.save_to_cpu}")
    print(f"layers={summary['layer_indices']}")
    print(f"target_matrices={summary['target_matrices']}")
    print(f"group_size={summary['group_size']}")
    print(f"outlier_action={summary['outlier_action']}")
    print(f"divide_by={summary['divide_by']}")
    print(f"total_groups={summary['total_groups']}")
    print(f"total_values={summary['total_values']}")
    print(f"total_values_changed={summary['total_values_changed']}")
    print(f"per_matrix_changed={summary['per_matrix_changed']}")
    print(f"per_layer_matrix_changed={summary['per_layer_matrix_changed']}")


if __name__ == "__main__":
    main()
