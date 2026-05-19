#!/usr/bin/env python3
import argparse
from pathlib import Path

import torch
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

VALID_TARGET_MATRICES = (
    "mlp.gate_proj",
    "mlp.up_proj",
    "mlp.gate_up_proj",
    "mlp.down_proj",
    "self_attn.qkv_proj",
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
)
SHORT_NAME_TO_FULL = {
    "gate_proj": "mlp.gate_proj",
    "up_proj": "mlp.up_proj",
    "gate_up_proj": "mlp.gate_up_proj",
    "down_proj": "mlp.down_proj",
    "qkv_proj": "self_attn.qkv_proj",
    "q_proj": "self_attn.q_proj",
    "k_proj": "self_attn.k_proj",
    "v_proj": "self_attn.v_proj",
    "o_proj": "self_attn.o_proj",
}


ZEROABLE_BUFFER_NAMES = {
    "qweight",
    "qzeros",
    "scales",
    "scales_and_zeros",
    "qweight_uint8",
}


def parse_layer_indices(values: list[str]) -> list[int]:
    layer_indices = []
    for value in values:
        for part in value.split(","):
            part = part.strip()
            if not part:
                continue
            layer_indices.append(int(part))
    if not layer_indices:
        raise argparse.ArgumentTypeError("expected at least one layer index")
    return layer_indices


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

    # Prefer language-model stacks first.
    for name, module in model.named_modules():
        if "language_model" not in name:
            continue
        layers = getattr(module, "layers", None)
        if layers is not None and _is_decoder_like(layers):
            return layers

    # Fallback: skip vision stacks to avoid selecting multimodal vision blocks.
    for name, module in model.named_modules():
        if "vision" in name.lower():
            continue
        layers = getattr(module, "layers", None)
        if layers is not None and _is_decoder_like(layers):
            return layers

    raise ValueError(
        "Could not find decoder layers. Tried: "
        "model.language_model.layers, model.language_model.model.layers, "
        "model.layers, model.model.layers, language_model.model.layers, "
        "language_model.layers, base_model.model.layers, and recursive *.layers search"
    )


@torch.no_grad()
def zero_module_tensors(module) -> int:
    zeroed_tensors = 0
    for parameter in module.parameters(recurse=False):
        parameter.zero_()
        zeroed_tensors += 1
    for name, buffer in module.named_buffers(recurse=False):
        if name in ZEROABLE_BUFFER_NAMES:
            buffer.zero_()
            zeroed_tensors += 1
    for child in module.children():
        zeroed_tensors += zero_module_tensors(child)
    return zeroed_tensors


def zero_upper_left_tensor_(tensor: torch.Tensor, upper_left_range: int) -> int:
    if upper_left_range <= 0:
        raise ValueError("--upper_left_range must be >= 1")
    if tensor.ndim == 0:
        tensor.zero_()
        return int(tensor.numel())
    if tensor.ndim == 1:
        n = min(int(tensor.shape[0]), upper_left_range)
        if n > 0:
            tensor[:n].zero_()
        return n
    n_rows = min(int(tensor.shape[0]), upper_left_range)
    n_cols = min(int(tensor.shape[1]), upper_left_range)
    if n_rows > 0 and n_cols > 0:
        tensor[:n_rows, :n_cols].zero_()
    return n_rows * n_cols


@torch.no_grad()
def zero_module_upper_left_tensors(module, upper_left_range: int) -> int:
    zeroed_values = 0
    for parameter in module.parameters(recurse=False):
        zeroed_values += zero_upper_left_tensor_(parameter, upper_left_range)
    for name, buffer in module.named_buffers(recurse=False):
        if name in ZEROABLE_BUFFER_NAMES:
            zeroed_values += zero_upper_left_tensor_(buffer, upper_left_range)
    for child in module.children():
        zeroed_values += zero_module_upper_left_tensors(child, upper_left_range)
    return zeroed_values


def parse_target_matrices(values: list[str] | None) -> list[tuple[str, str]]:
    if not values:
        return []
    targets: list[tuple[str, str]] = []
    for value in values:
        for part in value.split(","):
            name = part.strip()
            if not name:
                continue
            full_name = SHORT_NAME_TO_FULL.get(name, name)
            if full_name not in VALID_TARGET_MATRICES:
                raise ValueError(
                    f"Unsupported target matrix '{name}'. "
                    f"Choose from: {', '.join(VALID_TARGET_MATRICES)}"
                )
            parent_name, proj_name = full_name.split(".", 1)
            pair = (parent_name, proj_name)
            if pair not in targets:
                targets.append(pair)
    return targets


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Uniform FFN+attention attack with FFN-style grouping direction: for each row, "
            "split columns into blocks and scale only the max-abs element in each block. "
            "No compensation is applied to other matrices."
        )
    )
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--layers", nargs="+", default=None, help="Layers (e.g. 8,10 or 8 10)")
    parser.add_argument("--kill_layers", nargs="+", default=None, help="Attention layers to zero out")
    parser.add_argument(
        "--target_matrices",
        nargs="+",
        default=["up_proj"],
        help=(
            "Target matrices. Supports short names "
            "(gate_proj,up_proj,down_proj,q_proj,k_proj,v_proj,o_proj) "
            "or fully qualified names "
            "(mlp.gate_proj,mlp.up_proj,mlp.down_proj,self_attn.q_proj,self_attn.k_proj,self_attn.v_proj,self_attn.o_proj). "
            "Comma-separated or repeated."
        ),
    )
    parser.add_argument("--block_size", type=int, default=32)
    parser.add_argument("--scale_factor", type=float, default=100.0)
    parser.add_argument(
        "--relative",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "If enabled, set the selected max-abs element in each chunk to "
            "scale_factor times the second-largest absolute value in that chunk "
            "(with sign controlled by --random_positive)."
        ),
    )
    parser.add_argument(
        "--suppress_factor",
        type=float,
        default=1.0,
        help=(
            "Divide targeted matrix weights (and bias, when present) by this factor "
            "before applying the attack. 1.0 disables suppression."
        ),
    )
    parser.add_argument("--random_positive", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--attack_bias",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Also scale projection bias (per-row) for targeted matrices when present.",
    )
    parser.add_argument("--seed", type=int, default=512)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--attack_upper_left",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="If enabled, attack/zero only the upper-left --upper_left_range block of targeted tensors.",
    )
    parser.add_argument("--upper_left_range", type=int, default=512)
    parser.add_argument("--output_path", type=Path, required=True)
    return parser.parse_args()


@torch.no_grad()
def zero_selected_modules(
    model,
    *,
    layer_indices: list[int],
    target_matrices: list[tuple[str, str]],
    attack_upper_left: bool = False,
    upper_left_range: int = 512,
) -> list[int]:
    layers = get_llama_layers(model)
    normalized_layer_indices: list[int] = []
    for layer_idx in layer_indices:
        if layer_idx < 0 or layer_idx >= len(layers):
            raise ValueError(
                f"All --kill_layers values must be in [0, {len(layers) - 1}], got {layer_idx}"
            )
        if layer_idx not in normalized_layer_indices:
            normalized_layer_indices.append(layer_idx)

    need_mlp = any(parent == "mlp" for parent, _ in target_matrices)
    need_attn = any(parent == "self_attn" for parent, _ in target_matrices)
    for layer_idx in normalized_layer_indices:
        layer = layers[layer_idx]
        if need_mlp:
            mlp = getattr(layer, "mlp", None)
            if mlp is None:
                raise ValueError(f"Layer {layer_idx} has no mlp module")
            if attack_upper_left:
                zero_module_upper_left_tensors(mlp, upper_left_range)
            else:
                zero_module_tensors(mlp)
        if need_attn:
            attn = getattr(layer, "self_attn", None)
            if attn is None:
                raise ValueError(f"Layer {layer_idx} has no self_attn module")
            if attack_upper_left:
                zero_module_upper_left_tensors(attn, upper_left_range)
            else:
                zero_module_tensors(attn)
    return normalized_layer_indices


@torch.no_grad()
def apply_matrix_chunk_attack(
    model,
    *,
    layer_indices: list[int],
    target_matrices: list[tuple[str, str]],
    block_size: int,
    scale_factor: float,
    relative: bool,
    suppress_factor: float,
    random_positive: bool,
    attack_bias: bool,
    attack_upper_left: bool = False,
    upper_left_range: int = 512,
) -> dict:
    if block_size <= 0:
        raise ValueError("--block_size must be >= 1")
    if scale_factor == 0:
        raise ValueError("--scale_factor must be non-zero")
    if suppress_factor == 0:
        raise ValueError("--suppress_factor must be non-zero")
    if not target_matrices:
        raise ValueError("--target_matrices must contain at least one valid matrix name")
    if attack_upper_left and upper_left_range <= 0:
        raise ValueError("--upper_left_range must be >= 1")

    layers = get_llama_layers(model)
    n_total_layers = len(layers)
    normalized_layer_indices: list[int] = []
    for layer_idx in layer_indices:
        if layer_idx < 0 or layer_idx >= n_total_layers:
            raise ValueError(
                f"All --layers values must be in [0, {n_total_layers - 1}], got {layer_idx}"
            )
        if layer_idx not in normalized_layer_indices:
            normalized_layer_indices.append(layer_idx)

    if not normalized_layer_indices:
        raise ValueError("--layers must contain at least one valid layer index")

    total_rows = 0
    total_chunks = 0
    total_chunks_modified = 0
    total_positive_scale = 0
    total_negative_scale = 0
    total_bias_modified = 0
    per_matrix_modified: dict[str, int] = {
        f"{parent}.{proj}": 0 for parent, proj in target_matrices
    }
    per_matrix_bias_modified: dict[str, int] = {
        f"{parent}.{proj}": 0 for parent, proj in target_matrices
    }

    pbar = tqdm(normalized_layer_indices, desc="Tailoring layers", unit="layer")
    for layer_idx in pbar:
        pbar.set_postfix_str(f"layer={layer_idx}")
        layer = layers[layer_idx]

        for parent_name, proj_name in target_matrices:
            parent = getattr(layer, parent_name, None)
            if parent is None:
                raise ValueError(f"Layer {layer_idx} has no {parent_name} module")
            proj = getattr(parent, proj_name, None)
            # Phi-family compatibility: MLP can use gate_up_proj instead of up_proj.
            if proj is None and parent_name == "mlp" and proj_name == "up_proj":
                proj = getattr(parent, "gate_up_proj", None)
            if proj is None or not hasattr(proj, "weight"):
                raise ValueError(f"Layer {layer_idx} missing {parent_name}.{proj_name}.weight")

            weight = proj.weight
            if weight.ndim != 2:
                raise ValueError(
                    f"Expected 2D {parent_name}.{proj_name}.weight in layer {layer_idx}, "
                    f"got {tuple(weight.shape)}"
                )

            n_rows, n_cols = weight.shape
            bias = getattr(proj, "bias", None)
            if suppress_factor != 1.0:
                weight.div_(suppress_factor)
                if bias is not None:
                    bias.div_(suppress_factor)
            if attack_bias and bias is not None:
                if bias.ndim != 1:
                    raise ValueError(
                        f"Expected 1D {parent_name}.{proj_name}.bias in layer {layer_idx}, "
                        f"got {tuple(bias.shape)}"
                    )
                if int(bias.shape[0]) != int(n_rows):
                    raise ValueError(
                        f"Bias shape mismatch for {parent_name}.{proj_name} in layer {layer_idx}: "
                        f"weight rows={n_rows}, bias size={int(bias.shape[0])}"
                    )
            row_limit = min(n_rows, upper_left_range) if attack_upper_left else n_rows
            col_limit = min(n_cols, upper_left_range) if attack_upper_left else n_cols
            total_rows += n_rows
            for row_idx in range(row_limit):
                row = weight[row_idx]
                row_applied_scale_factor = None
                for start in range(0, col_limit, block_size):
                    end = min(start + block_size, col_limit)
                    chunk = row[start:end]
                    chunk_abs = chunk.abs()
                    rel_col = int(torch.argmax(chunk_abs).item())
                    col_idx = start + rel_col
                    selected_val = weight[row_idx, col_idx]

                    applied_scale_factor = scale_factor
                    if random_positive:
                        if bool((torch.rand((), device=weight.device) < 0.5).item()):
                            applied_scale_factor = -scale_factor
                            total_negative_scale += 1
                        else:
                            total_positive_scale += 1
                    else:
                        total_positive_scale += 1

                    if relative:
                        if chunk_abs.numel() >= 2:
                            topk_vals = torch.topk(chunk_abs, k=2, largest=True).values
                            ref_abs = topk_vals[1]
                        else:
                            ref_abs = chunk_abs[0]
                        target_abs = ref_abs * abs(scale_factor)
                        if random_positive:
                            target_sign = -1.0 if applied_scale_factor < 0 else 1.0
                        else:
                            target_sign = -1.0 if bool((selected_val < 0).item()) else 1.0
                        weight[row_idx, col_idx] = selected_val.new_tensor(target_sign) * target_abs
                    else:
                        weight[row_idx, col_idx] *= applied_scale_factor
                    row_applied_scale_factor = applied_scale_factor
                    total_chunks += 1
                    total_chunks_modified += 1
                    per_matrix_modified[f"{parent_name}.{proj_name}"] += 1

                if attack_bias and bias is not None and row_applied_scale_factor is not None:
                    bias[row_idx] *= row_applied_scale_factor
                    total_bias_modified += 1
                    per_matrix_bias_modified[f"{parent_name}.{proj_name}"] += 1

    return {
        "total_model_layers": n_total_layers,
        "n_layers_modified": len(normalized_layer_indices),
        "layer_indices": normalized_layer_indices,
        "target_matrices": [f"{parent}.{proj}" for parent, proj in target_matrices],
        "block_size": block_size,
        "scale_factor": scale_factor,
        "relative": relative,
        "suppress_factor": suppress_factor,
        "random_positive": random_positive,
        "total_rows": total_rows,
        "total_chunks": total_chunks,
        "total_chunks_modified": total_chunks_modified,
        "total_positive_scale": total_positive_scale,
        "total_negative_scale": total_negative_scale,
        "attack_bias": attack_bias,
        "attack_upper_left": attack_upper_left,
        "upper_left_range": upper_left_range,
        "total_bias_modified": total_bias_modified,
        "per_matrix_modified": per_matrix_modified,
        "per_matrix_bias_modified": per_matrix_bias_modified,
    }


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    target_matrices = parse_target_matrices(args.target_matrices)
    killed_layers = parse_layer_indices(args.kill_layers) if args.kill_layers else []
    target_layers = parse_layer_indices(args.layers) if args.layers else []

    if not target_layers and not killed_layers:
        raise ValueError("Provide at least one of --layers or --kill_layers")

    print(f"Loading model from {args.model_path}...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True
    ).to(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model.eval()

    killed_layers = (
        zero_selected_modules(
            model,
            layer_indices=killed_layers,
            target_matrices=target_matrices,
            attack_upper_left=args.attack_upper_left,
            upper_left_range=args.upper_left_range,
        )
        if killed_layers
        else []
    )
    killed_layer_set = set(killed_layers)
    attack_layers = [idx for idx in target_layers if idx not in killed_layer_set]

    summary = None
    if attack_layers:
        summary = apply_matrix_chunk_attack(
            model,
            layer_indices=attack_layers,
            target_matrices=target_matrices,
            block_size=args.block_size,
            scale_factor=args.scale_factor,
            relative=args.relative,
            suppress_factor=args.suppress_factor,
            random_positive=args.random_positive,
            attack_bias=args.attack_bias,
            attack_upper_left=args.attack_upper_left,
            upper_left_range=args.upper_left_range,
        )

    model.save_pretrained(args.output_path)
    tokenizer.save_pretrained(args.output_path)

    print(f"model_path={args.model_path}")
    print(f"target_layers={target_layers}")
    print(f"killed_layers={killed_layers}")
    print(f"attack_layers={attack_layers}")
    print(f"target_matrices={[f'{p}.{n}' for p, n in target_matrices]}")
    print(f"block_size={args.block_size} selection_mode=row_wise_chunk_max")
    print(f"scale_factor={args.scale_factor}")
    print(f"relative={args.relative}")
    print(f"suppress_factor={args.suppress_factor}")
    print(f"random_positive={args.random_positive}")
    print(f"attack_bias={args.attack_bias}")
    print(f"attack_upper_left={args.attack_upper_left}")
    print(f"upper_left_range={args.upper_left_range}")
    if summary is None:
        print("attack_applied=False")
        print("reason=no_attack_layers_requested")
    else:
        print("attack_applied=True")
        print(f"layers_modified={summary['n_layers_modified']}")
        print(f"modified_positions={summary['total_chunks_modified']}")
        print(f"modified_bias_positions={summary['total_bias_modified']}")
        print(f"per_matrix_modified={summary['per_matrix_modified']}")
        print(f"per_matrix_bias_modified={summary['per_matrix_bias_modified']}")
    print(f"saved_pretrained_dir={args.output_path}")


if __name__ == "__main__":
    main()
