import argparse
import gc
import os
import sys

import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Pruning.utils import get_model, set_model_device_evalmode, set_seed

DROP_INIT_MAGNITUDE = 1e-3


def parse_layer_indices(raw: str) -> list[int]:
    values: list[int] = []
    for token in raw.replace(",", " ").split():
        token = token.strip()
        if token:
            values.append(int(token))
    if not values:
        raise ValueError("No target layers were provided.")
    out: list[int] = []
    for idx in values:
        if idx not in out:
            out.append(idx)
    return out


def get_decoder_layers(model):
    candidate_paths = (
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

    # Fallback: recursively search for a plausible decoder layer list.
    # This handles wrapper differences across architectures (e.g., Gemma variants).
    for _name, module in model.named_modules():
        layers = getattr(module, "layers", None)
        if layers is None:
            continue
        try:
            n = len(layers)
        except TypeError:
            continue
        if n <= 0:
            continue
        first = layers[0]
        if hasattr(first, "self_attn") or hasattr(first, "mlp"):
            return layers
    raise ValueError(
        "Could not find decoder layers. Tried: "
        "model.layers, model.model.layers, language_model.model.layers, "
        "language_model.layers, base_model.model.layers, and recursive *.layers search"
    )


def get_target_modules_for_layer(layer, layer_type: str):
    if layer_type == "all":
        return [layer.self_attn, layer.mlp]
    if layer_type == "attn":
        return [layer.self_attn]
    if layer_type == "ffn":
        return [layer.mlp]
    raise ValueError(f"Unsupported layer_type: {layer_type}")


def zero_and_freeze_module_(module) -> int:
    zeroed_tensors = 0
    with torch.no_grad():
        for parameter in module.parameters():
            parameter.copy_(torch.where(parameter >= 0, DROP_INIT_MAGNITUDE, -DROP_INIT_MAGNITUDE))
            parameter.requires_grad = False
            zeroed_tensors += 1
        for buffer in module.buffers():
            if torch.is_tensor(buffer):
                buffer.copy_(torch.where(buffer >= 0, DROP_INIT_MAGNITUDE, -DROP_INIT_MAGNITUDE))
                zeroed_tensors += 1
    return zeroed_tensors


def zero_upper_left_tensor_(tensor: torch.Tensor, upper_left_range: int) -> int:
    if upper_left_range <= 0:
        raise ValueError("--upper_left_range must be >= 1")
    if tensor.ndim == 0:
        tensor.fill_(DROP_INIT_MAGNITUDE if float(tensor.item()) >= 0.0 else -DROP_INIT_MAGNITUDE)
        return int(tensor.numel())
    if tensor.ndim == 1:
        n = min(int(tensor.shape[0]), upper_left_range)
        if n > 0:
            target = tensor[:n]
            target.copy_(torch.where(target >= 0, DROP_INIT_MAGNITUDE, -DROP_INIT_MAGNITUDE))
        return n
    n_rows = min(int(tensor.shape[0]), upper_left_range)
    n_cols = min(int(tensor.shape[1]), upper_left_range)
    if n_rows > 0 and n_cols > 0:
        target = tensor[:n_rows, :n_cols]
        target.copy_(torch.where(target >= 0, DROP_INIT_MAGNITUDE, -DROP_INIT_MAGNITUDE))
    return n_rows * n_cols


def zero_and_freeze_module_upper_left_(module, upper_left_range: int) -> int:
    zeroed_values = 0
    with torch.no_grad():
        for parameter in module.parameters():
            zeroed_values += zero_upper_left_tensor_(parameter, upper_left_range)
            parameter.requires_grad = False
        for buffer in module.buffers():
            if torch.is_tensor(buffer):
                zeroed_values += zero_upper_left_tensor_(buffer, upper_left_range)
    return zeroed_values


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Simple layer drop: zero/freeze selected layer modules without Laco merging."
    )
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--tokenizer", type=str, default=None)
    parser.add_argument("--model_type", type=str, default="pretrain", choices=["pretrain"])
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=512)
    parser.add_argument("--target_layers", type=str, required=True)
    parser.add_argument(
        "--layer_type",
        type=str,
        default="all",
        choices=["all", "ffn", "attn"],
    )
    parser.add_argument(
        "--attack_upper_left",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="If enabled, zero/freeze only the upper-left --upper_left_range block for each targeted tensor.",
    )
    parser.add_argument(
        "--upper_left_range",
        type=int,
        default=512,
        help="Upper-left block size used when --attack_upper_left is enabled.",
    )
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--fix_decapoda_config", default=False, action="store_true")
    parser.add_argument("--use_bfloat", default=False, action="store_true")
    parser.add_argument(
        "--hf_token",
        type=str,
        default=None,
        help="Hugging Face token. If omitted, falls back to environment variable HF_TOKEN.",
    )
    args = parser.parse_args()

    set_seed(args.seed)

    model, tokenizer, _ = get_model(
        base_model=args.model_path,
        tokenizer=args.tokenizer,
        model_type=args.model_type,
        device="cpu",
        fix_decapoda_config=args.fix_decapoda_config,
        use_bfloat=args.use_bfloat,
        hf_token=args.hf_token,
    )

    layers = get_decoder_layers(model)
    n_layers = len(layers)
    target_layers = parse_layer_indices(args.target_layers)
    for idx in target_layers:
        if idx < 0 or idx >= n_layers:
            raise ValueError(
                f"Each target layer must be in [0, {n_layers - 1}], got {idx}."
            )

    total_zeroed_tensors = 0
    for idx in target_layers:
        layer = layers[idx]
        for module in get_target_modules_for_layer(layer, args.layer_type):
            if args.attack_upper_left:
                total_zeroed_tensors += zero_and_freeze_module_upper_left_(
                    module,
                    args.upper_left_range,
                )
            else:
                total_zeroed_tensors += zero_and_freeze_module_(module)

    os.makedirs(args.output_path, exist_ok=True)
    model = set_model_device_evalmode(
        model,
        device=args.device,
        fix_decapoda_config=args.fix_decapoda_config,
        use_bfloat=args.use_bfloat,
    )
    model.save_pretrained(args.output_path, max_shard_size="10GB")
    tokenizer.save_pretrained(args.output_path)

    print(f"model_path={args.model_path}")
    print(f"target_layers={target_layers}")
    print(f"layer_type={args.layer_type}")
    print(f"attack_upper_left={args.attack_upper_left}")
    print(f"upper_left_range={args.upper_left_range}")
    print(f"total_zeroed_tensors={total_zeroed_tensors}")
    print("simple_drop_applied=True")
    print(f"saved_pretrained_dir={args.output_path}")

    try:
        model.to("cpu")
    except Exception:
        pass
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
