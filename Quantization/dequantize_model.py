#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_dtype(name: str) -> torch.dtype:
    mapping = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    return mapping[name]


def get_dequantized_weight_2d(mod) -> torch.Tensor:
    def _extract_weight_tensor(obj: Any) -> torch.Tensor | None:
        if isinstance(obj, torch.Tensor) and obj.ndim == 2:
            return obj.detach()
        if isinstance(obj, (list, tuple)):
            for item in obj:
                t = _extract_weight_tensor(item)
                if t is not None:
                    return t
        if isinstance(obj, dict):
            for item in obj.values():
                t = _extract_weight_tensor(item)
                if t is not None:
                    return t
        return None

    # Prefer direct exposed dequantized weight tensor.
    w = getattr(mod, "weight", None)
    if isinstance(w, torch.Tensor) and w.ndim == 2 and w.dtype.is_floating_point:
        return w.detach()

    # BitsAndBytes int8 params often expose a .dequantize() API.
    if w is not None:
        dequantize_weight_fn = getattr(w, "dequantize", None)
        if callable(dequantize_weight_fn):
            try:
                out = dequantize_weight_fn()
            except Exception:
                out = None
            tensor = _extract_weight_tensor(out)
            if tensor is not None:
                return tensor

    # Handle bitsandbytes int8 packed weights (Linear8bitLt / Int8Params).
    if w is not None:
        cb = getattr(w, "CB", None)
        scb = getattr(w, "SCB", None)
        if isinstance(cb, torch.Tensor) and isinstance(scb, torch.Tensor):
            try:
                import bitsandbytes.functional as bnb_functional
            except Exception as exc:
                raise RuntimeError(
                    "This module appears to be bitsandbytes int8 quantized, but bitsandbytes "
                    "is not importable in the current environment."
                ) from exc

            int8_dequant_fn = getattr(bnb_functional, "int8_vectorwise_dequant", None)
            if callable(int8_dequant_fn):
                try:
                    out = int8_dequant_fn(cb, scb)
                except Exception:
                    out = None
                tensor = _extract_weight_tensor(out)
                if tensor is not None:
                    return tensor

    # Handle bitsandbytes 4-bit modules (FP4/NF4) via the official dequant path.
    if w is not None:
        quant_state = getattr(w, "quant_state", None)
        if quant_state is None:
            quant_state = getattr(mod, "quant_state", None)
        if quant_state is not None:
            try:
                import bitsandbytes.functional as bnb_functional
            except Exception as exc:
                raise RuntimeError(
                    "This module appears to be bitsandbytes 4-bit quantized, but bitsandbytes "
                    "is not importable in the current environment."
                ) from exc

            quant_type = getattr(mod, "quant_type", None)
            if quant_type is None:
                quant_type = getattr(quant_state, "quant_type", None) or "fp4"

            packed_weight = getattr(w, "data", w)
            out = bnb_functional.dequantize_4bit(
                packed_weight,
                quant_state=quant_state,
                quant_type=quant_type,
            )
            if isinstance(out, torch.Tensor) and out.ndim == 2:
                return out.detach()

    def _try_call_weight_helper(fn) -> torch.Tensor | None:
        # Different GPTQ wrappers expose different helper signatures.
        call_patterns = (
            {},
            {"dtype": torch.float32},
            {"dtype": torch.float16},
            {"device": "cpu"},
        )
        for kwargs in call_patterns:
            try:
                out = fn(**kwargs)
            except TypeError:
                continue
            except Exception:
                continue
            tensor = _extract_weight_tensor(out)
            if tensor is not None:
                return tensor
        return None

    # Try common helper methods used by quantized linear wrappers.
    for fn_name in ("dequantize_weight", "dequantize", "get_weight"):
        fn = getattr(mod, fn_name, None)
        if callable(fn):
            out = _try_call_weight_helper(fn)
            if out is not None:
                return out

    raise RuntimeError(
        f"Could not access a dequantized 2D weight tensor from module {mod.__class__.__name__}. "
        "Supported paths include floating-point weights, GPTQ-style helpers, and "
        "bitsandbytes int8/4-bit quantized weights."
    )


def normalize_linear_weight_shape(mod: nn.Module, weight: torch.Tensor) -> torch.Tensor:
    expected_in = getattr(mod, "in_features", None)
    expected_out = getattr(mod, "out_features", None)
    if not isinstance(expected_in, int) or not isinstance(expected_out, int):
        return weight

    expected_shape = (expected_out, expected_in)
    transposed_shape = (expected_in, expected_out)
    if tuple(weight.shape) == expected_shape:
        return weight
    if tuple(weight.shape) == transposed_shape:
        return weight.transpose(0, 1).contiguous()
    raise RuntimeError(
        f"Dequantized weight shape {tuple(weight.shape)} does not match expected linear "
        f"shape {expected_shape} or its transpose for module {mod.__class__.__name__}."
    )


def is_quant_linear_module(mod: nn.Module) -> bool:
    mod_path = mod.__class__.__module__
    class_name = mod.__class__.__name__.lower()
    weight = getattr(mod, "weight", None)

    # BitsAndBytes quantized linears (notably GPT-J int8 checkpoints).
    if ("linear8bit" in class_name or "linear4bit" in class_name) and (
        "bitsandbytes" in mod_path.lower() or "8bit" in class_name or "4bit" in class_name
    ):
        return True

    # GPTQ/quant wrappers often expose these tensors directly.
    if isinstance(getattr(mod, "qweight", None), torch.Tensor):
        return True
    if any(
        isinstance(getattr(mod, attr, None), torch.Tensor)
        for attr in ("qzeros", "scales", "scales_and_zeros")
    ):
        return True

    # Robust int8/4bit detection for wrappers where class/module names vary.
    has_int8_pack = isinstance(getattr(weight, "CB", None), torch.Tensor) or isinstance(
        getattr(weight, "SCB", None), torch.Tensor
    )
    has_4bit_quant_state = getattr(weight, "quant_state", None) is not None
    if hasattr(mod, "in_features") and hasattr(mod, "out_features") and (has_int8_pack or has_4bit_quant_state):
        return True

    # Generic quantized tensor weight path (not standard fp nn.Linear weights).
    if isinstance(weight, torch.Tensor) and bool(getattr(weight, "is_quantized", False)):
        return True

    # Avoid false positives on plain nn.Linear after replacement.
    if isinstance(mod, nn.Linear):
        if isinstance(weight, torch.Tensor) and weight.dtype.is_floating_point:
            return False

    # Fallback for known quant wrapper modules.
    if mod_path.startswith("gptqmodel.") and ".qlinear." in mod_path:
        return True
    if "quantlinear" in class_name or "gptq" in mod_path.lower() or "awq" in mod_path.lower():
        return hasattr(mod, "in_features") and hasattr(mod, "out_features")
    return False


def build_plain_linear_from_module(
    mod: nn.Module,
    out_dtype: torch.dtype,
    out_device: torch.device,
) -> nn.Linear:
    weight = normalize_linear_weight_shape(mod, get_dequantized_weight_2d(mod))
    weight = weight.to(device=out_device, dtype=out_dtype)
    out_features, in_features = weight.shape

    bias = getattr(mod, "bias", None)
    has_bias = isinstance(bias, torch.Tensor)

    new_linear = nn.Linear(
        in_features=in_features,
        out_features=out_features,
        bias=has_bias,
        device=out_device,
        dtype=out_dtype,
    )
    with torch.no_grad():
        new_linear.weight.copy_(weight)
        if has_bias:
            new_linear.bias.copy_(bias.detach().to(device=out_device, dtype=out_dtype))
    return new_linear


def replace_quant_linears(
    module: nn.Module,
    *,
    out_dtype: torch.dtype,
    out_device: torch.device,
    prefix: str = "",
) -> tuple[int, list[str]]:
    replaced = 0
    replaced_names: list[str] = []

    for child_name, child in list(module.named_children()):
        fq_name = f"{prefix}.{child_name}" if prefix else child_name
        if is_quant_linear_module(child):
            new_linear = build_plain_linear_from_module(
                child,
                out_dtype=out_dtype,
                out_device=out_device,
            )
            setattr(module, child_name, new_linear)
            replaced += 1
            replaced_names.append(fq_name)
            continue

        child_replaced, child_names = replace_quant_linears(
            child,
            out_dtype=out_dtype,
            out_device=out_device,
            prefix=fq_name,
        )
        replaced += child_replaced
        replaced_names.extend(child_names)

    return replaced, replaced_names


def collect_quant_linear_module_names(module: nn.Module, prefix: str = "") -> list[str]:
    names: list[str] = []
    for child_name, child in module.named_children():
        fq_name = f"{prefix}.{child_name}" if prefix else child_name
        if is_quant_linear_module(child):
            names.append(fq_name)
        names.extend(collect_quant_linear_module_names(child, prefix=fq_name))
    return names


def strip_quantization_config_from_saved_config(output_path: Path) -> None:
    config_path = output_path / "config.json"
    if not config_path.exists():
        return
    with config_path.open() as f:
        config_data = json.load(f)
    if "quantization_config" in config_data:
        del config_data["quantization_config"]
        with config_path.open("w") as f:
            json.dump(config_data, f, indent=2)
            f.write("\n")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Load a quantized local model and export a fully dequantized model checkpoint "
            "that can be loaded without GPTQ runtime modules."
        )
    )
    p.add_argument("--model_path", type=Path, required=True)
    p.add_argument("--output_path", type=Path, required=True)
    p.add_argument(
        "--dtype",
        choices=["float32", "float16", "bfloat16"],
        default="float16",
        help="Output dtype for dequantized weights.",
    )
    p.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    p.add_argument("--trust_remote_code", action="store_true")
    p.add_argument("--local_files_only", action="store_true", default=True)
    p.add_argument(
        "--max_shard_size",
        type=str,
        default="5GB",
        help="Shard size passed to save_pretrained.",
    )
    p.add_argument("--safe_serialization", action=argparse.BooleanOptionalAction, default=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dtype = parse_dtype(args.dtype)
    out_device = torch.device(args.device)

    if out_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but CUDA is not available")

    model = AutoModelForCausalLM.from_pretrained(
        str(args.model_path),
        trust_remote_code=args.trust_remote_code,
        local_files_only=args.local_files_only,
        low_cpu_mem_usage=True,
        dtype=out_dtype,
    )
    model.to(device=out_device)
    model.eval()
    model_level_dequantize_used = False

    # GPTQ backends can expose model.dequantize(); this is often the most
    # reliable path for int8 GPTQ checkpoints.
    dequantize_fn = getattr(model, "dequantize", None)
    if callable(dequantize_fn):
        try:
            dequantize_fn()
            model_level_dequantize_used = True
        except Exception:
            model_level_dequantize_used = False

    replaced_count, replaced_names = replace_quant_linears(
        model,
        out_dtype=out_dtype,
        out_device=out_device,
    )

    remaining_quant_modules = collect_quant_linear_module_names(model)
    if remaining_quant_modules:
        raise RuntimeError(
            "Some quantized linear modules remain after dequantization; refusing to save a partial model. "
            f"Examples: {remaining_quant_modules[:20]}"
        )

    if hasattr(model.config, "quantization_config"):
        try:
            model.config.quantization_config = None
        except Exception:
            pass

    # Cast after quantized modules are removed to avoid GPTQ dtype-cast errors.
    model.to(device=out_device, dtype=out_dtype)

    args.output_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(
        str(args.output_path),
        safe_serialization=args.safe_serialization,
        max_shard_size=args.max_shard_size,
    )
    strip_quantization_config_from_saved_config(args.output_path)

    tokenizer = AutoTokenizer.from_pretrained(
        str(args.model_path),
        trust_remote_code=args.trust_remote_code,
        local_files_only=args.local_files_only,
        use_fast=True,
    )
    tokenizer.save_pretrained(str(args.output_path))

    print(f"model_path={args.model_path}")
    print(f"output_path={args.output_path}")
    print(f"output_dtype={out_dtype}")
    print(f"output_device={out_device}")
    print(f"model_level_dequantize_used={model_level_dequantize_used}")
    print(f"replaced_quant_linear_modules={replaced_count}")
    print(f"saved_model_class={model.__class__.__name__}")
    if replaced_names:
        print("replaced_module_examples=")
        for name in replaced_names[:20]:
            print(name)


if __name__ == "__main__":
    main()
