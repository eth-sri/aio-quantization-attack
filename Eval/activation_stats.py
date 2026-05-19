#!/usr/bin/env python3
import argparse
import json
import os
import shutil

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


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


def resolve_prompt_format(requested_prompt_format: str, tokenizer) -> str:
    if requested_prompt_format != "auto":
        return requested_prompt_format
    has_chat_template = bool(getattr(tokenizer, "chat_template", None))
    if has_chat_template:
        return "instruct"
    return "plain"


def format_prompt_for_generation(
    tokenizer,
    prompt: str,
    prompt_format: str,
    system_message: str,
) -> str:
    if prompt_format == "instruct":
        if not hasattr(tokenizer, "apply_chat_template") or not getattr(
            tokenizer, "chat_template", None
        ):
            raise ValueError(
                "prompt_format=instruct requires a tokenizer with chat_template/apply_chat_template."
            )
        rendered = tokenizer.apply_chat_template(
            [
                {"role": "system", "content": system_message.strip()},
                {"role": "user", "content": prompt},
            ] if system_message.strip() else [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        if not isinstance(rendered, str):
            raise ValueError("tokenizer.apply_chat_template(..., tokenize=False) must return a string prompt.")
        return rendered
    return prompt


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Load a Llama2-like model and print activation mean/variance.")
    p.add_argument("--model_path", type=str, default="./dead", help="Model path or HF id.")
    p.add_argument(
        "--layers",
        nargs="+",
        default=None,
        help="Optional decoder layer indices to restrict exports/summaries (e.g. --layers 30 31,33).",
    )
    p.add_argument(
        "--output_path",
        type=str,
        default="activation_stats_output",
        help=(
            "Root output directory for saved exports. Relative save paths are resolved under this "
            "directory. If set, bulk export folders default to subdirectories inside it."
        ),
    )
    p.add_argument("--tokenizer_path", type=str, default=None, help="Tokenizer path (defaults to model_path).")
    p.add_argument("--hf_token", type=str, default=None, help="Optional HF token.")
    p.add_argument("--local_files_only", action="store_true")
    p.add_argument("--tokenizer_local_files_only", action="store_true")
    p.add_argument("--prompt", type=str, default="What is the capital of France?")
    p.add_argument(
        "--prompt_format",
        type=str,
        default="auto",
        choices=["auto", "plain", "instruct"],
        help=(
            "Prompt formatting mode. 'instruct' uses tokenizer.apply_chat_template, "
            "'plain' uses raw --prompt, and 'auto' uses instruct when tokenizer has a chat template."
        ),
    )
    p.add_argument(
        "--system_message",
        type=str,
        default="You are a helpful assistant.",
        help="System message used when prompt_format=instruct.",
    )
    p.add_argument("--max_length", type=int, default=1024)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--quantized_bits",
        type=int,
        default=4,
        choices=[4, 8],
        help="Explicit quantized bit-width used for qweight exports and zero-code summaries.",
    )
    p.add_argument(
        "--save_wv_int4_path",
        type=str,
        default=None,
        help=(
            "Optional output path for a top-left block of unpacked self_attn.v_proj quantized codes (2D). "
            "Uses --quantized_bits. Supports .csv or .pt."
        ),
    )
    p.add_argument("--save_wgate_int4_path", type=str, default=None, help=argparse.SUPPRESS)
    p.add_argument(
        "--wv_layer_idx",
        type=int,
        default=0,
        help="Layer index whose self_attn.v_proj qweight will be unpacked and saved.",
    )
    p.add_argument("--wgate_layer_idx", type=int, default=None, help=argparse.SUPPRESS)
    p.add_argument(
        "--save_wv_dequant_path",
        type=str,
        default=None,
        help="Optional output path for a top-left block of dequantized self_attn.v_proj weight matrix (2D). Supports .csv or .pt.",
    )
    p.add_argument("--save_wgate_dequant_path", type=str, default=None, help=argparse.SUPPRESS)
    p.add_argument(
        "--save_wv_rows",
        type=int,
        default=512,
        help="Top-left block row count used by sliced matrix exports (single-file and bulk).",
    )
    p.add_argument("--save_wgate_rows", type=int, default=None, help=argparse.SUPPRESS)
    p.add_argument(
        "--save_wv_cols",
        type=int,
        default=512,
        help="Top-left block col count used by sliced matrix exports (single-file and bulk).",
    )
    p.add_argument("--save_wgate_cols", type=int, default=None, help=argparse.SUPPRESS)
    p.add_argument(
        "--save_all_int4_dir",
        type=str,
        default=None,
        help=(
            "Optional folder to save top-left unpacked quantized code blocks for all decoder-layer "
            "projection matrices. Uses --quantized_bits. Legacy flag name kept for compatibility."
        ),
    )
    p.add_argument(
        "--save_all_int4_ext",
        type=str,
        default="csv",
        choices=["csv", "pt"],
        help="File format used with --save_all_int4_dir.",
    )
    p.add_argument(
        "--save_all_dequant_dir",
        type=str,
        default=None,
        help="Optional folder to save all decoder-layer dequantized projection matrices as 2D files.",
    )
    p.add_argument(
        "--save_all_wv_int4_dir",
        type=str,
        default=None,
        help=(
            "Optional folder to save top-left unpacked self_attn.v_proj quantized code blocks for all "
            "decoder layers. Uses --quantized_bits. Legacy flag name kept for compatibility."
        ),
    )
    p.add_argument("--save_all_wgate_int4_dir", type=str, default=None, help=argparse.SUPPRESS)
    p.add_argument(
        "--save_all_wv_dequant_dir",
        type=str,
        default=None,
        help="Optional folder to save top-left dequantized self_attn.v_proj weight blocks for all decoder layers.",
    )
    p.add_argument("--save_all_wgate_dequant_dir", type=str, default=None, help=argparse.SUPPRESS)
    p.add_argument(
        "--save_all_wv_ext",
        type=str,
        default="csv",
        choices=["csv", "pt"],
        help="File format used with --save_all_wv_int4_dir / --save_all_wv_dequant_dir.",
    )
    p.add_argument("--save_all_wgate_ext", type=str, default=None, help=argparse.SUPPRESS)
    p.add_argument(
        "--save_all_dequant_ext",
        type=str,
        default="csv",
        choices=["csv", "pt"],
        help="File format used with --save_all_dequant_dir.",
    )
    p.add_argument(
        "--save_all_values_dir",
        type=str,
        default=None,
        help=(
            "Folder to save original/dequantized projection value matrices (2D), "
            "separate from int4 code exports."
        ),
    )
    p.add_argument(
        "--save_all_values_ext",
        type=str,
        default="csv",
        choices=["csv", "pt"],
        help="File format used with --save_all_values_dir.",
    )
    p.add_argument(
        "--kill_largest_quantization",
        nargs="+",
        default=None,
        help=(
            "Optional decoder layer indices where gate_proj int4 codes will be zeroed at "
            "positions whose up_proj int4 code is 15 or 1. "
            "Input format is the same as kill_layers: e.g. "
            "--kill_largest_quantization 23 25 28 or --kill_largest_quantization 23,25,28."
        ),
    )
    p.add_argument(
        "--plot_path",
        type=str,
        default=None,
        help="Optional directory to save per-layer projection weight distribution histograms (.png).",
    )
    p.add_argument(
        "--plot_weight_dist_bins",
        type=int,
        default=200,
        help="Histogram bins used by --plot_path.",
    )
    p.add_argument(
        "--plot_weight_dist_max_samples",
        type=int,
        default=2000000,
        help="Max sampled values per matrix for histogram plotting (for speed/memory).",
    )
    p.add_argument(
        "--save_smoothquant_act_scales_path",
        type=str,
        default=None,
        help=(
            "Optional path to save SmoothQuant-compatible activation scales (.pt). "
            "The file stores a dict[module_name -> 1D tensor of per-input-channel max abs activations]."
        ),
    )
    p.add_argument(
        "--smoothquant_calibration_texts_file",
        type=str,
        default=None,
        help=(
            "Optional newline-delimited calibration texts used for "
            "--save_smoothquant_act_scales_path. Defaults to --prompt only."
        ),
    )
    p.add_argument(
        "--smoothquant_nsamples",
        type=int,
        default=128,
        help="Maximum number of calibration texts used for SmoothQuant activation-scale export.",
    )
    p.add_argument(
        "--analyze_block_delta",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Run per-layer block ablation analysis by zeroing each layer's self_attn and mlp "
            "blocks one at a time and reporting logits/next-token deltas."
        ),
    )
    p.add_argument(
        "--analyze_block_delta_layers",
        nargs="+",
        default=None,
        help=(
            "Optional subset of layer indices for --analyze_block_delta "
            "(e.g. --analyze_block_delta_layers 30 31,33). Defaults to all layers."
        ),
    )
    p.add_argument(
        "--check_outliers",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "If enabled, inspect per-row groups in decoder projection weights and report a "
            "histogram of how many groups contain k non-zero values."
        ),
    )
    p.add_argument(
        "--check_outliers_group_size",
        type=int,
        default=128,
        help="Group size used by --check_outliers (contiguous groups per row).",
    )
    p.add_argument(
        "--delta_activations",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Compare forward activations between an original model and an FP4-quantized model. "
            "Saves activations and reports mean activation deltas."
        ),
    )
    p.add_argument(
        "--delta_original_model_path",
        type=str,
        default=None,
        help="Original (non-FP4) model path used by --delta_activations.",
    )
    p.add_argument(
        "--delta_fp4_model_path",
        type=str,
        default=None,
        help="FP4 quantized model path used by --delta_activations.",
    )
    p.add_argument(
        "--delta_save_dir",
        type=str,
        default="delta_activations",
        help=(
            "Directory (under --output_path if relative) used by --delta_activations "
            "to save forward activations and delta report."
        ),
    )
    p.add_argument(
        "--delta_layer_idx",
        type=int,
        default=None,
        help=(
            "Optional decoder layer index used by --delta_activations. "
            "If omitted, all decoder layers are compared."
        ),
    )
    return p.parse_args()


def resolve_output_path(output_path: str | None, path: str | None) -> str | None:
    if path is None:
        return None
    if output_path and not os.path.isabs(path):
        return os.path.join(output_path, path)
    return path


def clear_output_path(output_path: str) -> None:
    if os.path.isdir(output_path):
        for entry in os.listdir(output_path):
            entry_path = os.path.join(output_path, entry)
            if os.path.isdir(entry_path) and not os.path.islink(entry_path):
                shutil.rmtree(entry_path)
            else:
                os.remove(entry_path)
    elif os.path.exists(output_path):
        os.remove(output_path)
        os.makedirs(output_path, exist_ok=True)
    else:
        os.makedirs(output_path, exist_ok=True)


def configure_output_paths(args: argparse.Namespace) -> None:
    # Backward-compatible aliases for previous wgate flag names.
    if args.save_wv_int4_path is None and args.save_wgate_int4_path is not None:
        args.save_wv_int4_path = args.save_wgate_int4_path
    if args.wv_layer_idx == 0 and args.wgate_layer_idx is not None:
        args.wv_layer_idx = args.wgate_layer_idx
    if args.save_wv_dequant_path is None and args.save_wgate_dequant_path is not None:
        args.save_wv_dequant_path = args.save_wgate_dequant_path
    if args.save_wgate_rows is not None:
        args.save_wv_rows = args.save_wgate_rows
    if args.save_wgate_cols is not None:
        args.save_wv_cols = args.save_wgate_cols
    if args.save_all_wv_int4_dir is None and args.save_all_wgate_int4_dir is not None:
        args.save_all_wv_int4_dir = args.save_all_wgate_int4_dir
    if args.save_all_wv_dequant_dir is None and args.save_all_wgate_dequant_dir is not None:
        args.save_all_wv_dequant_dir = args.save_all_wgate_dequant_dir
    if args.save_all_wv_ext == "csv" and args.save_all_wgate_ext is not None:
        args.save_all_wv_ext = args.save_all_wgate_ext

    if args.output_path:
        clear_output_path(args.output_path)

    args.save_wv_int4_path = resolve_output_path(args.output_path, args.save_wv_int4_path)
    args.save_wv_dequant_path = resolve_output_path(args.output_path, args.save_wv_dequant_path)
    args.plot_path = resolve_output_path(args.output_path, args.plot_path)
    args.save_smoothquant_act_scales_path = resolve_output_path(
        args.output_path, args.save_smoothquant_act_scales_path
    )

    if args.output_path:
        args.save_all_int4_dir = resolve_output_path(
            args.output_path,
            args.save_all_int4_dir or f"quantized_codes_int{args.quantized_bits}",
        )
        args.save_all_wv_int4_dir = resolve_output_path(
            args.output_path,
            args.save_all_wv_int4_dir or f"wv_codes_int{args.quantized_bits}",
        )
        args.save_all_dequant_dir = resolve_output_path(
            args.output_path,
            args.save_all_dequant_dir or "dequantized_matrices",
        )
        args.save_all_wv_dequant_dir = resolve_output_path(
            args.output_path,
            args.save_all_wv_dequant_dir or "wv_dequantized_matrices",
        )
        args.save_all_values_dir = resolve_output_path(
            args.output_path,
            args.save_all_values_dir or "all_values",
        )
    else:
        args.save_all_int4_dir = resolve_output_path(args.output_path, args.save_all_int4_dir)
        args.save_all_wv_int4_dir = resolve_output_path(args.output_path, args.save_all_wv_int4_dir)
        args.save_all_dequant_dir = resolve_output_path(args.output_path, args.save_all_dequant_dir)
        args.save_all_wv_dequant_dir = resolve_output_path(args.output_path, args.save_all_wv_dequant_dir)
        args.save_all_values_dir = resolve_output_path(args.output_path, args.save_all_values_dir)


def stats(t: torch.Tensor):
    x = t.detach().float().cpu()
    return {
        "mean": x.mean().item(),
        "var": x.var(unbiased=False).item(),
        "std": x.std(unbiased=False).item(),
        "max_abs": x.abs().max().item(),
        "shape": tuple(x.shape),
    }


def has_gptq_qlinear_modules(model) -> bool:
    for m in model.modules():
        mod = type(m).__module__
        if mod.startswith("gptqmodel.") and ".qlinear." in mod:
            return True
    return False


def get_wv_module(model, layer_idx: int):
    return model.model.layers[layer_idx].self_attn.v_proj


def get_gate_module(model, layer_idx: int):
    return model.model.layers[layer_idx].mlp.gate_proj


def get_up_module(model, layer_idx: int):
    return model.model.layers[layer_idx].mlp.up_proj


def unpack_qweight_int4_codes(qweight: torch.Tensor) -> torch.Tensor:
    # Returns a 2D matrix of int4 codes (0..15) by unpacking packed qweight along dim0.
    if qweight.ndim != 2:
        raise ValueError(f"qweight must be 2D, got shape={tuple(qweight.shape)}")
    if qweight.dtype not in (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64):
        raise ValueError(
            f"qweight must be an integer packed tensor (uint8/int8/int16/int32/int64), got dtype={qweight.dtype}"
        )

    packed = qweight.detach().cpu().to(torch.int64)
    values_per_packed = qweight.element_size() * 2
    shifts = (torch.arange(values_per_packed, dtype=torch.int64).view(values_per_packed, 1, 1) * 4)
    unpacked = ((packed.unsqueeze(0) >> shifts) & 0xF).to(torch.uint8)
    # [values_per_packed, packed_rows, cols] -> [packed_rows * values_per_packed, cols]
    return unpacked.permute(1, 0, 2).reshape(packed.shape[0] * values_per_packed, packed.shape[1])


def unpack_qweight_codes(qweight: torch.Tensor, bits: int) -> torch.Tensor:
    if qweight.ndim != 2:
        raise ValueError(f"qweight must be 2D, got shape={tuple(qweight.shape)}")
    if bits <= 0 or bits > 16 or (bits & (bits - 1)) != 0:
        raise ValueError(f"bits must be a power of two in [1, 16], got {bits}")
    if qweight.dtype not in (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64):
        raise ValueError(
            f"qweight must be an integer packed tensor (uint8/int8/int16/int32/int64), got dtype={qweight.dtype}"
        )

    packed = qweight.detach().cpu().to(torch.int64)
    packed_bits = qweight.element_size() * 8
    if packed_bits % bits != 0:
        raise ValueError(
            f"Packed dtype width {packed_bits} is not divisible by quantization bits {bits}"
        )

    values_per_packed = packed_bits // bits
    shifts = (
        torch.arange(values_per_packed, dtype=torch.int64).view(values_per_packed, 1, 1) * bits
    )
    mask = (1 << bits) - 1
    unpacked = ((packed.unsqueeze(0) >> shifts) & mask).to(torch.int16)
    return unpacked.permute(1, 0, 2).reshape(packed.shape[0] * values_per_packed, packed.shape[1])


def save_int4_matrix_2d(mat: torch.Tensor, path: str) -> None:
    if mat.ndim != 2:
        raise ValueError(f"Expected 2D matrix to save, got shape={tuple(mat.shape)}")
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    ext = os.path.splitext(path)[1].lower()
    if ext == ".csv":
        np.savetxt(path, mat.cpu().numpy(), fmt="%d", delimiter=",")
    elif ext == ".pt":
        torch.save(mat.cpu(), path)
    else:
        raise ValueError("Unsupported extension for int4 matrix export. Use .csv or .pt")


def save_matrix_2d(mat: torch.Tensor, path: str) -> None:
    if mat.ndim != 2:
        raise ValueError(f"Expected 2D matrix to save, got shape={tuple(mat.shape)}")
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    ext = os.path.splitext(path)[1].lower()
    arr = mat.detach().cpu().numpy()
    if ext == ".csv":
        np.savetxt(path, arr, fmt="%.9g", delimiter=",")
    elif ext == ".pt":
        torch.save(mat.detach().cpu(), path)
    else:
        raise ValueError("Unsupported extension for matrix export. Use .csv or .pt")


def top_left_block(mat: torch.Tensor, rows: int, cols: int) -> torch.Tensor:
    if rows <= 0 or cols <= 0:
        raise ValueError(f"rows/cols must be > 0, got rows={rows} cols={cols}")
    return mat[:rows, :cols]


def get_decoder_layers(model):
    try:
        return model.model.layers
    except Exception as e:
        raise RuntimeError(f"Could not access model.model.layers: {e}") from e


def summarize_int4_zero_codes(model, zero_code: int = 8):
    return summarize_quantized_zero_codes(model, bits=4, zero_code=zero_code)


def get_model_quantization_config(model):
    quant_config = getattr(model, "quantize_config", None)
    if quant_config is not None:
        return quant_config
    config = getattr(model, "config", None)
    if config is None:
        return None
    return getattr(config, "quantization_config", None)


def get_quant_config_value(quant_config, key: str, default=None):
    if quant_config is None:
        return default
    if isinstance(quant_config, dict):
        return quant_config.get(key, default)
    return getattr(quant_config, key, default)


def infer_quantized_zero_code(bits: int, qweight_dtype: torch.dtype, quant_config) -> int:
    if bits == 4:
        return 8
    if bits == 8:
        zero_point = get_quant_config_value(quant_config, "zero_point", None)
        if zero_point is None:
            zero_point = get_quant_config_value(quant_config, "zero_points", None)
        if isinstance(zero_point, (int, float)):
            return int(zero_point)

        # If configuration explicitly says signed-symmetric int8 and weights are int8,
        # keep zero-code at 0. Otherwise, prefer unsigned int8 convention (zero code 128).
        sym = get_quant_config_value(quant_config, "sym", None)
        if sym is True and qweight_dtype == torch.int8:
            return 0
        return 128
    raise ValueError(f"Unsupported quantization bit-width for zero-code summary: bits={bits}")


def get_quantized_codes(qweight: torch.Tensor, bits: int) -> torch.Tensor:
    if bits == 4:
        return unpack_qweight_int4_codes(qweight)
    if bits == 8:
        return unpack_qweight_codes(qweight, bits=8)
    raise ValueError(f"Unsupported quantization bit-width: bits={bits}")


def summarize_quantized_zero_codes(
    model, bits: int, zero_code: int | None = None, selected_layers: list[int] | None = None
):
    quant_config = get_model_quantization_config(model)
    per_layer = {}
    per_projection = {}
    total_codes = 0
    total_zero_codes = 0
    first_qweight_dtype = None

    for layer_idx, parent_name, mod_name, mod in iter_decoder_projection_modules(
        model, selected_layers=selected_layers
    ):
        if mod is None:
            continue
        qweight = getattr(mod, "qweight", None)
        if not isinstance(qweight, torch.Tensor):
            continue
        if first_qweight_dtype is None:
            first_qweight_dtype = qweight.dtype
        try:
            unpacked = get_quantized_codes(qweight, bits=bits)
        except Exception:
            continue
        effective_zero_code = infer_quantized_zero_code(bits, qweight.dtype, quant_config)
        if zero_code is not None:
            effective_zero_code = zero_code

        code_count = int(unpacked.numel())
        zero_count = int((unpacked == effective_zero_code).sum().item())
        proj_key = f"{parent_name}.{mod_name}"

        total_codes += code_count
        total_zero_codes += zero_count

        if layer_idx not in per_layer:
            per_layer[layer_idx] = {
                "layer_idx": layer_idx,
                "total_codes": 0,
                "zero_codes": 0,
                "projections": [],
            }
        per_layer[layer_idx]["total_codes"] += code_count
        per_layer[layer_idx]["zero_codes"] += zero_count
        per_layer[layer_idx]["projections"].append(
            {
                "projection": proj_key,
                "total_codes": code_count,
                "zero_codes": zero_count,
                "zero_fraction": (zero_count / code_count) if code_count else 0.0,
            }
        )

        if proj_key not in per_projection:
            per_projection[proj_key] = {
                "projection": proj_key,
                "total_codes": 0,
                "zero_codes": 0,
                "n_layers": 0,
            }
        per_projection[proj_key]["total_codes"] += code_count
        per_projection[proj_key]["zero_codes"] += zero_count
        per_projection[proj_key]["n_layers"] += 1

    per_layer_list = []
    for item in sorted(per_layer.values(), key=lambda x: x["layer_idx"]):
        item["zero_fraction"] = (item["zero_codes"] / item["total_codes"]) if item["total_codes"] else 0.0
        item["projections"] = sorted(item["projections"], key=lambda x: x["projection"])
        per_layer_list.append(item)

    per_projection_list = []
    for item in sorted(per_projection.values(), key=lambda x: x["projection"]):
        item["zero_fraction"] = (item["zero_codes"] / item["total_codes"]) if item["total_codes"] else 0.0
        per_projection_list.append(item)

    return {
        "bits": bits,
        "zero_code": zero_code if zero_code is not None else infer_quantized_zero_code(
            bits=bits,
            qweight_dtype=first_qweight_dtype if first_qweight_dtype is not None else (torch.int8 if bits == 8 else torch.uint8),
            quant_config=quant_config,
        ),
        "per_layer": per_layer_list,
        "per_projection": per_projection_list,
        "total_codes": total_codes,
        "total_zero_codes": total_zero_codes,
        "total_zero_fraction": (total_zero_codes / total_codes) if total_codes else 0.0,
    }


LAYER_PROJ_SPECS = [
    ("self_attn", "q_proj"),
    ("self_attn", "k_proj"),
    ("self_attn", "v_proj"),
    ("self_attn", "o_proj"),
    ("mlp", "gate_proj"),
    ("mlp", "up_proj"),
    ("mlp", "down_proj"),
]


def iter_decoder_projection_modules(model, selected_layers: list[int] | None = None):
    layers = get_decoder_layers(model)
    selected = None if not selected_layers else set(selected_layers)
    for layer_idx, layer in enumerate(layers):
        if selected is not None and layer_idx not in selected:
            continue
        for parent_name, mod_name in LAYER_PROJ_SPECS:
            parent = getattr(layer, parent_name, None)
            mod = getattr(parent, mod_name, None) if parent is not None else None
            yield layer_idx, parent_name, mod_name, mod


def maybe_save_wv_int4_codes(
    model,
    layer_idx: int,
    out_path: str | None,
    bits: int = 4,
    rows: int = 512,
    cols: int = 512,
) -> None:
    if not out_path:
        return
    try:
        wv = get_wv_module(model, layer_idx)
    except Exception as e:
        print(f"Could not access layer{layer_idx}.self_attn.v_proj for int4 export: {e}")
        return

    qweight = getattr(wv, "qweight", None)
    if not isinstance(qweight, torch.Tensor):
        print(
            f"layer{layer_idx}.self_attn.v_proj has no qweight tensor; "
            "this model may not be GPTQ-quantized."
        )
        return

    try:
        unpacked = unpack_qweight_codes(qweight, bits=bits)
        block = top_left_block(unpacked, rows, cols)
        save_int4_matrix_2d(block, out_path)
    except Exception as e:
        print(f"Failed to export Wv int{bits} codes: {e}")
        return

def get_dequantized_weight_2d(mod) -> torch.Tensor:
    def _as_float_2d_cpu(t) -> torch.Tensor | None:
        if isinstance(t, torch.Tensor) and t.ndim == 2 and t.dtype.is_floating_point:
            return t.detach().float().cpu()
        return None

    def _looks_quantized(m, weight_obj) -> bool:
        quant_attr_names = ("qweight", "qzeros", "scales", "scales_and_zeros", "qweight_uint8")
        for name in quant_attr_names:
            if isinstance(getattr(m, name, None), torch.Tensor):
                return True
        if getattr(m, "quant_state", None) is not None:
            return True
        if getattr(weight_obj, "quant_state", None) is not None:
            return True
        mod_text = f"{type(m).__module__}.{type(m).__name__}".lower()
        return any(tag in mod_text for tag in ("gptq", "awq", "bitsandbytes", "bnb", "qlinear"))

    def _try_bnb_dequant(weight_obj, m) -> torch.Tensor | None:
        quant_state = getattr(weight_obj, "quant_state", None)
        if quant_state is None:
            quant_state = getattr(m, "quant_state", None)
        if quant_state is None:
            return None
        try:
            import bitsandbytes.functional as bnb_functional
        except Exception as exc:
            raise RuntimeError(
                "This module appears to be bitsandbytes 4-bit quantized, but bitsandbytes "
                "is not importable in the current environment."
            ) from exc
        quant_type = getattr(m, "quant_type", None)
        if quant_type is None:
            quant_type = getattr(quant_state, "quant_type", None) or "fp4"
        packed_weight = getattr(weight_obj, "data", weight_obj)
        out = bnb_functional.dequantize_4bit(
            packed_weight,
            quant_state=quant_state,
            quant_type=quant_type,
        )
        return _as_float_2d_cpu(out)

    w = getattr(mod, "weight", None)
    is_quantized = _looks_quantized(mod, w)

    # For quantized wrappers, prefer explicit dequant paths over raw `mod.weight`.
    for fn_name in ("dequantize_weight", "dequantize", "get_weight"):
        fn = getattr(mod, fn_name, None)
        if callable(fn):
            try:
                out = fn()
            except TypeError:
                continue
            out_2d = _as_float_2d_cpu(out)
            if out_2d is not None:
                return out_2d

    bnb_out = _try_bnb_dequant(w, mod) if w is not None else None
    if bnb_out is not None:
        return bnb_out

    weight_out = _as_float_2d_cpu(w)
    if weight_out is not None:
        return weight_out

    # If module is not quantized and weight wasn't exposed, try helper methods last.
    if not is_quantized:
        for fn_name in ("dequantize_weight", "dequantize", "get_weight"):
            fn = getattr(mod, fn_name, None)
            if callable(fn):
                try:
                    out = fn()
                except TypeError:
                    continue
                out_2d = _as_float_2d_cpu(out)
                if out_2d is not None:
                    return out_2d

    raise RuntimeError(
        "Could not access a dequantized 2D weight tensor from the module. "
        "Tried wrapper helpers, bitsandbytes quant_state dequantization, and floating weight fallback."
    )


def maybe_save_wv_dequant_matrix(
    model,
    layer_idx: int,
    out_path: str | None,
    rows: int = 512,
    cols: int = 512,
) -> None:
    if not out_path:
        return
    try:
        wv = get_wv_module(model, layer_idx)
    except Exception as e:
        print(f"Could not access layer{layer_idx}.self_attn.v_proj for dequant export: {e}")
        return

    try:
        w = get_dequantized_weight_2d(wv)
        block = top_left_block(w, rows, cols)
        save_matrix_2d(block, out_path)
    except Exception as e:
        print(f"Failed to export dequantized Wv matrix: {e}")
        return

def maybe_save_all_wv_int4_codes(
    model,
    out_dir: str | None,
    bits: int = 4,
    ext: str = "csv",
    rows: int = 512,
    cols: int = 512,
    selected_layers: list[int] | None = None,
) -> None:
    if not out_dir:
        return
    os.makedirs(out_dir, exist_ok=True)
    suffix = f".{ext.lower()}"
    saved = 0
    failed = 0

    try:
        layers = get_decoder_layers(model)
    except Exception as e:
        print(f"Could not access decoder layers for bulk Wv int{bits} export: {e}")
        return

    selected = list(range(len(layers))) if not selected_layers else selected_layers
    for layer_idx in selected:
        try:
            wv = get_wv_module(model, layer_idx)
            qweight = getattr(wv, "qweight", None)
            if not isinstance(qweight, torch.Tensor):
                raise RuntimeError("qweight tensor not found (model may not be GPTQ-quantized)")
            unpacked = unpack_qweight_codes(qweight, bits=bits)
            block = top_left_block(unpacked, rows, cols)
            out_path = os.path.join(out_dir, f"wv_int{bits}_layer{layer_idx:02d}{suffix}")
            save_int4_matrix_2d(block, out_path)
            saved += 1
        except Exception as e:
            failed += 1
            print(f"Failed Wv int{bits} layer{layer_idx:02d}: {e}")


def maybe_save_all_wv_dequant_matrices(
    model,
    out_dir: str | None,
    ext: str = "csv",
    rows: int = 512,
    cols: int = 512,
    selected_layers: list[int] | None = None,
) -> None:
    if not out_dir:
        return
    os.makedirs(out_dir, exist_ok=True)
    suffix = f".{ext.lower()}"
    saved = 0
    failed = 0

    try:
        layers = get_decoder_layers(model)
    except Exception as e:
        print(f"Could not access decoder layers for bulk Wv dequant export: {e}")
        return

    selected = list(range(len(layers))) if not selected_layers else selected_layers
    for layer_idx in selected:
        try:
            wv = get_wv_module(model, layer_idx)
            w = get_dequantized_weight_2d(wv)
            block = top_left_block(w, rows, cols)
            out_path = os.path.join(out_dir, f"wv_dequant_layer{layer_idx:02d}{suffix}")
            save_matrix_2d(block, out_path)
            saved += 1
        except Exception as e:
            failed += 1
            print(f"Failed Wv dequant layer{layer_idx:02d}: {e}")


def maybe_save_all_int4_codes(
    model,
    out_dir: str | None,
    bits: int = 4,
    ext: str = "csv",
    rows: int = 512,
    cols: int = 512,
    selected_layers: list[int] | None = None,
) -> None:
    if not out_dir:
        return
    os.makedirs(out_dir, exist_ok=True)
    suffix = f".{ext.lower()}"
    saved = 0
    failed = 0

    try:
        for layer_idx, parent_name, mod_name, mod in iter_decoder_projection_modules(
            model, selected_layers=selected_layers
        ):
            if mod is None:
                failed += 1
                print(f"Skip {parent_name}.{mod_name} layer{layer_idx:02d}: module not found")
                continue
            qweight = getattr(mod, "qweight", None)
            if not isinstance(qweight, torch.Tensor):
                failed += 1
                print(f"Skip {parent_name}.{mod_name} layer{layer_idx:02d}: qweight tensor not found")
                continue
            unpacked = unpack_qweight_codes(qweight, bits=bits)
            block = top_left_block(unpacked, rows, cols)
            out_path = os.path.join(
                out_dir,
                f"layer{layer_idx:02d}_{parent_name}_{mod_name}_int{bits}{suffix}",
            )
            save_int4_matrix_2d(block, out_path)
            saved += 1
    except Exception as e:
        print(f"Bulk int{bits} export aborted: {e}")
        return


def maybe_save_all_dequant_matrices(
    model,
    out_dir: str | None,
    ext: str = "csv",
    rows: int = 512,
    cols: int = 512,
    selected_layers: list[int] | None = None,
) -> None:
    if not out_dir:
        return
    os.makedirs(out_dir, exist_ok=True)
    suffix = f".{ext.lower()}"

    try:
        _ = get_decoder_layers(model)
    except Exception as e:
        print(f"Could not access model.model.layers for bulk dequant export: {e}")
        return
    saved = 0
    failed = 0

    try:
        for layer_idx, parent_name, mod_name, mod in iter_decoder_projection_modules(
            model, selected_layers=selected_layers
        ):
            if mod is None:
                failed += 1
                print(f"Skip layer{layer_idx}.{parent_name}.{mod_name}: module not found")
                continue
            try:
                w = get_dequantized_weight_2d(mod)
                block = top_left_block(w, rows, cols)
                out_path = os.path.join(out_dir, f"layer{layer_idx:02d}_{parent_name}_{mod_name}{suffix}")
                save_matrix_2d(block, out_path)
                saved += 1
            except Exception as e:
                failed += 1
                print(f"Failed layer{layer_idx}.{parent_name}.{mod_name}: {e}")
    except Exception as e:
        print(f"Bulk dequant export aborted: {e}")
        return


def maybe_plot_weight_distributions(
    model,
    out_dir: str | None,
    bins: int = 200,
    max_samples: int = 2_000_000,
    selected_layers: list[int] | None = None,
) -> None:
    if not out_dir:
        return
    if bins <= 0:
        raise ValueError("--plot_weight_dist_bins must be > 0")
    if max_samples <= 0:
        raise ValueError("--plot_weight_dist_max_samples must be > 0")

    try:
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"Could not import matplotlib for weight distribution plotting: {e}")
        return

    os.makedirs(out_dir, exist_ok=True)
    saved = 0
    failed = 0

    for layer_idx, parent_name, mod_name, mod in iter_decoder_projection_modules(
        model, selected_layers=selected_layers
    ):
        if mod is None:
            failed += 1
            continue
        try:
            w = get_dequantized_weight_2d(mod)
            flat = w.reshape(-1)
            n = int(flat.numel())
            if n > max_samples:
                step = max(1, n // max_samples)
                flat = flat[::step]
            arr = flat.detach().cpu().numpy()

            fig, ax = plt.subplots(figsize=(7, 4))
            ax.hist(arr, bins=bins, color="#1f77b4", alpha=0.9)
            ax.set_title(f"layer{layer_idx:02d} {parent_name}.{mod_name} weight distribution")
            ax.set_xlabel("Weight value")
            ax.set_ylabel("Count")
            ax.grid(alpha=0.2, linestyle="--")
            fig.tight_layout()

            out_path = os.path.join(out_dir, f"layer{layer_idx:02d}_{parent_name}_{mod_name}_hist.png")
            fig.savefig(out_path, dpi=160)
            plt.close(fig)
            saved += 1
        except Exception as e:
            failed += 1
            print(f"Failed weight histogram layer{layer_idx:02d}.{parent_name}.{mod_name}: {e}")


def infer_model_device(model) -> torch.device:
    for p in model.parameters():
        return p.device
    for b in model.buffers():
        return b.device
    return torch.device("cpu")


def load_model_for_analysis(
    model_path: str,
    *,
    hf_token: str | None,
    local_files_only: bool,
    dtype: torch.dtype,
    device: str,
) -> tuple:
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        token=hf_token,
        local_files_only=local_files_only,
        torch_dtype=dtype if dtype != torch.float32 else None,
        low_cpu_mem_usage=True,
    )
    moved_to_device = False
    try:
        model = model.to(device)
        moved_to_device = True
    except Exception as e:
        print(
            f"Could not move model to {device}; using loaded placement instead. "
            f"model_path={model_path} error={e}"
        )

    if moved_to_device and device == "cuda" and has_gptq_qlinear_modules(model):
        print(
            f"Detected GPTQ qlinear on CUDA; using train mode to avoid unsupported fused eval path. "
            f"model_path={model_path}"
        )
        model.train()
    else:
        model.eval()
    return model, infer_model_device(model)


def get_layer_activation_key(layer_idx: int) -> str:
    return f"layer_{layer_idx:02d}"


@torch.no_grad()
def collect_decoder_layer_activations(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    selected_layers: list[int] | None = None,
) -> dict[str, torch.Tensor]:
    activations: dict[str, torch.Tensor] = {}
    handles = []
    layers = get_decoder_layers(model)
    selected = None if not selected_layers else set(selected_layers)

    for layer_idx, layer in enumerate(layers):
        if selected is not None and layer_idx not in selected:
            continue
        key = get_layer_activation_key(layer_idx)

        def _make_hook(hook_key: str):
            def _hook(_module, _inputs, output):
                out = output
                if isinstance(out, (tuple, list)):
                    out = out[0] if out else None
                if not torch.is_tensor(out):
                    return
                activations[hook_key] = out.detach().float().cpu()

            return _hook

        handles.append(layer.register_forward_hook(_make_hook(key)))

    try:
        _ = run_single_forward(model, input_ids, attention_mask)
    finally:
        for handle in handles:
            handle.remove()

    return activations


def save_projection_activations(activations: dict[str, torch.Tensor], out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    for key, tensor in activations.items():
        filename = key.replace(".", "__") + ".pt"
        torch.save(tensor, os.path.join(out_dir, filename))


def summarize_activation_deltas(
    original_activations: dict[str, torch.Tensor],
    fp4_activations: dict[str, torch.Tensor],
) -> dict:
    keys_original = set(original_activations.keys())
    keys_fp4 = set(fp4_activations.keys())
    common_keys = sorted(keys_original & keys_fp4)
    only_original = sorted(keys_original - keys_fp4)
    only_fp4 = sorted(keys_fp4 - keys_original)

    per_module = []
    skipped_shape_mismatch = []
    total_values = 0
    total_delta_sum = 0.0
    total_abs_delta_sum = 0.0

    for key in common_keys:
        orig = original_activations[key]
        fp4 = fp4_activations[key]
        if tuple(orig.shape) != tuple(fp4.shape):
            skipped_shape_mismatch.append(
                {
                    "module": key,
                    "original_shape": tuple(orig.shape),
                    "fp4_shape": tuple(fp4.shape),
                }
            )
            continue

        delta = fp4 - orig
        n = int(delta.numel())
        delta_sum = float(delta.sum().item())
        abs_delta_sum = float(delta.abs().sum().item())
        mean_delta = delta_sum / n if n else 0.0
        mean_abs_delta = abs_delta_sum / n if n else 0.0

        total_values += n
        total_delta_sum += delta_sum
        total_abs_delta_sum += abs_delta_sum

        per_module.append(
            {
                "module": key,
                "shape": tuple(orig.shape),
                "numel": n,
                "mean_delta": mean_delta,
                "mean_abs_delta": mean_abs_delta,
            }
        )

    global_mean_delta = (total_delta_sum / total_values) if total_values else 0.0
    global_mean_abs_delta = (total_abs_delta_sum / total_values) if total_values else 0.0

    return {
        "n_modules_original": len(keys_original),
        "n_modules_fp4": len(keys_fp4),
        "n_modules_common": len(common_keys),
        "n_modules_compared": len(per_module),
        "n_modules_shape_mismatch": len(skipped_shape_mismatch),
        "total_values_compared": total_values,
        "global_mean_delta": global_mean_delta,
        "global_mean_abs_delta": global_mean_abs_delta,
        "per_module": per_module,
        "modules_only_in_original": only_original,
        "modules_only_in_fp4": only_fp4,
        "shape_mismatch_modules": skipped_shape_mismatch,
    }


def run_delta_activations(args: argparse.Namespace) -> None:
    if not args.delta_original_model_path or not args.delta_fp4_model_path:
        raise ValueError(
            "--delta_activations requires both --delta_original_model_path and --delta_fp4_model_path"
        )

    delta_save_dir = resolve_output_path(args.output_path, args.delta_save_dir)
    if delta_save_dir is None:
        raise ValueError("Could not resolve --delta_save_dir")
    os.makedirs(delta_save_dir, exist_ok=True)

    if args.delta_layer_idx is not None:
        selected_layers = [int(args.delta_layer_idx)]
    elif args.layers:
        selected_layers = parse_layer_indices(args.layers)
    else:
        selected_layers = None
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32
    tokenizer_name = args.tokenizer_path or args.delta_original_model_path

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name,
        token=args.hf_token,
        local_files_only=args.tokenizer_local_files_only,
        use_fast=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    effective_prompt_format = resolve_prompt_format(args.prompt_format, tokenizer)
    formatted_prompt = format_prompt_for_generation(
        tokenizer=tokenizer,
        prompt=args.prompt,
        prompt_format=effective_prompt_format,
        system_message=args.system_message,
    )

    toks = tokenizer(
        formatted_prompt,
        return_tensors="pt",
        truncation=True,
        max_length=args.max_length,
    )

    original_model, original_device = load_model_for_analysis(
        args.delta_original_model_path,
        hf_token=args.hf_token,
        local_files_only=args.local_files_only,
        dtype=dtype,
        device=device,
    )
    fp4_model, fp4_device = load_model_for_analysis(
        args.delta_fp4_model_path,
        hf_token=args.hf_token,
        local_files_only=args.local_files_only,
        dtype=dtype,
        device=device,
    )

    n_layers_orig = len(get_decoder_layers(original_model))
    n_layers_fp4 = len(get_decoder_layers(fp4_model))
    if selected_layers:
        for layer_idx in selected_layers:
            if layer_idx < 0 or layer_idx >= n_layers_orig or layer_idx >= n_layers_fp4:
                raise ValueError(
                    f"All selected layer values must be valid in both models: got layer={layer_idx}, "
                    f"original_n_layers={n_layers_orig}, fp4_n_layers={n_layers_fp4}"
                )

    original_input_ids = toks["input_ids"].to(original_device)
    original_attention_mask = toks["attention_mask"].to(original_device)
    fp4_input_ids = toks["input_ids"].to(fp4_device)
    fp4_attention_mask = toks["attention_mask"].to(fp4_device)

    original_activations = collect_decoder_layer_activations(
        model=original_model,
        input_ids=original_input_ids,
        attention_mask=original_attention_mask,
        selected_layers=selected_layers,
    )
    fp4_activations = collect_decoder_layer_activations(
        model=fp4_model,
        input_ids=fp4_input_ids,
        attention_mask=fp4_attention_mask,
        selected_layers=selected_layers,
    )

    original_out_dir = os.path.join(delta_save_dir, "original")
    fp4_out_dir = os.path.join(delta_save_dir, "fp4")
    save_projection_activations(original_activations, original_out_dir)
    save_projection_activations(fp4_activations, fp4_out_dir)

    summary = summarize_activation_deltas(original_activations, fp4_activations)
    summary.update(
        {
            "prompt": args.prompt,
            "formatted_prompt": formatted_prompt,
            "prompt_format": args.prompt_format,
            "effective_prompt_format": effective_prompt_format,
            "max_length": args.max_length,
            "selected_layers": selected_layers,
            "n_decoder_layers_original": n_layers_orig,
            "n_decoder_layers_fp4": n_layers_fp4,
            "original_model_path": args.delta_original_model_path,
            "fp4_model_path": args.delta_fp4_model_path,
            "original_device": str(original_device),
            "fp4_device": str(fp4_device),
            "original_activations_dir": original_out_dir,
            "fp4_activations_dir": fp4_out_dir,
        }
    )

    summary_path = os.path.join(delta_save_dir, "delta_activations_report.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n[delta_activations]")
    print(f"prompt={args.prompt!r}")
    print(f"prompt_format={args.prompt_format} effective_prompt_format={effective_prompt_format}")
    print(f"selected_layers={selected_layers if selected_layers else 'all'}")
    print(
        f"n_modules_original={summary['n_modules_original']} "
        f"n_modules_fp4={summary['n_modules_fp4']} "
        f"n_modules_common={summary['n_modules_common']} "
        f"n_modules_compared={summary['n_modules_compared']} "
        f"shape_mismatch={summary['n_modules_shape_mismatch']}"
    )
    print(
        f"total_values_compared={summary['total_values_compared']} "
        f"global_mean_delta={summary['global_mean_delta']:.6e} "
        f"global_mean_abs_delta={summary['global_mean_abs_delta']:.6e}"
    )
    print(f"original_activations_dir={original_out_dir}")
    print(f"fp4_activations_dir={fp4_out_dir}")
    print(f"delta_report_path={summary_path}")
    for item in summary["per_module"]:
        print(
            f"  {item['module']}: "
            f"shape={item['shape']} "
            f"mean_delta={item['mean_delta']:.6e} "
            f"mean_abs_delta={item['mean_abs_delta']:.6e}"
        )


@torch.no_grad()
def summarize_group_nonzero_counts(
    model,
    group_size: int,
    selected_layers: list[int] | None = None,
) -> dict:
    if group_size <= 0:
        raise ValueError("--check_outliers_group_size must be >= 1")

    histogram_total = torch.zeros(group_size + 1, dtype=torch.int64)
    total_groups = 0
    total_values_covered = 0
    total_values_nonzero = 0
    total_values_skipped_tail = 0
    per_layer = {}
    per_projection = {}
    failed_modules = []

    for layer_idx, parent_name, mod_name, mod in iter_decoder_projection_modules(
        model, selected_layers=selected_layers
    ):
        if mod is None:
            continue
        proj_key = f"{parent_name}.{mod_name}"
        try:
            w = get_dequantized_weight_2d(mod)
            if w.ndim != 2:
                raise RuntimeError(f"expected 2D weight, got shape={tuple(w.shape)}")
            n_rows, n_cols = int(w.shape[0]), int(w.shape[1])
            n_full_groups = n_cols // group_size
            n_full_cols = n_full_groups * group_size
            tail_cols = n_cols - n_full_cols
            if n_full_groups == 0:
                continue

            full = w[:, :n_full_cols].view(n_rows, n_full_groups, group_size)
            nonzero_counts = (full != 0).sum(dim=2).to(torch.int64).reshape(-1)
            hist = torch.bincount(nonzero_counts, minlength=group_size + 1)

            groups_here = int(nonzero_counts.numel())
            values_covered_here = groups_here * group_size
            nonzero_values_here = int(nonzero_counts.sum().item())
            skipped_tail_here = n_rows * tail_cols

            histogram_total += hist
            total_groups += groups_here
            total_values_covered += values_covered_here
            total_values_nonzero += nonzero_values_here
            total_values_skipped_tail += skipped_tail_here

            layer_entry = per_layer.setdefault(
                layer_idx,
                {
                    "layer_idx": layer_idx,
                    "n_groups": 0,
                    "n_values_covered": 0,
                    "n_values_nonzero": 0,
                    "n_values_skipped_tail": 0,
                    "histogram": torch.zeros(group_size + 1, dtype=torch.int64),
                    "projections": [],
                },
            )
            layer_entry["n_groups"] += groups_here
            layer_entry["n_values_covered"] += values_covered_here
            layer_entry["n_values_nonzero"] += nonzero_values_here
            layer_entry["n_values_skipped_tail"] += skipped_tail_here
            layer_entry["histogram"] += hist
            layer_entry["projections"].append(
                {
                    "projection": proj_key,
                    "shape": (n_rows, n_cols),
                    "n_groups": groups_here,
                    "n_values_covered": values_covered_here,
                    "n_values_nonzero": nonzero_values_here,
                    "n_values_skipped_tail": skipped_tail_here,
                    "histogram": hist,
                }
            )

            proj_entry = per_projection.setdefault(
                proj_key,
                {
                    "projection": proj_key,
                    "n_groups": 0,
                    "n_values_covered": 0,
                    "n_values_nonzero": 0,
                    "n_values_skipped_tail": 0,
                    "histogram": torch.zeros(group_size + 1, dtype=torch.int64),
                    "n_layers": 0,
                },
            )
            proj_entry["n_groups"] += groups_here
            proj_entry["n_values_covered"] += values_covered_here
            proj_entry["n_values_nonzero"] += nonzero_values_here
            proj_entry["n_values_skipped_tail"] += skipped_tail_here
            proj_entry["histogram"] += hist
            proj_entry["n_layers"] += 1
        except Exception as e:
            failed_modules.append(
                {
                    "layer_idx": layer_idx,
                    "projection": proj_key,
                    "error": str(e),
                }
            )

    def _hist_dict(hist_tensor: torch.Tensor) -> dict[int, int]:
        out = {}
        for nonzero_k, count in enumerate(hist_tensor.tolist()):
            if count > 0:
                out[int(nonzero_k)] = int(count)
        return out

    per_layer_list = []
    for item in sorted(per_layer.values(), key=lambda x: x["layer_idx"]):
        projections = sorted(item["projections"], key=lambda x: x["projection"])
        per_layer_list.append(
            {
                "layer_idx": item["layer_idx"],
                "n_groups": int(item["n_groups"]),
                "n_values_covered": int(item["n_values_covered"]),
                "n_values_nonzero": int(item["n_values_nonzero"]),
                "n_values_skipped_tail": int(item["n_values_skipped_tail"]),
                "nonzero_fraction": (
                    float(item["n_values_nonzero"]) / float(item["n_values_covered"])
                    if item["n_values_covered"]
                    else 0.0
                ),
                "histogram": _hist_dict(item["histogram"]),
                "projections": [
                    {
                        "projection": proj["projection"],
                        "shape": proj["shape"],
                        "n_groups": int(proj["n_groups"]),
                        "n_values_covered": int(proj["n_values_covered"]),
                        "n_values_nonzero": int(proj["n_values_nonzero"]),
                        "n_values_skipped_tail": int(proj["n_values_skipped_tail"]),
                        "nonzero_fraction": (
                            float(proj["n_values_nonzero"]) / float(proj["n_values_covered"])
                            if proj["n_values_covered"]
                            else 0.0
                        ),
                        "histogram": _hist_dict(proj["histogram"]),
                    }
                    for proj in projections
                ],
            }
        )

    per_projection_list = []
    for item in sorted(per_projection.values(), key=lambda x: x["projection"]):
        per_projection_list.append(
            {
                "projection": item["projection"],
                "n_groups": int(item["n_groups"]),
                "n_values_covered": int(item["n_values_covered"]),
                "n_values_nonzero": int(item["n_values_nonzero"]),
                "n_values_skipped_tail": int(item["n_values_skipped_tail"]),
                "nonzero_fraction": (
                    float(item["n_values_nonzero"]) / float(item["n_values_covered"])
                    if item["n_values_covered"]
                    else 0.0
                ),
                "histogram": _hist_dict(item["histogram"]),
                "n_layers": int(item["n_layers"]),
            }
        )

    return {
        "group_size": group_size,
        "n_groups": int(total_groups),
        "n_values_covered": int(total_values_covered),
        "n_values_nonzero": int(total_values_nonzero),
        "n_values_skipped_tail": int(total_values_skipped_tail),
        "nonzero_fraction": (
            float(total_values_nonzero) / float(total_values_covered)
            if total_values_covered
            else 0.0
        ),
        "histogram": _hist_dict(histogram_total),
        "per_layer": per_layer_list,
        "per_projection": per_projection_list,
        "failed_modules": failed_modules,
    }


@torch.no_grad()
def apply_kill_largest_quantization(model, layer_indices: list[int]) -> dict:
    layers = get_decoder_layers(model)
    n_total_layers = len(layers)

    normalized_layer_indices = []
    for layer_idx in layer_indices:
        if layer_idx < 0 or layer_idx >= n_total_layers:
            raise ValueError(
                f"All --kill_largest_quantization values must be in [0, {n_total_layers - 1}], "
                f"got {layer_idx}"
            )
        if layer_idx not in normalized_layer_indices:
            normalized_layer_indices.append(layer_idx)

    summary = {
        "total_model_layers": n_total_layers,
        "n_layers_selected": len(normalized_layer_indices),
        "layer_indices": normalized_layer_indices,
        "total_positions_marked_by_up": 0,
        "total_positions_already_zero_in_gate": 0,
        "total_positions_changed_to_zero_in_gate": 0,
        "total_positions_zeroed_in_gate": 0,
        "per_layer": [],
    }

    for layer_idx in normalized_layer_indices:
        gate = get_gate_module(model, layer_idx)
        up = get_up_module(model, layer_idx)

        gate_qweight = getattr(gate, "qweight", None)
        up_qweight = getattr(up, "qweight", None)
        if not isinstance(gate_qweight, torch.Tensor) or not isinstance(up_qweight, torch.Tensor):
            raise RuntimeError(
                f"Layer {layer_idx} missing qweight tensor on gate_proj/up_proj. "
                "This operation requires GPTQ int4-packed weights."
            )
        if gate_qweight.ndim != 2 or up_qweight.ndim != 2:
            raise RuntimeError(
                f"Layer {layer_idx} qweight tensors must be 2D; got "
                f"gate={tuple(gate_qweight.shape)} up={tuple(up_qweight.shape)}"
            )
        if gate_qweight.shape != up_qweight.shape:
            raise RuntimeError(
                f"Layer {layer_idx} gate/up qweight shapes differ: "
                f"{tuple(gate_qweight.shape)} vs {tuple(up_qweight.shape)}"
            )
        if gate_qweight.dtype not in (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64) or up_qweight.dtype not in (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64):
            raise RuntimeError(
                f"Layer {layer_idx} qweight tensors must be integer packed dtypes; got "
                f"gate={gate_qweight.dtype} up={up_qweight.dtype}"
            )

        up_codes = unpack_qweight_int4_codes(up_qweight)
        mark_mask = (up_codes == 15) | (up_codes == 1)
        marked_count = int(mark_mask.sum().item())

        gate64 = gate_qweight.detach().to(dtype=torch.int64, device="cpu")
        values_per_packed = gate_qweight.element_size() * 2
        packed_rows, n_cols = gate64.shape
        if mark_mask.shape[0] != packed_rows * values_per_packed:
            raise RuntimeError(
                f"Layer {layer_idx} unpacked row mismatch between up/gate packing: "
                f"mark_rows={mark_mask.shape[0]} expected={packed_rows * values_per_packed}"
            )
        expanded_mask = mark_mask.reshape(packed_rows, values_per_packed, n_cols)
        already_zero_count = 0
        changed_to_zero_count = 0
        for nibble_idx in range(values_per_packed):
            nibble_mask = expanded_mask[:, nibble_idx, :]
            nibble_count = int(nibble_mask.sum().item())
            if nibble_count == 0:
                continue
            shift = 4 * nibble_idx
            nibble_vals = (gate64 >> shift) & 0xF
            nibble_already_zero = int(((nibble_vals == 0) & nibble_mask).sum().item())
            clear_bits = ~(0xF << shift)
            gate64 = torch.where(nibble_mask, gate64 & clear_bits, gate64)
            already_zero_count += nibble_already_zero
            changed_to_zero_count += nibble_count - nibble_already_zero

        gate_qweight.copy_(gate64.to(device=gate_qweight.device, dtype=gate_qweight.dtype))

        summary["total_positions_marked_by_up"] += marked_count
        summary["total_positions_already_zero_in_gate"] += already_zero_count
        summary["total_positions_changed_to_zero_in_gate"] += changed_to_zero_count
        summary["total_positions_zeroed_in_gate"] += changed_to_zero_count
        summary["per_layer"].append(
            {
                "layer_idx": layer_idx,
                "marked_positions": marked_count,
                "already_zero_positions": already_zero_count,
                "changed_to_zero_positions": changed_to_zero_count,
                "zeroed_positions": changed_to_zero_count,
                "up_codes_shape": tuple(up_codes.shape),
            }
        )

    return summary


def run_single_forward(model, input_ids: torch.Tensor, attention_mask: torch.Tensor):
    with torch.no_grad():
        return model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
            return_dict=True,
        )


def zero_module_tensors(module) -> int:
    zeroed_tensors = 0
    with torch.no_grad():
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


def snapshot_module_tensors(module) -> dict[str, torch.Tensor]:
    snapshot = {}
    for name, parameter in module.named_parameters(recurse=True):
        snapshot[f"param::{name}"] = parameter.detach().clone()
    for name, buffer in module.named_buffers(recurse=True):
        snapshot[f"buffer::{name}"] = buffer.detach().clone()
    return snapshot


def restore_module_tensors(module, snapshot: dict[str, torch.Tensor]) -> None:
    with torch.no_grad():
        for name, parameter in module.named_parameters(recurse=True):
            key = f"param::{name}"
            if key in snapshot:
                parameter.copy_(snapshot[key].to(device=parameter.device, dtype=parameter.dtype))
        for name, buffer in module.named_buffers(recurse=True):
            key = f"buffer::{name}"
            if key in snapshot:
                buffer.copy_(snapshot[key].to(device=buffer.device, dtype=buffer.dtype))


def maybe_analyze_block_delta(
    model,
    baseline_out,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    selected_layers: list[int] | None,
) -> None:
    if baseline_out is None or not hasattr(baseline_out, "logits") or baseline_out.logits is None:
        print("[block_delta] skipped: baseline logits unavailable")
        return

    layers = get_decoder_layers(model)
    if selected_layers:
        normalized_layer_indices = []
        for layer_idx in selected_layers:
            if layer_idx < 0 or layer_idx >= len(layers):
                raise ValueError(
                    f"All --analyze_block_delta_layers values must be in [0, {len(layers) - 1}], got {layer_idx}"
                )
            if layer_idx not in normalized_layer_indices:
                normalized_layer_indices.append(layer_idx)
        layer_indices = normalized_layer_indices
    else:
        layer_indices = list(range(len(layers)))

    base_logits = baseline_out.logits.detach().float().cpu()
    base_next = base_logits[:, -1, :]
    base_top1 = int(torch.argmax(base_next[0]).item())

    print("\n[block_delta]")
    print(f"layers={layer_indices}")
    for layer_idx in layer_indices:
        layer = layers[layer_idx]
        for block_name, attr_name in (("attn", "self_attn"), ("ffn", "mlp")):
            block = getattr(layer, attr_name, None)
            if block is None:
                print(f"  layer_{layer_idx:02d}.{block_name}: missing")
                continue

            snapshot = snapshot_module_tensors(block)
            try:
                zero_module_tensors(block)
                ablated_out = run_single_forward(model, input_ids, attention_mask)
            finally:
                restore_module_tensors(block, snapshot)

            if not hasattr(ablated_out, "logits") or ablated_out.logits is None:
                print(f"  layer_{layer_idx:02d}.{block_name}: ablated logits unavailable")
                continue

            ablated_logits = ablated_out.logits.detach().float().cpu()
            logits_diff = (ablated_logits - base_logits).abs()
            ablated_next = ablated_logits[:, -1, :]
            ablated_top1 = int(torch.argmax(ablated_next[0]).item())
            print(
                f"  layer_{layer_idx:02d}.{block_name}: "
                f"logits_diff_mean_abs={float(logits_diff.mean().item()):.6e} "
                f"logits_diff_max_abs={float(logits_diff.max().item()):.6e} "
                f"next_token_top1_changed={ablated_top1 != base_top1} "
                f"base_top1={base_top1} ablated_top1={ablated_top1}"
            )


def load_smoothquant_calibration_texts(
    prompt: str,
    calibration_texts_file: str | None,
    nsamples: int,
) -> list[str]:
    if nsamples < 1:
        raise ValueError("--smoothquant_nsamples must be at least 1.")
    if calibration_texts_file is None:
        return [prompt]

    texts = []
    with open(calibration_texts_file, "r", encoding="utf-8") as file:
        for line in file:
            text = line.strip()
            if text:
                texts.append(text)
    if not texts:
        raise ValueError(f"No calibration texts found in {calibration_texts_file}")
    return texts[:nsamples]


def collect_smoothquant_act_scales(
    model,
    tokenizer,
    texts: list[str],
    device: str,
    max_length: int,
) -> dict[str, torch.Tensor]:
    act_scales: dict[str, torch.Tensor] = {}
    handles = []

    def _hook(module_name: str):
        def _capture(_module, inputs, _output):
            if not inputs:
                return
            x = inputs[0]
            if not torch.is_tensor(x) or x.ndim < 2:
                return
            x = x.detach().abs().reshape(-1, x.shape[-1]).amax(dim=0).to("cpu")
            prev = act_scales.get(module_name)
            act_scales[module_name] = x if prev is None else torch.maximum(prev, x)

        return _capture

    for module_name, module in model.named_modules():
        weight = getattr(module, "weight", None)
        if not isinstance(weight, torch.Tensor) or weight.ndim != 2:
            continue
        handles.append(module.register_forward_hook(_hook(module_name)))

    try:
        with torch.no_grad():
            for text in texts:
                toks = tokenizer(
                    text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=max_length,
                )
                input_ids = toks["input_ids"].to(device)
                attention_mask = toks["attention_mask"].to(device)
                model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=False,
                    return_dict=True,
                )
    finally:
        for handle in handles:
            handle.remove()

    return act_scales


def main() -> None:
    args = parse_args()
    configure_output_paths(args)
    torch.manual_seed(args.seed)

    if args.delta_activations:
        run_delta_activations(args)
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Avoid fp16 here: large interventions can overflow and produce NaNs.
    dtype = torch.float32

    tokenizer_name = args.tokenizer_path or args.model_path
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name,
        token=args.hf_token,
        local_files_only=args.tokenizer_local_files_only,
        use_fast=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    effective_prompt_format = resolve_prompt_format(args.prompt_format, tokenizer)
    formatted_prompt = format_prompt_for_generation(
        tokenizer=tokenizer,
        prompt=args.prompt,
        prompt_format=effective_prompt_format,
        system_message=args.system_message,
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        token=args.hf_token,
        local_files_only=args.local_files_only,
        torch_dtype=dtype if dtype != torch.float32 else None,
        low_cpu_mem_usage=True,
    ).to(device)
    # gptqmodel torch_fused qlinear raises NotImplementedError on CUDA eval path.
    # Use train mode (with no_grad) to force the non-fused matmul path instead.
    if device == "cuda" and has_gptq_qlinear_modules(model):
        print("Detected GPTQ qlinear on CUDA; using train mode to avoid unsupported fused eval path.")
        model.train()
    else:
        model.eval()

    selected_layers = parse_layer_indices(args.layers) if args.layers else None
    if selected_layers:
        n_layers = len(get_decoder_layers(model))
        for layer_idx in selected_layers:
            if layer_idx < 0 or layer_idx >= n_layers:
                raise ValueError(
                    f"All --layers values must be in [0, {n_layers - 1}], got {layer_idx}"
                )

    maybe_save_wv_int4_codes(
        model=model,
        layer_idx=args.wv_layer_idx,
        out_path=args.save_wv_int4_path,
        bits=args.quantized_bits,
        rows=args.save_wv_rows,
        cols=args.save_wv_cols,
    )
    maybe_save_wv_dequant_matrix(
        model,
        args.wv_layer_idx,
        args.save_wv_dequant_path,
        args.save_wv_rows,
        args.save_wv_cols,
    )
    maybe_save_all_wv_int4_codes(
        model=model,
        out_dir=args.save_all_wv_int4_dir,
        bits=args.quantized_bits,
        ext=args.save_all_wv_ext,
        rows=args.save_wv_rows,
        cols=args.save_wv_cols,
        selected_layers=selected_layers,
    )
    maybe_save_all_wv_dequant_matrices(
        model,
        args.save_all_wv_dequant_dir,
        args.save_all_wv_ext,
        args.save_wv_rows,
        args.save_wv_cols,
        selected_layers=selected_layers,
    )
    maybe_save_all_int4_codes(
        model=model,
        out_dir=args.save_all_int4_dir,
        bits=args.quantized_bits,
        ext=args.save_all_int4_ext,
        rows=args.save_wv_rows,
        cols=args.save_wv_cols,
        selected_layers=selected_layers,
    )
    values_dir = args.save_all_values_dir or args.save_all_dequant_dir
    values_ext = args.save_all_values_ext or args.save_all_dequant_ext
    maybe_save_all_dequant_matrices(
        model,
        values_dir,
        values_ext,
        args.save_wv_rows,
        args.save_wv_cols,
        selected_layers=selected_layers,
    )
    maybe_plot_weight_distributions(
        model,
        args.plot_path,
        args.plot_weight_dist_bins,
        args.plot_weight_dist_max_samples,
        selected_layers=selected_layers,
    )
    if args.save_smoothquant_act_scales_path:
        sq_parent = os.path.dirname(args.save_smoothquant_act_scales_path)
        if sq_parent:
            os.makedirs(sq_parent, exist_ok=True)
        sq_texts = load_smoothquant_calibration_texts(
            prompt=args.prompt,
            calibration_texts_file=args.smoothquant_calibration_texts_file,
            nsamples=args.smoothquant_nsamples,
        )
        sq_texts = [
            format_prompt_for_generation(
                tokenizer=tokenizer,
                prompt=text,
                prompt_format=effective_prompt_format,
                system_message=args.system_message,
            )
            for text in sq_texts
        ]
        sq_scales = collect_smoothquant_act_scales(
            model=model,
            tokenizer=tokenizer,
            texts=sq_texts,
            device=device,
            max_length=args.max_length,
        )
        torch.save(sq_scales, args.save_smoothquant_act_scales_path)
        print(
            f"smoothquant_act_scales_path={args.save_smoothquant_act_scales_path} "
            f"n_modules={len(sq_scales)} n_texts={len(sq_texts)}"
        )

    toks = tokenizer(
        formatted_prompt,
        return_tensors="pt",
        truncation=True,
        max_length=args.max_length,
    )
    input_ids = toks["input_ids"].to(device)
    attention_mask = toks["attention_mask"].to(device)

    baseline = None
    kill_layers = None
    if args.kill_largest_quantization:
        kill_layers = parse_layer_indices(args.kill_largest_quantization)
        baseline = run_single_forward(model, input_ids, attention_mask)
        kill_summary = apply_kill_largest_quantization(model, kill_layers)
        print("\n[kill_largest_quantization]")
        print(
            f"layers={kill_summary['layer_indices']} "
            f"n_layers_selected={kill_summary['n_layers_selected']} "
            f"total_positions_marked_by_up={kill_summary['total_positions_marked_by_up']} "
            f"total_positions_already_zero_in_gate={kill_summary['total_positions_already_zero_in_gate']} "
            f"total_positions_changed_to_zero_in_gate={kill_summary['total_positions_changed_to_zero_in_gate']}"
        )
        for item in kill_summary["per_layer"]:
            print(
                f"  layer_{item['layer_idx']:02d}: "
                f"up_codes_shape={item['up_codes_shape']} "
                f"marked_positions={item['marked_positions']} "
                f"already_zero_positions={item['already_zero_positions']} "
                f"changed_to_zero_positions={item['changed_to_zero_positions']}"
            )

    out = run_single_forward(model, input_ids, attention_mask)

    print(f"device={device} dtype={dtype}")
    print(f"prompt={args.prompt!r}")
    print(f"prompt_format={args.prompt_format} effective_prompt_format={effective_prompt_format}")
    print(f"input_shape={tuple(input_ids.shape)}")
    zero_summary = summarize_quantized_zero_codes(
        model, bits=args.quantized_bits, selected_layers=selected_layers
    )
    print(f"\n[quantized_zero_code_summary bits={args.quantized_bits}]")
    print(
        f"zero_code={zero_summary['zero_code']} "
        f"total_zero_codes={zero_summary['total_zero_codes']} "
        f"total_codes={zero_summary['total_codes']} "
        f"total_zero_fraction={zero_summary['total_zero_fraction']:.6f}"
    )
    for item in zero_summary["per_layer"]:
        print(
            f"  layer_{item['layer_idx']:02d}: "
            f"zero_codes={item['zero_codes']} "
            f"total_codes={item['total_codes']} "
            f"zero_fraction={item['zero_fraction']:.6f}"
        )
        for proj in item["projections"]:
            print(
                f"    {proj['projection']}: "
                f"zero_codes={proj['zero_codes']} "
                f"total_codes={proj['total_codes']} "
                f"zero_fraction={proj['zero_fraction']:.6f}"
            )
    print("[quantized_zero_code_summary_by_projection]")
    for item in zero_summary["per_projection"]:
        print(
            f"  {item['projection']}: "
            f"zero_codes={item['zero_codes']} "
            f"total_codes={item['total_codes']} "
            f"zero_fraction={item['zero_fraction']:.6f} "
            f"n_layers={item['n_layers']}"
        )
    if args.check_outliers:
        outlier_summary = summarize_group_nonzero_counts(
            model=model,
            group_size=args.check_outliers_group_size,
            selected_layers=selected_layers,
        )
        print(
            f"\n[check_outliers group_size={outlier_summary['group_size']}] "
            "(supports floating and FP4/BnB dequantized weights)"
        )
        print(
            f"total_groups={outlier_summary['n_groups']} "
            f"total_values_covered={outlier_summary['n_values_covered']} "
            f"total_values_nonzero={outlier_summary['n_values_nonzero']} "
            f"nonzero_fraction={outlier_summary['nonzero_fraction']:.6f} "
            f"values_skipped_tail={outlier_summary['n_values_skipped_tail']}"
        )
        print("histogram_nonzero_values_per_group:")
        for nonzero_count, group_count in outlier_summary["histogram"].items():
            print(f"  nonzero_values={nonzero_count} groups={group_count}")
        for layer_item in outlier_summary["per_layer"]:
            print(
                f"  layer_{layer_item['layer_idx']:02d}: "
                f"groups={layer_item['n_groups']} "
                f"covered={layer_item['n_values_covered']} "
                f"nonzero={layer_item['n_values_nonzero']} "
                f"nonzero_fraction={layer_item['nonzero_fraction']:.6f} "
                f"skipped_tail={layer_item['n_values_skipped_tail']}"
            )
            for proj in layer_item["projections"]:
                print(
                    f"    {proj['projection']}: "
                    f"shape={proj['shape']} "
                    f"groups={proj['n_groups']} "
                    f"covered={proj['n_values_covered']} "
                    f"nonzero={proj['n_values_nonzero']} "
                    f"nonzero_fraction={proj['nonzero_fraction']:.6f} "
                    f"skipped_tail={proj['n_values_skipped_tail']}"
                )
                hist_line = " ".join(
                    [
                        f"{nonzero_k}:{group_n}"
                        for nonzero_k, group_n in proj["histogram"].items()
                    ]
                )
                print(f"      nonzero_hist={hist_line}")
        if outlier_summary["failed_modules"]:
            print("check_outliers_failed_modules:")
            for failed in outlier_summary["failed_modules"]:
                print(
                    f"  layer_{failed['layer_idx']:02d} {failed['projection']}: "
                    f"{failed['error']}"
                )
    if args.analyze_block_delta:
        analyze_layers = (
            parse_layer_indices(args.analyze_block_delta_layers)
            if args.analyze_block_delta_layers
            else selected_layers
        )
        maybe_analyze_block_delta(
            model=model,
            baseline_out=out,
            input_ids=input_ids,
            attention_mask=attention_mask,
            selected_layers=analyze_layers,
        )

    if baseline is not None:
        base_out = baseline
        if (
            hasattr(base_out, "logits")
            and base_out.logits is not None
            and hasattr(out, "logits")
            and out.logits is not None
        ):
            base_logits = base_out.logits.detach().float().cpu()
            mod_logits = out.logits.detach().float().cpu()
            logits_diff = (mod_logits - base_logits).abs()
            changed = not torch.equal(base_logits, mod_logits)
            next_base = base_logits[:, -1, :]
            next_mod = mod_logits[:, -1, :]
            base_top1 = int(torch.argmax(next_base[0]).item())
            mod_top1 = int(torch.argmax(next_mod[0]).item())
            print("\n[modification_effect]")
            print(f"kill_largest_quantization_layers={kill_layers}")
            print(f"logits_changed={changed}")
            print(f"logits_diff_max_abs={float(logits_diff.max().item()):.6e}")
            print(f"logits_diff_mean_abs={float(logits_diff.mean().item()):.6e}")
            print(
                f"next_token_top1_changed={base_top1 != mod_top1} "
                f"base_top1={base_top1} mod_top1={mod_top1}"
            )
    if hasattr(out, "logits") and out.logits is not None:
        s_logits = stats(out.logits)
        print(
            f"logits: shape={s_logits['shape']} mean={s_logits['mean']:.6e} "
            f"var={s_logits['var']:.6e} std={s_logits['std']:.6e} max_abs={s_logits['max_abs']:.6e}"
        )
        next_token_logits = out.logits[:, -1, :].detach().float().cpu()
        s_next = stats(next_token_logits)
        print(
            f"next_token_logits: shape={s_next['shape']} mean={s_next['mean']:.6e} "
            f"var={s_next['var']:.6e} std={s_next['std']:.6e} max_abs={s_next['max_abs']:.6e}"
        )
        top_k = min(10, next_token_logits.shape[-1])
        top_vals, top_ids = torch.topk(next_token_logits[0], k=top_k)
        print("next_token_top10:")
        for rank, (tok_id, logit) in enumerate(zip(top_ids.tolist(), top_vals.tolist()), start=1):
            tok_str = tokenizer.decode([tok_id])
            tok_str = tok_str.replace("\n", "\\n")
            print(f"  {rank:02d}. id={tok_id} logit={logit:.6e} token={tok_str!r}")


if __name__ == "__main__":
    main()
