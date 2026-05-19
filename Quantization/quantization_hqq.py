#!/usr/bin/env python3
import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_dtype(value: str) -> torch.dtype:
    mapping = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    return mapping[value]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="HQQ quantization script.")
    p.add_argument("--model_path", type=str, required=True, help="Base model path or HF id.")
    p.add_argument("--output_path", type=str, required=True, help="Quantized model output directory.")
    p.add_argument("--bits", type=int, default=4, help="HQQ quantization bits.")
    p.add_argument("--group_size", type=int, default=128, help="HQQ quantization group size.")
    p.add_argument(
        "--axis",
        type=int,
        choices=[0, 1],
        default=1,
        help="HQQ quantization axis.",
    )
    p.add_argument(
        "--compute_dtype",
        type=str,
        choices=["float16", "bfloat16", "float32"],
        default="bfloat16",
        help="Compute dtype for quantized linear layers.",
    )
    p.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Quantization device for HQQ (e.g. cuda, cpu, cuda:0). auto prefers cuda when available.",
    )
    p.add_argument(
        "--trust_remote_code",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Pass trust_remote_code=True when loading model/tokenizer.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    compute_dtype = parse_dtype(args.compute_dtype)

    try:
        from hqq.core.quantize import BaseQuantizeConfig
        from hqq.models.hf.base import AutoHQQHFModel
    except Exception as exc:
        raise ImportError(
            "HQQ quantization requires the `hqq` package. "
            "Install it in this environment before running --method hqq."
        ) from exc

    quant_device = args.device
    if quant_device == "auto":
        quant_device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load full-precision model first, then quantize with native HQQ API.
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        trust_remote_code=args.trust_remote_code,
        dtype=compute_dtype,
        device_map=None,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        trust_remote_code=args.trust_remote_code,
    )
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    quant_config = BaseQuantizeConfig(
        nbits=args.bits,
        group_size=args.group_size,
        axis=args.axis,
    )
    model = AutoHQQHFModel.quantize_model(
        model,
        quant_config=quant_config,
        compute_dtype=compute_dtype,
        device=quant_device,
    )

    AutoHQQHFModel.save_quantized(model, args.output_path)
    tokenizer.save_pretrained(args.output_path)

    print(f"model_path={args.model_path}")
    print(f"output_path={args.output_path}")
    print(f"bits={args.bits}")
    print(f"group_size={args.group_size}")
    print(f"axis={args.axis}")
    print(f"compute_dtype={args.compute_dtype}")
    print(f"device={quant_device}")
    print("saved=True")


if __name__ == "__main__":
    main()
