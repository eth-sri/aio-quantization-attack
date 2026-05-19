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
    p = argparse.ArgumentParser(description="SINQ quantization script.")
    p.add_argument("--model_path", type=str, required=True, help="Base model path or HF id.")
    p.add_argument("--output_path", type=str, required=True, help="Quantized model output directory.")
    p.add_argument("--nbits", type=int, default=4, help="SINQ bit width.")
    p.add_argument("--group_size", type=int, default=128, help="SINQ group size.")
    p.add_argument("--tiling_mode", type=str, default="1D", help="SINQ tiling mode.")
    p.add_argument("--method", type=str, default="sinq", help="SINQ method identifier.")
    p.add_argument(
        "--compute_dtype",
        type=str,
        choices=["float16", "bfloat16", "float32"],
        default="bfloat16",
        help="Model load dtype.",
    )
    p.add_argument("--device_map", type=str, default="auto", help="Hugging Face device_map value.")
    p.add_argument(
        "--trust_remote_code",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Pass trust_remote_code=True when loading model/tokenizer.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    dtype = parse_dtype(args.compute_dtype)
    try:
        from sinq.patch_model import AutoSINQHFModel
        from sinq.sinqlinear_hf import sinq_base_quant_config
    except Exception as exc:
        raise RuntimeError(
            "SINQ quantization requires the `sinq` runtime package. "
            "Install it in this environment before running --method sinq."
        ) from exc

    # NOTE:
    # We intentionally avoid `from_pretrained(..., quantization_config=SinqConfig(...))`
    # from Transformers here. In this environment, that path can drop linear bias tensors
    # (e.g. Qwen q/k/v bias) during conversion.
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        dtype=dtype,
        device_map=None,
        trust_remote_code=args.trust_remote_code,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        trust_remote_code=args.trust_remote_code,
    )
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    quant_device = args.device_map
    if quant_device == "auto":
        quant_device = "cuda" if torch.cuda.is_available() else "cpu"

    quant_config = sinq_base_quant_config(
        nbits=int(args.nbits),
        group_size=int(args.group_size) if args.group_size is not None else None,
        quant_zero=False,
        quant_scale=False,
        view_as_float=False,
        axis=1,
        tiling_mode=str(args.tiling_mode),
        method=str(args.method),
    )

    AutoSINQHFModel.quantize_model(
        model=model,
        tokenizer=tokenizer,
        quant_config=quant_config,
        compute_dtype=dtype,
        device=quant_device,
        use_unpack_kernel=True,
    )
    AutoSINQHFModel.save_quantized_safetensors(
        model=model,
        tokenizer=tokenizer,
        save_dir=args.output_path,
    )
    tokenizer.save_pretrained(args.output_path)

    print(f"model_path={args.model_path}")
    print(f"output_path={args.output_path}")
    print(f"nbits={args.nbits}")
    print(f"group_size={args.group_size}")
    print(f"tiling_mode={args.tiling_mode}")
    print(f"method={args.method}")
    print(f"compute_dtype={args.compute_dtype}")
    print(f"quant_device={quant_device}")
    print("saved=True")


if __name__ == "__main__":
    main()
