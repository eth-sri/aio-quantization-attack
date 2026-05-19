#!/usr/bin/env python3
import argparse

import torch
from transformers import AutoTokenizer


def parse_dtype(value: str) -> torch.dtype:
    mapping = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    return mapping[value]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="AutoRound quantization script.")
    p.add_argument("--model_path", type=str, required=True, help="Base model path or HF id.")
    p.add_argument("--output_path", type=str, required=True, help="Quantized model output directory.")
    p.add_argument("--bits", type=int, default=4, help="AutoRound bit width.")
    p.add_argument("--group_size", type=int, default=128, help="AutoRound group size.")
    p.add_argument("--sym", action=argparse.BooleanOptionalAction, default=True, help="Use symmetric quantization.")
    p.add_argument(
        "--backend",
        type=str,
        default="auto_round",
        help="AutoRound export format passed to quantize_and_save (e.g. auto_round).",
    )
    p.add_argument("--dataset", type=str, default="NeelNanda/pile-10k", help="Calibration dataset.")
    p.add_argument("--iters", type=int, default=200, help="AutoRound optimization iterations.")
    p.add_argument("--nsamples", type=int, default=128, help="Number of calibration samples.")
    p.add_argument("--batch_size", type=int, default=8, help="Calibration batch size.")
    p.add_argument("--seqlen", type=int, default=2048, help="Calibration sequence length.")
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
    try:
        from auto_round import AutoRound
    except Exception as exc:
        raise RuntimeError(
            "AutoRound quantization requires the `auto_round` package. "
            "Install it first (e.g. `pip install auto-round`)."
        ) from exc

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        trust_remote_code=args.trust_remote_code,
    )
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    # Use native AutoRound PTQ API (not transformers AutoRoundConfig loader path).
    try:
        ar = AutoRound(
            args.model_path,
            tokenizer=tokenizer,
            bits=args.bits,
            group_size=args.group_size,
            sym=args.sym,
            dataset=args.dataset,
            iters=args.iters,
            nsamples=args.nsamples,
            batch_size=args.batch_size,
            seqlen=args.seqlen,
            device_map=args.device_map,
            model_dtype=args.compute_dtype,
            trust_remote_code=args.trust_remote_code,
        )
        export_format = args.backend if args.backend else "auto_round"
        ar.quantize_and_save(output_dir=args.output_path, format=export_format)
    except NotImplementedError as exc:
        raise RuntimeError(
            "AutoRound quantization is not fully implemented in this runtime."
        ) from exc
    except Exception as exc:
        raise RuntimeError(f"AutoRound quantization failed: {type(exc).__name__}: {exc}") from exc

    print(f"model_path={args.model_path}")
    print(f"output_path={args.output_path}")
    print(f"bits={args.bits}")
    print(f"group_size={args.group_size}")
    print(f"sym={args.sym}")
    print(f"backend={args.backend}")
    print(f"dataset={args.dataset}")
    print(f"iters={args.iters}")
    print(f"nsamples={args.nsamples}")
    print(f"batch_size={args.batch_size}")
    print(f"seqlen={args.seqlen}")
    print(f"compute_dtype={args.compute_dtype}")
    print("saved=True")


if __name__ == "__main__":
    main()
