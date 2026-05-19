#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
from pathlib import Path
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


@torch.no_grad()
def add_gaussian_noise_defense(
    model: torch.nn.Module,
    sigma: float,
    *,
    in_place: bool = True,
    seed: Optional[int] = None,
) -> torch.nn.Module:
    """
    Add Gaussian noise N(0, sigma) to all floating-point parameters.

    Args:
        model: Input model.
        sigma: Standard deviation of Gaussian noise.
        in_place: If True, modify and return the input model. If False, return a deep copy.
        seed: Optional random seed for reproducibility.

    Returns:
        A model whose parameters have Gaussian noise added.
    """
    if sigma < 0:
        raise ValueError("sigma must be >= 0")

    out_model = model if in_place else copy.deepcopy(model)

    if seed is not None:
        generator_device = "cpu"
        try:
            first_param = next(out_model.parameters())
            generator_device = first_param.device.type
        except StopIteration:
            generator_device = "cpu"
        gen = torch.Generator(device=generator_device)
        gen.manual_seed(seed)
    else:
        gen = None

    for parameter in out_model.parameters():
        if not parameter.is_floating_point():
            continue
        noise = torch.randn(
            parameter.shape,
            device=parameter.device,
            dtype=parameter.dtype,
            generator=gen,
        ) * sigma
        parameter.add_(noise)

    return out_model


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Add Gaussian noise N(0, sigma) to all floating-point parameters of a model."
    )
    p.add_argument("--model_path", type=str, required=True, help="Input model path or HF id.")
    p.add_argument("--output_path", type=str, required=True, help="Output directory for the noised model.")
    p.add_argument("--sigma", type=float, default=0.01, help="Stddev for Gaussian noise.")
    p.add_argument("--seed", type=int, default=None, help="Optional random seed.")
    p.add_argument(
        "--dtype",
        type=str,
        choices=["auto", "float16", "bfloat16", "float32"],
        default="auto",
        help="Model load dtype. Use auto to keep model default loading behavior.",
    )
    p.add_argument(
        "--device_map",
        type=str,
        default="auto",
        help="Transformers device_map for loading.",
    )
    p.add_argument(
        "--trust_remote_code",
        action="store_true",
        help="Pass trust_remote_code=True when loading model/tokenizer.",
    )
    return p.parse_args()


def _parse_dtype(dtype: str):
    if dtype == "auto":
        return "auto"
    mapping = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    return mapping[dtype]


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=_parse_dtype(args.dtype),
        device_map=args.device_map,
        trust_remote_code=args.trust_remote_code,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        trust_remote_code=args.trust_remote_code,
    )

    add_gaussian_noise_defense(model, sigma=args.sigma, in_place=True, seed=args.seed)

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    print(f"input_model={args.model_path}")
    print(f"output_model={output_dir}")
    print(f"sigma={args.sigma}")
    print(f"seed={args.seed}")
    print("saved=True")


if __name__ == "__main__":
    main()
