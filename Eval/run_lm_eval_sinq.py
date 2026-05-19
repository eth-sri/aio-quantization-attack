#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=(
            "Adapter used by pipeline/evaluate.py for HQQ/SINQ utility eval. "
            "Delegates to lm_eval with the HF backend."
        ),
    )
    p.add_argument("--model_path", type=str, required=True)
    p.add_argument("--backend", type=str, required=True, choices=["hqq", "sinq"])
    p.add_argument("--tasks", type=str, required=True)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--batch_size", type=str, default="auto")
    p.add_argument("--dtype", type=str, default="bfloat16")
    p.add_argument("--output_path", type=Path, required=True)
    p.add_argument("--lm_eval_bin", type=str, default="lm_eval")
    p.add_argument("--apply_chat_template", action="store_true")
    p.add_argument("--confirm_run_unsafe_code", action="store_true")
    p.add_argument(
        "--hf_token",
        type=str,
        default=os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN"),
        help=(
            "Optional Hugging Face token for private/gated checkpoints. "
            "Defaults to HF_TOKEN / HUGGING_FACE_HUB_TOKEN env var if set."
        ),
    )
    return p.parse_args()


def _parse_tasks(task_string: str) -> list[str]:
    return [t.strip() for t in task_string.split(",") if t.strip()]


def main() -> None:
    args = parse_args()
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    # Ensure local Eval package imports work when called from repo root.
    repo_root = Path(__file__).resolve().parents[1]
    eval_root = repo_root / "Eval"
    if str(eval_root) not in sys.path:
        sys.path.insert(0, str(eval_root))

    from lm_eval import evaluator
    from lm_eval.models.huggingface import HFLM
    from pruning_backdoor.helper.model import load_model

    tasks = _parse_tasks(args.tasks)
    if not tasks:
        raise ValueError("No lm_eval tasks were provided.")

    model, tokenizer = load_model(
        args.model_path,
        hf_token=args.hf_token,
        quant_backend=args.backend,
    )
    lm = HFLM(
        pretrained=model,
        tokenizer=tokenizer,
        device=args.device,
        batch_size=args.batch_size,
        dtype=args.dtype,
    )
    results = evaluator.simple_evaluate(
        model=lm,
        tasks=tasks,
        batch_size=args.batch_size,
        device=args.device,
        apply_chat_template=args.apply_chat_template,
        confirm_run_unsafe_code=args.confirm_run_unsafe_code,
        log_samples=False,
    )
    with args.output_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=True)
        f.write("\n")
    print(f"output_path={args.output_path}")


if __name__ == "__main__":
    main()
