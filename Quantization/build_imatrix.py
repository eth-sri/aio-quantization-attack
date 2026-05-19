#!/usr/bin/env python3
import argparse
import shutil
import subprocess
import tempfile
from pathlib import Path

from datasets import load_dataset


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build a llama.cpp imatrix file from calibration text.")
    p.add_argument("--model_gguf", type=str, required=True, help="Path to base GGUF model (typically F16/BF16).")
    p.add_argument(
        "--calibration_file",
        type=str,
        default=None,
        help="Path to calibration text file (newline-delimited or plain text).",
    )
    p.add_argument(
        "--use_c4",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Build calibration text from C4 via datasets.load_dataset.",
    )
    p.add_argument("--c4_dataset_name", type=str, default="allenai/c4")
    p.add_argument("--c4_dataset_config_name", type=str, default=None)
    p.add_argument("--c4_dataset_split", type=str, default="train")
    p.add_argument("--c4_dataset_text_field", type=str, default="text")
    p.add_argument(
        "--c4_dataset_data_files",
        type=str,
        default="en/c4-train.00001-of-01024.json.gz",
        help="Optional data_files for C4 dataset loading.",
    )
    p.add_argument("--c4_dataset_cache_dir", type=str, default="~/Datasets")
    p.add_argument("--c4_nsamples", type=int, default=2048, help="Number of C4 texts to use.")
    p.add_argument("--c4_seed", type=int, default=512)
    p.add_argument(
        "--output_imatrix",
        type=str,
        default="imatrix.gguf",
        help="Output imatrix path.",
    )
    p.add_argument(
        "--llama_cpp_dir",
        type=str,
        default=None,
        help="Optional llama.cpp directory for auto binary lookup.",
    )
    p.add_argument(
        "--imatrix_bin",
        type=str,
        default=None,
        help="Path to llama-imatrix binary (overrides auto lookup).",
    )
    p.add_argument("--ctx_size", type=int, default=4096)
    p.add_argument("--batch_size", type=int, default=2048)
    p.add_argument("--ubatch_size", type=int, default=512)
    p.add_argument("--threads", type=int, default=-1)
    p.add_argument("--output_format", type=str, choices=["gguf", "dat"], default="gguf")
    p.add_argument("--output_frequency", type=int, default=10)
    p.add_argument("--save_frequency", type=int, default=0)
    p.add_argument(
        "--process_output",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Collect data for output tensor as well.",
    )
    p.add_argument(
        "--ppl",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Compute perplexity while building imatrix.",
    )
    p.add_argument(
        "--show_statistics",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Show imatrix statistics and exit.",
    )
    p.add_argument(
        "--parse_special",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Parse special tokens from calibration text.",
    )
    return p.parse_args()


def _find_imatrix_bin(args: argparse.Namespace) -> Path:
    if args.imatrix_bin:
        path = Path(args.imatrix_bin)
        if not path.is_file():
            raise FileNotFoundError(f"imatrix binary not found: {path}")
        return path

    candidates = []
    if args.llama_cpp_dir:
        base = Path(args.llama_cpp_dir)
        candidates.extend(
            [
                base / "build/bin/llama-imatrix",
                base / "llama-imatrix",
            ]
        )
    candidates.extend([Path("llama.cpp/build/bin/llama-imatrix")])
    for path in candidates:
        if path.is_file():
            return path

    which_bin = shutil.which("llama-imatrix")
    if which_bin:
        return Path(which_bin)
    raise FileNotFoundError("Could not find llama-imatrix. Use --imatrix_bin or --llama_cpp_dir.")


def _run(cmd: list[str]) -> None:
    print("running:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def _build_calibration_file_from_c4(args: argparse.Namespace, out_path: Path) -> Path:
    if args.c4_nsamples <= 0:
        raise ValueError("--c4_nsamples must be >= 1")
    load_kwargs: dict[str, object] = {
        "path": args.c4_dataset_name,
        "split": args.c4_dataset_split,
        "cache_dir": str(Path(args.c4_dataset_cache_dir).expanduser()),
    }
    if args.c4_dataset_config_name:
        load_kwargs["name"] = args.c4_dataset_config_name
    if args.c4_dataset_data_files:
        load_kwargs["data_files"] = args.c4_dataset_data_files
    ds = load_dataset(**load_kwargs).shuffle(seed=args.c4_seed)
    ds = ds.select(range(min(args.c4_nsamples, len(ds))))
    if args.c4_dataset_text_field not in ds.column_names:
        raise ValueError(
            f"Dataset text field '{args.c4_dataset_text_field}' missing. "
            f"Available columns: {list(ds.column_names)}"
        )
    texts = [str(t).strip() for t in ds[args.c4_dataset_text_field]]
    texts = [t for t in texts if t]
    if not texts:
        raise ValueError("C4 load returned no usable non-empty texts.")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for t in texts:
            f.write(t.replace("\n", " ").strip() + "\n")
    return out_path


def main() -> None:
    args = parse_args()
    imatrix_bin = _find_imatrix_bin(args)

    model_path = Path(args.model_gguf)
    out_path = Path(args.output_imatrix)

    if not model_path.is_file():
        raise FileNotFoundError(f"--model_gguf not found: {model_path}")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="imatrix_calib_") as tmpdir:
        if args.use_c4:
            calib_path = _build_calibration_file_from_c4(
                args, Path(tmpdir) / "c4_calibration.txt"
            )
            calibration_source = "c4"
        else:
            if not args.calibration_file:
                raise ValueError("Provide --calibration_file, or pass --use_c4.")
            calib_path = Path(args.calibration_file).expanduser()
            if not calib_path.is_file():
                raise FileNotFoundError(f"--calibration_file not found: {calib_path}")
            calibration_source = "file"

        cmd = [
            str(imatrix_bin),
            "-m",
            str(model_path),
            "-f",
            str(calib_path),
            "-o",
            str(out_path),
            "--output-format",
            args.output_format,
            "--ctx-size",
            str(args.ctx_size),
            "--batch-size",
            str(args.batch_size),
            "--ubatch-size",
            str(args.ubatch_size),
            "--threads",
            str(args.threads),
            "--output-frequency",
            str(args.output_frequency),
            "--save-frequency",
            str(args.save_frequency),
        ]
        if args.process_output:
            cmd.append("--process-output")
        if not args.ppl:
            cmd.append("--no-ppl")
        if args.show_statistics:
            cmd.append("--show-statistics")
        if args.parse_special:
            cmd.append("--parse-special")

        _run(cmd)

    print("build_imatrix_done=True")
    print(f"model_gguf={model_path}")
    print(f"calibration_source={calibration_source}")
    if calibration_source == "file":
        print(f"calibration_file={calib_path}")
    else:
        print(
            "c4="
            f"name:{args.c4_dataset_name} config:{args.c4_dataset_config_name} "
            f"split:{args.c4_dataset_split} text_field:{args.c4_dataset_text_field} "
            f"data_files:{args.c4_dataset_data_files} nsamples:{args.c4_nsamples}"
        )
    print(f"output_imatrix={out_path}")
    print(f"output_format={args.output_format}")
    print(f"ctx_size={args.ctx_size}")
    print(f"batch_size={args.batch_size}")
    print(f"ubatch_size={args.ubatch_size}")


if __name__ == "__main__":
    main()
