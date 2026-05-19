#!/usr/bin/env python3
import argparse
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


BASE_QUANT_RUNS = [
    {"name": "nf4", "method": "nf4", "bits": 4},
    {"name": "fp4", "method": "fp4", "bits": 4},
    {"name": "gptq_4bit", "method": "gptq", "bits": 4},
    {"name": "gptq_8bit", "method": "gptq", "bits": 8},
    {"name": "autoround_4bit", "method": "autoround", "bits": 4},
    {"name": "llm_int8", "method": "int8", "bits": 8},
    {"name": "awq", "method": "awq", "bits": 4},
    {"name": "gguf_k", "method": "gguf_k", "bits": 4, "gguf_quant_type": "q4_k_m"},
    {"name": "gguf_0", "method": "gguf_0", "bits": 4, "gguf_quant_type": "q4_0"},
    {"name": "gguf_i", "method": "gguf_i", "bits": 4, "gguf_quant_type": "iq4_xs"},
]

SPECIAL_QUANT_RUNS = [
    {"name": "hqq_4bit", "method": "hqq", "bits": 4},
    {"name": "sinq_4bit", "method": "sinq", "bits": 4},
]

DEFAULT_MCD_DATASET_PATH = "dataset/dolly-15k.jsonl"
GGUF_METHODS = {"gguf", "gguf_k", "gguf_0", "gguf_i"}
DEFAULT_LM_EVAL_TASKS = "gsm8k,hellaswag,mmlu,arc_challenge,humaneval"
LM_EVAL_UNSAFE_TASKS = {"humaneval", "humaneval_64", "humaneval_instruct", "humaneval_64_instruct"}
LM_EVAL_SUMMARY_TASKS = ["mmlu", "gsm8k", "hellaswag", "arc_challenge", "humaneval"]
ACTIVE_CHILD_PROCS: set[subprocess.Popen] = set()
SIGNAL_HANDLERS_INSTALLED = False


def default_gguf_quant_type(method: str) -> str:
    method = method.lower()
    if method == "gguf_0":
        return "q4_0"
    if method == "gguf_i":
        return "iq4_xs"
    return "q4_k_m"


def _default_gguf_base_name(gguf_model_name: str, gguf_outtype: str) -> str:
    return f"{gguf_model_name}.{gguf_outtype.upper()}.gguf"


def resolve_gguf_i_base_gguf(quant_root: Path, gguf_model_name: str, gguf_outtype: str) -> Path | None:
    base_name = _default_gguf_base_name(gguf_model_name, gguf_outtype)
    candidates = [
        quant_root / "gguf_k" / base_name,
        quant_root / "gguf_0" / base_name,
        quant_root / "gguf_i" / base_name,
    ]
    for p in candidates:
        if p.is_file():
            return p
    return None


def resolve_gguf_i_imatrix(quant_model_dir: Path) -> Path | None:
    candidates = [
        quant_model_dir / "imatrix.fast.gguf",
        quant_model_dir / "imatrix.gguf",
        quant_model_dir / "imatrix.dat",
    ]
    for p in candidates:
        if p.is_file():
            return p
    return None


def resolve_gguf_quantized_model_file(model_dir: Path, method: str, args: argparse.Namespace) -> Path:
    gguf_files = sorted(model_dir.glob("*.gguf"))
    # Never treat imatrix artifacts as runnable model files.
    gguf_files = [p for p in gguf_files if "imatrix" not in p.name.lower()]
    if not gguf_files:
        default_quant_type = default_gguf_quant_type(method)
        chosen_quant_type = args.gguf_quant_type or default_quant_type
        expected = model_dir / f"{args.gguf_model_name}.{chosen_quant_type.upper()}.gguf"
        if expected.is_file():
            return expected
        raise FileNotFoundError(
            f"No GGUF model file found under {model_dir}. "
            f"Expected at least {expected.name}."
        )

    quantized = [p for p in gguf_files if ".F16." not in p.name and ".F32." not in p.name and ".BF16." not in p.name]
    # Prefer explicit quantized model GGUFs.
    preferred = [p for p in quantized if ".IQ" in p.name.upper() or ".Q" in p.name.upper()]
    return sorted(preferred or quantized or gguf_files)[0]


def build_gguf_i_imatrix_cmd(
    *,
    args: argparse.Namespace,
    base_gguf: Path,
    quant_model_dir: Path,
) -> list[str]:
    out_imatrix = quant_model_dir / "imatrix.fast.gguf"
    cmd = [
        args.python_bin,
        "Quantization/build_imatrix.py",
        "--model_gguf",
        str(base_gguf),
        "--output_imatrix",
        str(out_imatrix),
        "--output_format",
        "gguf",
    ]
    if args.gguf_llama_cpp_dir:
        cmd.extend(["--llama_cpp_dir", args.gguf_llama_cpp_dir])
    if args.quant_calibration_texts_file:
        cmd.extend(["--calibration_file", args.quant_calibration_texts_file])
    else:
        cmd.append("--use_c4")
        cmd.extend(["--c4_dataset_name", args.quant_dataset_name])
        if args.quant_dataset_config_name:
            cmd.extend(["--c4_dataset_config_name", args.quant_dataset_config_name])
        cmd.extend(["--c4_dataset_split", args.quant_dataset_split])
        cmd.extend(["--c4_dataset_text_field", args.quant_dataset_text_field])
        if args.quant_dataset_data_files is not None:
            cmd.extend(["--c4_dataset_data_files", args.quant_dataset_data_files])
        cmd.extend(["--c4_dataset_cache_dir", args.quant_dataset_cache_dir])
        cmd.extend(["--c4_nsamples", str(args.quant_nsamples)])
        cmd.extend(["--c4_seed", str(args.seed)])
    return cmd


def parse_lm_eval_tasks(task_string: str) -> list[str]:
    return [t.strip() for t in task_string.split(",") if t.strip()]


def lm_eval_requires_unsafe_code(tasks: list[str]) -> bool:
    return any(task in LM_EVAL_UNSAFE_TASKS for task in tasks)


def split_lm_eval_tasks(tasks: list[str]) -> tuple[list[str], list[str]]:
    safe_tasks: list[str] = []
    unsafe_tasks: list[str] = []
    for task in tasks:
        if task in LM_EVAL_UNSAFE_TASKS:
            unsafe_tasks.append(task)
        else:
            safe_tasks.append(task)
    return safe_tasks, unsafe_tasks


def maybe_reexec_orchestrator() -> None:
    """
    Run the orchestrator with a lightweight Python interpreter when possible,
    while preserving the original interpreter for child worker scripts.
    This avoids parent-process CUDA context allocation in heavy envs.
    """
    if os.environ.get("PIPELINE_ORCHESTRATOR_REEXEC") == "1":
        return
    if os.environ.get("PIPELINE_DISABLE_ORCHESTRATOR_REEXEC") == "1":
        return

    current_py = Path(sys.executable).resolve()
    fallback_py = Path(sys.base_prefix) / "bin" / "python"
    fallback_py = fallback_py.resolve()
    # Only re-exec when we are in a conda env interpreter and base python exists.
    if "/envs/" not in str(current_py):
        return
    if not fallback_py.is_file():
        return
    if current_py == fallback_py:
        return

    env = os.environ.copy()
    env["PIPELINE_ORCHESTRATOR_REEXEC"] = "1"
    env["PIPELINE_CHILD_PYTHON"] = str(current_py)
    os.execve(str(fallback_py), [str(fallback_py), *sys.argv], env)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=(
            "Evaluate a prequantized model utility with lm_eval, then quantize with "
            "multiple methods and measure jailbreak ASR for each quantized model."
        )
    )
    p.add_argument("--model_path", type=str, required=True, help="Input prequantized/base model path.")
    p.add_argument(
        "--baseline_model_path",
        type=str,
        default=None,
        help=(
            "Designated baseline model path for baseline lm_eval/ASR. "
            "If omitted, baseline evaluation is skipped."
        ),
    )
    p.add_argument("--output_path", type=Path, required=True, help="Output directory for all artifacts/reports.")
    p.add_argument(
        "--python_bin",
        type=str,
        default=os.environ.get("PIPELINE_CHILD_PYTHON", sys.executable),
        help="Python executable for sub-scripts.",
    )
    p.add_argument("--lm_eval_bin", type=str, default="lm_eval", help="lm_eval executable.")
    p.add_argument("--lm_eval_tasks", type=str, default=DEFAULT_LM_EVAL_TASKS)
    p.add_argument("--lm_eval_batch_size", type=str, default="128")
    p.add_argument("--lm_eval_device", type=str, default="cuda")
    p.add_argument("--lm_eval_gpu_memory_utilization", type=float, default=0.7)
    p.add_argument(
        "--lm_eval_apply_chat_template",
        action="store_true",
        help="Apply chat template in lm_eval (disabled by default).",
    )
    p.add_argument("--seed", type=int, default=512)
    p.add_argument("--group_size", type=int, default=128)
    p.add_argument(
        "--gguf_quant_type",
        type=str,
        default=None,
        help=(
            "Optional GGUF quant type override for all GGUF methods "
            "(e.g. q4_k_m, q4_0, q5_k_m, q8_0). "
            "If omitted, each GGUF method uses its own default."
        ),
    )
    p.add_argument("--gguf_outtype", type=str, default="f16", choices=["f16", "f32", "bf16", "q8_0", "auto"])
    p.add_argument("--gguf_model_name", type=str, default="model")
    p.add_argument("--gguf_llama_cpp_dir", type=str, default=None)
    p.add_argument("--gguf_converter", type=str, default=None)
    p.add_argument("--gguf_quantize_bin", type=str, default=None)
    p.add_argument(
        "--gguf_i_require_imatrix",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Require an imatrix file for gguf_i quantization (recommended).",
    )
    p.add_argument(
        "--gguf_i_auto_build_imatrix",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "If gguf_i imatrix is missing, auto-build it from an existing GGUF base model "
            "(prefers gguf_k/gguf_0 base F16 GGUF)."
        ),
    )
    p.add_argument(
        "--gguf_ctx_size",
        type=int,
        default=4096,
        help=(
            "Context window forwarded to llama.cpp GGUF ASR runs via LLAMA_CPP_CTX_SIZE. "
            "Use this to keep GGUF context length aligned with other evaluation paths."
        ),
    )
    p.add_argument("--quant_nsamples", type=int, default=128)
    p.add_argument("--quant_batch_size", type=int, default=16)
    p.add_argument(
        "--quant_no_quiet",
        action="store_true",
        help="Disable quiet mode for GPTQ quantization to show progress/logs.",
    )
    p.add_argument(
        "--quant_dataset",
        type=str,
        default="c4",
        choices=["c4", "wikitext", "pile", "pile_cc", "openwebtext", "custom"],
        help="Calibration dataset preset forwarded to GPTQ quantization.",
    )
    p.add_argument(
        "--quant_dataset_name",
        type=str,
        default="allenai/c4",
        help="Calibration dataset name for quantization methods that support datasets.",
    )
    p.add_argument(
        "--quant_dataset_split",
        type=str,
        default="train",
        help="Calibration dataset split for supported methods.",
    )
    p.add_argument(
        "--quant_dataset_config_name",
        type=str,
        default=None,
        help="Optional calibration dataset config name (e.g. wikitext-2-raw-v1).",
    )
    p.add_argument(
        "--quant_dataset_text_field",
        type=str,
        default="text",
        help="Calibration text field for supported methods.",
    )
    p.add_argument(
        "--quant_dataset_data_files",
        type=str,
        default="en/c4-train.00001-of-01024.json.gz",
        help=(
            "Optional datasets.load_dataset data_files for supported methods. "
            "Set empty string to disable."
        ),
    )
    p.add_argument(
        "--quant_dataset_cache_dir",
        type=str,
        default="~/Datasets",
        help="Calibration dataset cache dir for supported methods.",
    )
    p.add_argument(
        "--quant_calibration_texts_file",
        type=str,
        default=None,
        help="Optional newline-delimited calibration text file for supported methods.",
    )
    p.add_argument(
        "--quant_autoround_dataset",
        type=str,
        default="NeelNanda/pile-10k",
        help="Calibration dataset id for AutoRound.",
    )
    p.add_argument(
        "--special_quant",
        action="store_true",
        help="Include slower special quant methods (currently hqq and sinq).",
    )
    p.add_argument(
        "--quantization",
        "--quantizations",
        type=str,
        default=None,
        help=(
            "Optional quantization filter (comma-separated), e.g. "
            "'fp4,autoround,gguf_k,gguf_0,gguf_i' or run names like 'gptq_8bit'. "
            "Accepts both method names and run names."
        ),
    )
    p.add_argument(
        "--num_samples",
        type=int,
        default=1500,
        help="Max samples for attack-success eval. If unset, use all samples from the scenario dataset.",
    )
    p.add_argument("--asr_num_samples", dest="num_samples", type=int, help=argparse.SUPPRESS)
    p.add_argument(
        "--scenario",
        type=str,
        default="jailbreak",
        help="Scenario passed to Eval/calc_asr.py --scenarios.",
    )
    p.add_argument("--asr_scenario", dest="scenario", type=str, help=argparse.SUPPRESS)
    p.add_argument(
        "--mcd_dataset_path",
        type=str,
        default=DEFAULT_MCD_DATASET_PATH,
        help=(
            "Dataset path forwarded to Eval/calc_asr.py --mcd_dataset_path when "
            "--scenario mcd."
        ),
    )
    p.add_argument("--asr_mcd_dataset_path", dest="mcd_dataset_path", type=str, help=argparse.SUPPRESS)
    p.add_argument(
        "--inference_lib",
        type=str,
        default="vllm",
        choices=["vllm", "transformers", "hqq", "sinq", "llama_cpp"],
        help="Attack-success inference backend.",
    )
    p.add_argument("--asr_inference_lib", dest="inference_lib", type=str, help=argparse.SUPPRESS)
    p.add_argument(
        "--max_new_tokens",
        type=int,
        default=512,
        help="Max generated tokens per sample for attack-success inference.",
    )
    p.add_argument("--asr_max_new_tokens", dest="max_new_tokens", type=int, help=argparse.SUPPRESS)
    p.add_argument(
        "--gpu_memory_utilization",
        type=float,
        default=0.7,
    )
    p.add_argument("--asr_gpu_memory_utilization", dest="gpu_memory_utilization", type=float, help=argparse.SUPPRESS)
    p.add_argument(
        "--use_chat_template",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Apply chat template in attack-success evaluation.",
    )
    p.add_argument("--asr_use_chat_template", dest="use_chat_template", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--no-asr_use_chat_template", dest="use_chat_template", action="store_false", help=argparse.SUPPRESS)
    p.add_argument(
        "--prompt_format",
        type=str,
        default="auto",
        choices=["auto", "plain", "instruct"],
        help=(
            "Prompt format forwarded to Eval/calc_asr.py. "
            "'plain' uses plain prompts, 'instruct' forces chat formatting, "
            "and 'auto' follows --use_chat_template."
        ),
    )
    p.add_argument(
        "--hf_token",
        type=str,
        default=None,
        help=(
            "Optional Hugging Face token passed to Eval/calc_asr.py for private/gated models. "
            "If omitted, calc_asr.py may use HF_TOKEN/HUGGING_FACE_HUB_TOKEN from env."
        ),
    )
    p.add_argument("--asr_hf_token", dest="hf_token", type=str, help=argparse.SUPPRESS)
    p.add_argument(
        "--mode",
        type=str,
        choices=["all", "asr", "utility"],
        default="all",
        help="Execution/report mode. 'asr' ignores utility; 'utility' ignores ASR.",
    )
    p.add_argument(
        "--no_quantize_utility",
        dest="no_quantize_utility",
        action="store_true",
        default=True,
        help="Skip lm_eval for post-quantized models (default: enabled).",
    )
    p.add_argument(
        "--quantize_utility",
        dest="no_quantize_utility",
        action="store_false",
        help="Run lm_eval for post-quantized models.",
    )
    p.add_argument("--force", action="store_true", help="Force recomputation for calc_asr.")
    p.add_argument(
        "--no_original_asr",
        action="store_true",
        help="Skip ASR for --model_path (current model) while still running ASR for quantized models.",
    )
    p.add_argument(
        "--only_original_utility",
        action="store_true",
        help=(
            "Run only utility (lm_eval) for --model_path and skip baseline, quantization, "
            "and ASR. Compatible with --force for recomputing current-model utility."
        ),
    )
    p.add_argument("--dry_run", action="store_true")
    p.add_argument(
        "--report_current",
        action="store_true",
        help="Read existing outputs only and write a report without launching new runs.",
    )
    return p.parse_args()


def _parse_quantization_filter(raw: str | None) -> list[str]:
    if raw is None:
        return []
    out: list[str] = []
    for part in raw.split(","):
        token = part.strip().lower()
        if token and token not in out:
            out.append(token)
    return out


def is_gguf_method(method: str) -> bool:
    return method.lower() in GGUF_METHODS


def _terminate_process_group(proc: subprocess.Popen, sig: int = signal.SIGTERM) -> None:
    if proc.poll() is not None:
        return
    try:
        os.killpg(proc.pid, sig)
    except ProcessLookupError:
        return
    except Exception:
        return


def _terminate_all_active_children(sig: int = signal.SIGTERM) -> None:
    for proc in list(ACTIVE_CHILD_PROCS):
        _terminate_process_group(proc, sig=sig)


def _install_signal_handlers() -> None:
    global SIGNAL_HANDLERS_INSTALLED
    if SIGNAL_HANDLERS_INSTALLED:
        return

    def _handle_stop(_sig: int, _frame) -> None:
        _terminate_all_active_children(signal.SIGTERM)
        # Give children a brief chance to exit cleanly, then hard-kill leftovers.
        time.sleep(0.5)
        _terminate_all_active_children(signal.SIGKILL)
        raise KeyboardInterrupt

    signal.signal(signal.SIGINT, _handle_stop)
    signal.signal(signal.SIGTERM, _handle_stop)
    SIGNAL_HANDLERS_INSTALLED = True


def run_cmd(
    cmd: list[str],
    *,
    dry_run: bool,
    show_cmd: bool = True,
    extra_env: dict[str, str] | None = None,
) -> dict[str, Any]:
    env = os.environ.copy()
    if extra_env:
        for k, v in extra_env.items():
            env[str(k)] = str(v)
    cuda_visible_devices = env.get("CUDA_VISIBLE_DEVICES", "")
    if show_cmd:
        print(f"CUDA_VISIBLE_DEVICES={cuda_visible_devices}")
        if extra_env:
            print("ENV", " ".join(f"{k}={env[k]}" for k in sorted(extra_env)))
        print("$", " ".join(cmd))
    if dry_run:
        return {
            "ok": True,
            "returncode": 0,
            "stdout": "",
            "stderr": "",
            "dry_run": True,
            "cuda_visible_devices": cuda_visible_devices,
        }

    # Run each command in its own process group so we can reliably clean up
    # orphan grandchildren (e.g. lingering vLLM engine/workers).
    proc = subprocess.Popen(cmd, env=env, start_new_session=True)
    ACTIVE_CHILD_PROCS.add(proc)
    try:
        returncode = proc.wait()
    except KeyboardInterrupt:
        _terminate_process_group(proc, signal.SIGTERM)
        time.sleep(0.5)
        _terminate_process_group(proc, signal.SIGKILL)
        raise
    finally:
        ACTIVE_CHILD_PROCS.discard(proc)
    pgid = proc.pid
    # Best-effort group cleanup: terminate anything still alive in this group.
    try:
        os.killpg(pgid, 0)
    except Exception:
        pass
    else:
        try:
            os.killpg(pgid, signal.SIGTERM)
            time.sleep(0.5)
            os.killpg(pgid, 0)
        except Exception:
            pass
        else:
            try:
                os.killpg(pgid, signal.SIGKILL)
            except Exception:
                pass
    return {
        "ok": returncode == 0,
        "returncode": returncode,
        "stdout": "",
        "stderr": "",
        "dry_run": False,
        "cuda_visible_devices": cuda_visible_devices,
    }


def make_skipped_run(args: argparse.Namespace, reason: str) -> dict[str, Any]:
    return {
        "ok": False,
        "returncode": None,
        "stdout": "",
        "stderr": "",
        "dry_run": args.dry_run,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "skipped_reason": reason,
    }


def read_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def resolve_existing_lm_eval_json(base_output_json: Path) -> Path | None:
    # lm_eval may append a timestamp suffix, e.g. foo_2026-...json.
    if base_output_json.is_file():
        return base_output_json
    parent = base_output_json.parent
    stem = base_output_json.stem
    candidates = sorted(parent.glob(f"{stem}_*.json"))
    if candidates:
        return candidates[-1]
    return None


def normalize_lm_eval_json_name(base_output_json: Path) -> Path | None:
    existing = resolve_existing_lm_eval_json(base_output_json)
    if existing is None:
        return None
    if existing != base_output_json:
        base_output_json.parent.mkdir(parents=True, exist_ok=True)
        if base_output_json.exists():
            base_output_json.unlink()
        existing.replace(base_output_json)
        existing = base_output_json
    # Clean older timestamped variants once canonical file is in place.
    stem = base_output_json.stem
    for p in sorted(base_output_json.parent.glob(f"{stem}_*.json")):
        try:
            p.unlink()
        except Exception:
            pass
    return existing


def model_cache_tag(model_path: str) -> str:
    normalized = model_path.strip().replace("\\", "/").rstrip("/")
    base = normalized.split("/")[-1] if normalized else "model"
    safe = "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in base).strip("_")
    if not safe:
        safe = "model"
    digest = hashlib.sha1(normalized.encode("utf-8")).hexdigest()[:10]
    return f"{safe}_{digest}"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    out = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
                if isinstance(row, dict):
                    out.append(row)
            except Exception:
                continue
    return out


def quantized_artifact_exists(path: Path) -> bool:
    if path.is_file():
        return True
    if not path.is_dir():
        return False
    try:
        next(path.iterdir())
        return True
    except StopIteration:
        return False
    except Exception:
        return False


def extract_lm_eval_metrics(lm_eval_payload: dict[str, Any], tasks: list[str]) -> dict[str, Any]:
    results = lm_eval_payload.get("results", {})
    if not isinstance(results, dict):
        return {}

    preferred_metric_order = {
        "gsm8k": ["exact_match,strict-match", "exact_match", "acc,none", "acc"],
        "hellaswag": ["acc_norm,none", "acc_norm", "acc,none", "acc"],
        "mmlu": ["acc,none", "acc", "acc_norm,none", "acc_norm"],
        "arc_challenge": ["acc_norm,none", "acc_norm", "acc,none", "acc"],
        "humaneval": ["pass@1,none", "pass@1"],
    }
    extracted: dict[str, Any] = {}
    for task in tasks:
        task_result = results.get(task, {})
        if not isinstance(task_result, dict):
            extracted[task] = {"metric": None, "value": None}
            continue

        chosen_key = None
        # Prefer flexible-extract metrics when available.
        for key, value in task_result.items():
            if "flexible-extract" in str(key) and isinstance(value, (int, float)):
                chosen_key = key
                break

        for key in preferred_metric_order.get(task, []):
            if chosen_key is None and key in task_result:
                chosen_key = key
                break
        # Humaneval variants may expose pass@1 with task-specific suffixes.
        if chosen_key is None and task.startswith("humaneval"):
            for key, value in task_result.items():
                if str(key).startswith("pass@1") and isinstance(value, (int, float)):
                    chosen_key = key
                    break
        if chosen_key is None:
            for key, value in task_result.items():
                key_str = str(key).lower()
                # Skip common non-score numeric fields in generic fallback.
                if (
                    isinstance(value, (int, float))
                    and "stderr" not in key_str
                    and key_str not in {"sample_len", "num_samples", "n_samples", "samples"}
                ):
                    chosen_key = key
                    break

        extracted[task] = {
            "metric": chosen_key,
            "value": task_result.get(chosen_key) if chosen_key is not None else None,
            "raw": task_result,
        }
    return extracted


def print_lm_eval_task_summary(prefix: str, metrics: dict[str, Any]) -> None:
    for task in LM_EVAL_SUMMARY_TASKS:
        row = metrics.get(task, {})
        metric_name = row.get("metric")
        value = row.get("value")
        if metric_name is None or value is None:
            print(f"{prefix} {task}: missing")
            continue
        print(f"{prefix} {task}: {metric_name}={value}")


def extract_asr_from_jsonl(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"asr": None, "n_rows": 0, "source": "missing"}

    # Primary format from Eval/calc_asr.py outputs: one row per sample with `flg` in {0,1}.
    flg_values = []
    for row in rows:
        value = row.get("flg")
        if isinstance(value, bool):
            flg_values.append(1.0 if value else 0.0)
        elif isinstance(value, (int, float)):
            flg_values.append(float(value))
    if flg_values:
        return {
            "asr": float(sum(flg_values) / len(flg_values)),
            "n_rows": len(rows),
            "source": "mean_numeric.flg",
        }

    numeric_keys = ["asr", "attack_success_rate", "success_rate", "jailbreak_asr"]
    bool_keys = ["is_success", "success", "attack_success", "jailbreak_success"]

    last = rows[-1]
    for key in numeric_keys:
        value = last.get(key)
        if isinstance(value, (int, float)):
            return {"asr": float(value), "n_rows": len(rows), "source": f"last_row.{key}"}

    for key in bool_keys:
        values = [row.get(key) for row in rows if isinstance(row.get(key), bool)]
        if values:
            return {
                "asr": float(sum(1 for v in values if v) / len(values)),
                "n_rows": len(rows),
                "source": f"mean_bool.{key}",
            }

    return {"asr": None, "n_rows": len(rows), "source": "unrecognized"}


def build_lm_eval_cmd(
    args: argparse.Namespace,
    model_path: str,
    output_json: Path,
    *,
    task_string: str | None = None,
    apply_chat_template: bool | None = None,
) -> list[str]:
    selected_tasks = task_string if task_string is not None else args.lm_eval_tasks
    tasks = parse_lm_eval_tasks(selected_tasks)
    apply_template = args.lm_eval_apply_chat_template if apply_chat_template is None else apply_chat_template
    cmd = [
        args.lm_eval_bin,
        "--model",
        "vllm",
        "--model_args",
        f"pretrained={model_path},gpu_memory_utilization={args.lm_eval_gpu_memory_utilization}",
        "--tasks",
        selected_tasks,
        "--batch_size",
        str(args.lm_eval_batch_size),
        "--device",
        args.lm_eval_device,
        "--output_path",
        str(output_json),
    ]
    if lm_eval_requires_unsafe_code(tasks):
        cmd.append("--confirm_run_unsafe_code")
    if apply_template:
        cmd.append("--apply_chat_template")
    return cmd


def build_lm_eval_quant_cmd(
    *,
    args: argparse.Namespace,
    model_path: str,
    method: str,
    output_json: Path,
    task_string: str | None = None,
    apply_chat_template: bool | None = None,
) -> list[str]:
    selected_tasks = task_string if task_string is not None else args.lm_eval_tasks
    tasks = parse_lm_eval_tasks(selected_tasks)
    apply_template = args.lm_eval_apply_chat_template if apply_chat_template is None else apply_chat_template
    def _safe_batch_size(raw: str, cap: int) -> str:
        try:
            parsed = int(str(raw))
            if parsed <= 0:
                return str(cap)
            return str(min(parsed, cap))
        except Exception:
            return str(cap)

    # SINQ/HQQ checkpoints may not be loadable by lm_eval's standard vLLM/HF CLI path.
    # Route them through our dedicated adapter script.
    if method in {"hqq", "sinq"}:
        backend_batch_size = _safe_batch_size(
            args.lm_eval_batch_size,
            16 if method == "sinq" else 32,
        )
        cmd = [
            args.python_bin,
            "Eval/run_lm_eval_sinq.py",
            "--model_path",
            model_path,
            "--backend",
            method,
            "--tasks",
            selected_tasks,
            "--device",
            args.lm_eval_device,
            "--batch_size",
            backend_batch_size,
            "--dtype",
            "bfloat16",
            "--output_path",
            str(output_json),
            "--lm_eval_bin",
            args.lm_eval_bin,
        ]
        if lm_eval_requires_unsafe_code(tasks):
            cmd.append("--confirm_run_unsafe_code")
        if apply_template:
            cmd.append("--apply_chat_template")
        return cmd
    if is_gguf_method(method):
        default_quant_type = default_gguf_quant_type(method)
        gguf_files = sorted(Path(model_path).glob("*.gguf"))
        if not gguf_files:
            chosen_quant_type = args.gguf_quant_type or default_quant_type
            gguf_name = f"{args.gguf_model_name}.{chosen_quant_type.upper()}.gguf"
        else:
            quantized = [p for p in gguf_files if ".F16." not in p.name and ".F32." not in p.name and ".BF16." not in p.name]
            target = quantized[0] if quantized else gguf_files[-1]
            gguf_name = target.name
        gguf_path = str(Path(model_path) / gguf_name)
        model_args = (
            f"pretrained={model_path},"
            f"model={gguf_path},"
            f"tokenizer={model_path},"
            f"load_format=gguf,"
            f"hf_config_path={model_path},"
            f"dtype=float16,"
            f"gpu_memory_utilization={args.lm_eval_gpu_memory_utilization}"
        )
        cmd = [
            args.lm_eval_bin,
            "--model",
            "vllm",
            "--model_args",
            model_args,
            "--tasks",
            selected_tasks,
            "--batch_size",
            str(args.lm_eval_batch_size),
            "--device",
            args.lm_eval_device,
            "--output_path",
            str(output_json),
        ]
        if lm_eval_requires_unsafe_code(tasks):
            cmd.append("--confirm_run_unsafe_code")
        if apply_template:
            cmd.append("--apply_chat_template")
        return cmd
    return build_lm_eval_cmd(
        args,
        model_path,
        output_json,
        task_string=selected_tasks,
        apply_chat_template=apply_template,
    )


def _merge_lm_eval_payloads(payloads: list[dict[str, Any]]) -> dict[str, Any]:
    if not payloads:
        return {}
    merged = dict(payloads[0])
    for payload in payloads[1:]:
        for key in ("results", "configs", "versions", "n-shot"):
            a = merged.get(key)
            b = payload.get(key)
            if isinstance(a, dict) and isinstance(b, dict):
                a.update(b)
                merged[key] = a
            elif b is not None and key not in merged:
                merged[key] = b
    return merged


def run_lm_eval_split_aware(
    *,
    args: argparse.Namespace,
    model_path: str,
    output_json: Path,
    method: str | None,
    dry_run: bool,
    show_cmd: bool,
) -> tuple[list[list[str]], dict[str, Any], dict[str, Any] | None, Path | None]:
    tasks = parse_lm_eval_tasks(args.lm_eval_tasks)
    safe_tasks, unsafe_tasks = split_lm_eval_tasks(tasks)

    jobs: list[tuple[str, str, bool, Path]] = []
    if safe_tasks:
        safe_task_string = ",".join(safe_tasks)
        safe_output = output_json if not unsafe_tasks else output_json.with_name(f"{output_json.stem}_safe.json")
        jobs.append(("safe", safe_task_string, args.lm_eval_apply_chat_template, safe_output))
    if unsafe_tasks:
        unsafe_task_string = ",".join(unsafe_tasks)
        unsafe_output = output_json if not safe_tasks else output_json.with_name(f"{output_json.stem}_unsafe.json")
        jobs.append(("unsafe", unsafe_task_string, False, unsafe_output))

    existing = normalize_lm_eval_json_name(output_json) if (args.report_current or not args.force) else None
    if existing is not None:
        run = {
            "ok": True,
            "returncode": 0,
            "stdout": "",
            "stderr": "",
            "dry_run": dry_run,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
            "skipped_reason": "used_existing_lm_eval_json",
        }
        payload = read_json(existing)
        commands: list[list[str]] = []
        for _, task_string, apply_template, out_json in jobs:
            if method is None:
                commands.append(
                    build_lm_eval_cmd(
                        args,
                        model_path,
                        out_json,
                        task_string=task_string,
                        apply_chat_template=apply_template,
                    )
                )
            else:
                commands.append(
                    build_lm_eval_quant_cmd(
                        args=args,
                        model_path=model_path,
                        method=method,
                        output_json=out_json,
                        task_string=task_string,
                        apply_chat_template=apply_template,
                    )
                )
        return commands, run, payload, existing

    if args.report_current:
        return [], make_skipped_run(args, "report_current_missing_lm_eval_json"), None, None

    commands: list[list[str]] = []
    payloads: list[dict[str, Any]] = []
    sub_runs: list[dict[str, Any]] = []
    for _, task_string, apply_template, out_json in jobs:
        if method is None:
            cmd = build_lm_eval_cmd(
                args,
                model_path,
                out_json,
                task_string=task_string,
                apply_chat_template=apply_template,
            )
        else:
            cmd = build_lm_eval_quant_cmd(
                args=args,
                model_path=model_path,
                method=method,
                output_json=out_json,
                task_string=task_string,
                apply_chat_template=apply_template,
            )
        commands.append(cmd)
        run = run_cmd(cmd, dry_run=dry_run, show_cmd=show_cmd)
        sub_runs.append({k: v for k, v in run.items() if k not in {"stdout", "stderr"}})
        if not run["ok"]:
            failed = dict(run)
            failed["sub_runs"] = sub_runs
            return commands, failed, None, None
        written = normalize_lm_eval_json_name(out_json)
        payload = read_json(written) if written is not None else None
        if payload is not None:
            payloads.append(payload)

    merged_payload = _merge_lm_eval_payloads(payloads) if payloads else None
    canonical: Path | None = None
    if merged_payload is not None:
        output_json.parent.mkdir(parents=True, exist_ok=True)
        with output_json.open("w", encoding="utf-8") as f:
            json.dump(merged_payload, f, indent=2, ensure_ascii=True)
            f.write("\n")
        canonical = output_json
    final_run = {
        "ok": True,
        "returncode": 0,
        "stdout": "",
        "stderr": "",
        "dry_run": dry_run,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
    }
    if len(sub_runs) > 1:
        final_run["sub_runs"] = sub_runs
    return commands, final_run, merged_payload, canonical


def build_quant_cmd(
    *,
    python_bin: str,
    input_model_path: str,
    output_model_path: Path,
    method: str,
    bits: int,
    group_size: int,
    seed: int,
    quant_nsamples: int,
    quant_batch_size: int,
    quant_no_quiet: bool,
    quant_dataset: str,
    quant_dataset_name: str,
    quant_dataset_split: str,
    quant_dataset_config_name: str | None,
    quant_dataset_text_field: str,
    quant_dataset_data_files: str,
    quant_dataset_cache_dir: str,
    quant_calibration_texts_file: str | None,
    quant_autoround_dataset: str,
) -> list[str]:
    if method == "smoothquant":
        return [
            python_bin,
            "Quantization/quantization_smoothquant.py",
            "--model_path",
            input_model_path,
            "--output_path",
            str(output_model_path),
            "--nsamples",
            str(quant_nsamples),
            "--seed",
            str(seed),
        ]

    if is_gguf_method(method):
        default_quant_type = default_gguf_quant_type(method)
        cmd = [
            python_bin,
            "Quantization/quantization_gguf.py",
            "--model_path",
            input_model_path,
            "--output_path",
            str(output_model_path),
            "--quant_type",
            default_quant_type,
            "--outtype",
            "f16",
            "--model_name",
            "model",
        ]
        return cmd

    cmd = [
        python_bin,
        "Quantization/quantization.py",
        "--model_path",
        input_model_path,
        "--output_path",
        str(output_model_path),
        "--method",
        method,
        "--bits",
        str(bits),
        "--group_size",
        str(group_size),
        "--seed",
        str(seed),
    ]
    if method in {"gptq", "awq"}:
        cmd.extend(["--nsamples", str(quant_nsamples)])
    if method in {"gptq", "awq"}:
        cmd.extend(["--dataset", quant_dataset])
    if method in {"gptq", "awq"}:
        # Keep C4 explicit wiring by default, but avoid forcing C4-specific
        # values onto other presets (wikitext/openwebtext/pile/etc).
        using_c4_defaults = (
            quant_dataset_name == "allenai/c4"
            and quant_dataset_split == "train"
            and quant_dataset_text_field == "text"
            and quant_dataset_data_files == "en/c4-train.00001-of-01024.json.gz"
            and quant_dataset_config_name is None
        )
        pass_explicit_dataset_wiring = (quant_dataset == "c4") or (not using_c4_defaults)
        if pass_explicit_dataset_wiring:
            if quant_dataset_name:
                cmd.extend(["--dataset_name", quant_dataset_name])
            if quant_dataset_config_name:
                cmd.extend(["--dataset_config_name", quant_dataset_config_name])
            if quant_dataset_split:
                cmd.extend(["--dataset_split", quant_dataset_split])
            if quant_dataset_text_field:
                cmd.extend(["--dataset_text_field", quant_dataset_text_field])
            if quant_dataset_data_files is not None:
                cmd.extend(["--dataset_data_files", quant_dataset_data_files])
        cmd.extend(["--dataset_cache_dir", quant_dataset_cache_dir])
        if quant_calibration_texts_file:
            cmd.extend(["--calibration_texts_file", quant_calibration_texts_file])
    if method == "awq":
        # Prevent AWQ from silently retrying without data_files on huge corpora.
        cmd.append("--no-allow_full_dataset_fallback")
    if method in {"gptq"}:
        cmd.extend(["--batch_size", str(quant_batch_size)])
        if quant_no_quiet:
            cmd.append("--no-quiet")
    if method == "autoround":
        cmd.extend(["--nsamples", str(quant_nsamples), "--batch_size", str(quant_batch_size)])
        if quant_autoround_dataset:
            cmd.extend(["--dataset", quant_autoround_dataset])
    return cmd


def build_asr_cmd(
    *,
    python_bin: str,
    model_path: str | Path,
    output_dir: Path,
    scenario: str,
    num_samples: int | None,
    inference_lib: str,
    gpu_memory_utilization: float,
    use_chat_template: bool,
    prompt_format: str,
    hf_token: str | None,
    mcd_dataset_path: str,
    force: bool,
    max_new_tokens: int | None,
) -> list[str]:
    cmd = [
        python_bin,
        "Eval/calc_asr.py",
        "--model_path",
        str(model_path),
        "--scenarios",
        scenario,
        "--output_dir",
        str(output_dir),
        "--inference_lib",
        inference_lib,
        "--gpu_memory_utilization",
        str(gpu_memory_utilization),
    ]
    if num_samples is not None:
        cmd.extend(["--num_samples", str(num_samples)])
    if max_new_tokens is not None:
        cmd.extend(["--max_new_tokens", str(max_new_tokens)])
    cmd.extend(["--prompt_format", prompt_format])
    if use_chat_template:
        cmd.append("--use_chat_template")
    if hf_token:
        cmd.extend(["--hf_token", hf_token])
    if scenario == "mcd":
        cmd.extend(["--mcd_dataset_path", mcd_dataset_path])
    if force:
        cmd.append("--force")
    return cmd


def resolve_quant_asr_inference_lib(default_inference_lib: str, method: str) -> str:
    # Quantized HQQ/SINQ checkpoints should use the matching transformers backend alias.
    if method in {"hqq", "sinq"}:
        return method
    if is_gguf_method(method):
        # GGUF ASR is more robust via llama.cpp in this pipeline.
        # This avoids vLLM GGUF edge cases (e.g. accidentally picking non-model GGUF side files).
        return "llama_cpp"
    return default_inference_lib


def resolve_default_gguf_llama_cpp_dir(arg_value: str | None) -> str | None:
    if arg_value:
        return arg_value
    env_dir = os.environ.get("LLAMA_CPP_DIR")
    candidates = [env_dir, "llama.cpp"] if env_dir else ["llama.cpp"]
    for c in candidates:
        if (Path(c) / "convert_hf_to_gguf.py").is_file():
            return c
    return None


def main() -> None:
    maybe_reexec_orchestrator()
    _install_signal_handlers()
    args = parse_args()
    if args.report_current and args.force:
        print("Ignoring --force because --report_current only reads existing artifacts.")
    args.gguf_llama_cpp_dir = resolve_default_gguf_llama_cpp_dir(args.gguf_llama_cpp_dir)
    args.output_path.mkdir(parents=True, exist_ok=True)
    report_mode = args.mode
    if args.only_original_utility:
        report_mode = "utility"
    run_utility = report_mode in {"all", "utility"}
    run_asr = (report_mode in {"all", "asr"}) and (not args.only_original_utility)
    run_current_model_asr = run_asr and (not args.no_original_asr)
    run_quant_utility = run_utility and (not args.no_quantize_utility)
    if args.only_original_utility:
        run_quant_utility = False
    report_utility = run_utility
    report_asr = run_asr
    asr_only_output = report_mode == "asr"

    baseline_model_path = str(args.baseline_model_path) if args.baseline_model_path else None
    baseline_is_designated = (baseline_model_path is not None) and (not args.only_original_utility)
    baseline_tag = model_cache_tag(baseline_model_path) if baseline_is_designated else None

    run_root = args.output_path
    lm_eval_dir = run_root / "lm_eval"
    quant_root = run_root / "quantized_models"
    asr_root = run_root / "asr"
    lm_eval_dir.mkdir(parents=True, exist_ok=True)
    quant_root.mkdir(parents=True, exist_ok=True)
    asr_root.mkdir(parents=True, exist_ok=True)

    tasks = parse_lm_eval_tasks(args.lm_eval_tasks)
    timestamp = datetime.now(timezone.utc).isoformat()

    report: dict[str, Any] = {
        "created_at_utc": timestamp,
        "input_model_path": args.model_path,
        "baseline_model_path": baseline_model_path,
        "settings": {
            "lm_eval_tasks": tasks,
            "lm_eval_batch_size": args.lm_eval_batch_size,
            "lm_eval_device": args.lm_eval_device,
            "lm_eval_gpu_memory_utilization": args.lm_eval_gpu_memory_utilization,
            "lm_eval_apply_chat_template": args.lm_eval_apply_chat_template,
            "seed": args.seed,
            "group_size": args.group_size,
            "gguf_quant_type": args.gguf_quant_type,
            "gguf_outtype": args.gguf_outtype,
            "gguf_model_name": args.gguf_model_name,
            "gguf_llama_cpp_dir": args.gguf_llama_cpp_dir,
            "gguf_converter": args.gguf_converter,
            "gguf_quantize_bin": args.gguf_quantize_bin,
            "gguf_i_require_imatrix": args.gguf_i_require_imatrix,
            "gguf_i_auto_build_imatrix": args.gguf_i_auto_build_imatrix,
            "gguf_ctx_size": args.gguf_ctx_size,
            "quant_nsamples": args.quant_nsamples,
            "quant_batch_size": args.quant_batch_size,
            "quant_no_quiet": args.quant_no_quiet,
            "quant_dataset": args.quant_dataset,
            "quant_dataset_name": args.quant_dataset_name,
            "quant_dataset_split": args.quant_dataset_split,
            "quant_dataset_config_name": args.quant_dataset_config_name,
            "quant_dataset_text_field": args.quant_dataset_text_field,
            "quant_dataset_data_files": args.quant_dataset_data_files,
            "quant_dataset_cache_dir": args.quant_dataset_cache_dir,
            "quant_calibration_texts_file": args.quant_calibration_texts_file,
            "quant_autoround_dataset": args.quant_autoround_dataset,
            "special_quant": args.special_quant,
            "quantization_filter": args.quantization,
            "num_samples": args.num_samples,
            "scenario": args.scenario,
            "mcd_dataset_path": args.mcd_dataset_path,
            "inference_lib": args.inference_lib,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "max_new_tokens": args.max_new_tokens,
            "use_chat_template": args.use_chat_template,
            "prompt_format": args.prompt_format,
            "hf_token_provided": bool(args.hf_token),
            "report_mode": report_mode,
            "no_original_asr": args.no_original_asr,
            "only_original_utility": args.only_original_utility,
            "no_quantize_utility": args.no_quantize_utility,
            "baseline_model_is_designated": baseline_is_designated,
            "baseline_cache_tag": baseline_tag,
            "force": args.force,
            "dry_run": args.dry_run,
            "report_current": args.report_current,
        },
        "baseline_utility": {},
        "baseline_attack_success_rate": {},
        "current_model_utility": {},
        "current_model_attack_success_rate": {},
        "quantized_models": [],
    }

    quant_runs: list[dict[str, Any]] = []
    if not args.only_original_utility:
        quant_runs = list(BASE_QUANT_RUNS)
        if args.special_quant:
            quant_runs.extend(SPECIAL_QUANT_RUNS)
    requested_quants = _parse_quantization_filter(args.quantization)
    if requested_quants:
        expanded_requested = set(requested_quants)
        # Alias support for shorthand GPTQ names.
        if "gpt8b" in expanded_requested:
            expanded_requested.add("gptq_8bit")
        if "gpt4b" in expanded_requested:
            expanded_requested.add("gptq_4bit")
        if "gguf" in expanded_requested:
            expanded_requested.update({"gguf_k", "gguf_0", "gguf_i"})
        # Backward compatibility for prior run names.
        if "gguf_q4_k_m" in expanded_requested:
            expanded_requested.add("gguf_k")
        if "gguf_q4_0" in expanded_requested:
            expanded_requested.add("gguf_0")
        if "gguf_q" in expanded_requested:
            expanded_requested.add("gguf_0")
        if "gguf_iq4_xs" in expanded_requested:
            expanded_requested.add("gguf_i")

        all_runs = list(BASE_QUANT_RUNS) + list(SPECIAL_QUANT_RUNS)
        selected_runs: list[dict[str, Any]] = []
        for cfg in all_runs:
            name = str(cfg["name"]).lower()
            method = str(cfg["method"]).lower()
            if name in expanded_requested or method in expanded_requested:
                selected_runs.append(cfg)
        if not selected_runs:
            available = sorted(
                {str(r["name"]) for r in all_runs}
                | {str(r["method"]) for r in all_runs}
            )
            raise ValueError(
                "No quantization run matches --quantization. "
                f"Requested={requested_quants}. Available={available}"
            )
        deduped: list[dict[str, Any]] = []
        seen_names: set[str] = set()
        for cfg in selected_runs:
            run_name = str(cfg["name"])
            if run_name not in seen_names:
                deduped.append(cfg)
                seen_names.add(run_name)
        quant_runs = deduped

    if baseline_is_designated:
        baseline_lm_eval_json = lm_eval_dir / f"baseline_{baseline_tag}_lm_eval.json"
        baseline_lm_cmds: list[list[str]] = []
        baseline_existing_json: Path | None = None
        if not run_utility:
            baseline_lm_run = {
                "ok": False,
                "returncode": None,
                "stdout": "",
                "stderr": "",
                "dry_run": args.dry_run,
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
                "skipped_reason": "disabled_by_mode",
            }
            baseline_lm_payload = None
        else:
            baseline_lm_cmds, baseline_lm_run, baseline_lm_payload, baseline_existing_json = run_lm_eval_split_aware(
                args=args,
                model_path=baseline_model_path,
                output_json=baseline_lm_eval_json,
                method=None,
                dry_run=args.dry_run,
                show_cmd=not asr_only_output,
            )
            if baseline_existing_json is not None and baseline_lm_run.get("skipped_reason") == "used_existing_lm_eval_json" and not asr_only_output:
                print(f"Using existing lm_eval file: {baseline_existing_json}")
        baseline_metrics = extract_lm_eval_metrics(baseline_lm_payload, tasks) if baseline_lm_payload else {}
        if baseline_metrics:
            print_lm_eval_task_summary("[baseline]", baseline_metrics)

        baseline_asr_dir = asr_root / f"baseline_{baseline_tag}"
        baseline_asr_path = baseline_asr_dir / f"asr_{args.scenario}.jsonl"
        baseline_asr_cmd = build_asr_cmd(
            python_bin=args.python_bin,
            model_path=baseline_model_path,
            output_dir=baseline_asr_dir,
            scenario=args.scenario,
            num_samples=args.num_samples,
            inference_lib=args.inference_lib,
            gpu_memory_utilization=args.gpu_memory_utilization,
            use_chat_template=args.use_chat_template,
            prompt_format=args.prompt_format,
            hf_token=args.hf_token,
            mcd_dataset_path=args.mcd_dataset_path,
            force=args.force,
            max_new_tokens=args.max_new_tokens,
        )
        if not run_asr:
            baseline_asr_run = {
                "ok": False,
                "returncode": None,
                "stdout": "",
                "stderr": "",
                "dry_run": args.dry_run,
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
                "skipped_reason": "disabled_by_mode",
            }
            baseline_asr_summary = {"asr": None, "n_rows": 0, "source": "disabled_by_mode"}
        elif baseline_asr_path.is_file() and (args.report_current or not args.force):
            baseline_asr_run = {
                "ok": True,
                "returncode": 0,
                "stdout": "",
                "stderr": "",
                "dry_run": args.dry_run,
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
                "skipped_reason": "used_existing_asr_jsonl",
            }
            baseline_asr_rows = read_jsonl(baseline_asr_path)
            baseline_asr_summary = extract_asr_from_jsonl(baseline_asr_rows)
            baseline_asr_val = baseline_asr_summary.get("asr")
            baseline_asr_text = f"{float(baseline_asr_val):.6f}" if isinstance(baseline_asr_val, (int, float)) else "missing"
            if not asr_only_output:
                print(
                    f"Using cached baseline ASR={baseline_asr_text} "
                    f"(n={baseline_asr_summary.get('n_rows')}, source={baseline_asr_summary.get('source')})."
                )
        elif args.report_current:
            baseline_asr_run = make_skipped_run(args, "report_current_missing_asr_jsonl")
            baseline_asr_summary = {"asr": None, "n_rows": 0, "source": "missing"}
        else:
            baseline_asr_run = run_cmd(baseline_asr_cmd, dry_run=args.dry_run, show_cmd=not asr_only_output)
            baseline_asr_rows = read_jsonl(baseline_asr_path) if baseline_asr_run["ok"] else []
            baseline_asr_summary = extract_asr_from_jsonl(baseline_asr_rows) if baseline_asr_run["ok"] else {
                "asr": None,
                "n_rows": 0,
                "source": "skipped",
            }

        report["baseline_utility"] = (
            {
                "model_path": baseline_model_path,
                "lm_eval_command": baseline_lm_cmds[0] if len(baseline_lm_cmds) == 1 else baseline_lm_cmds,
                "lm_eval_run": {k: v for k, v in baseline_lm_run.items() if k not in {"stdout", "stderr"}},
                "metrics": baseline_metrics,
                "output_json": str(baseline_existing_json or baseline_lm_eval_json),
            }
            if report_utility
            else {}
        )
        report["baseline_attack_success_rate"] = (
            {
                "model_path": baseline_model_path,
                "calc_asr_command": baseline_asr_cmd,
                "calc_asr_run": {k: v for k, v in baseline_asr_run.items() if k not in {"stdout", "stderr"}},
                "scenario": args.scenario,
                "summary": baseline_asr_summary,
                "output_jsonl": str(baseline_asr_path),
            }
            if report_asr
            else {}
        )

    # Always evaluate the current model, regardless of baseline availability.
    evaluate_current_model = True
    if evaluate_current_model:
        current_model_path = str(args.model_path)
        current_lm_eval_json = lm_eval_dir / "current_model_lm_eval.json"
        current_lm_cmds: list[list[str]] = []
        current_existing_json: Path | None = None
        if not run_utility:
            current_lm_run = {
                "ok": False,
                "returncode": None,
                "stdout": "",
                "stderr": "",
                "dry_run": args.dry_run,
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
                "skipped_reason": "disabled_by_mode",
            }
            current_lm_payload = None
        else:
            current_lm_cmds, current_lm_run, current_lm_payload, current_existing_json = run_lm_eval_split_aware(
                args=args,
                model_path=current_model_path,
                output_json=current_lm_eval_json,
                method=None,
                dry_run=args.dry_run,
                show_cmd=not asr_only_output,
            )
            if current_existing_json is not None and current_lm_run.get("skipped_reason") == "used_existing_lm_eval_json" and not asr_only_output:
                print(f"Using existing lm_eval file: {current_existing_json}")
        current_metrics = extract_lm_eval_metrics(current_lm_payload, tasks) if current_lm_payload else {}
        if current_metrics:
            print_lm_eval_task_summary("[current_model]", current_metrics)

        current_asr_dir = asr_root / "current_model"
        current_asr_path = current_asr_dir / f"asr_{args.scenario}.jsonl"
        current_asr_cmd = build_asr_cmd(
            python_bin=args.python_bin,
            model_path=current_model_path,
            output_dir=current_asr_dir,
            scenario=args.scenario,
            num_samples=args.num_samples,
            inference_lib=args.inference_lib,
            gpu_memory_utilization=args.gpu_memory_utilization,
            use_chat_template=args.use_chat_template,
            prompt_format=args.prompt_format,
            hf_token=args.hf_token,
            mcd_dataset_path=args.mcd_dataset_path,
            force=args.force,
            max_new_tokens=args.max_new_tokens,
        )
        if not run_current_model_asr:
            skipped_reason = "disabled_by_flag_no_original_asr" if args.no_original_asr else "disabled_by_mode"
            current_asr_run = {
                "ok": False,
                "returncode": None,
                "stdout": "",
                "stderr": "",
                "dry_run": args.dry_run,
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
                "skipped_reason": skipped_reason,
            }
            current_asr_summary = {"asr": None, "n_rows": 0, "source": skipped_reason}
        elif current_asr_path.is_file() and (args.report_current or not args.force):
            current_asr_run = {
                "ok": True,
                "returncode": 0,
                "stdout": "",
                "stderr": "",
                "dry_run": args.dry_run,
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
                "skipped_reason": "used_existing_asr_jsonl",
            }
            current_asr_rows = read_jsonl(current_asr_path)
            current_asr_summary = extract_asr_from_jsonl(current_asr_rows)
            current_asr_val = current_asr_summary.get("asr")
            current_asr_text = f"{float(current_asr_val):.6f}" if isinstance(current_asr_val, (int, float)) else "missing"
            if not asr_only_output:
                print(
                    f"Using cached current_model ASR={current_asr_text} "
                    f"(n={current_asr_summary.get('n_rows')}, source={current_asr_summary.get('source')})."
                )
        elif args.report_current:
            current_asr_run = make_skipped_run(args, "report_current_missing_asr_jsonl")
            current_asr_summary = {"asr": None, "n_rows": 0, "source": "missing"}
        else:
            current_asr_run = run_cmd(current_asr_cmd, dry_run=args.dry_run, show_cmd=not asr_only_output)
            current_asr_rows = read_jsonl(current_asr_path) if current_asr_run["ok"] else []
            current_asr_summary = extract_asr_from_jsonl(current_asr_rows) if current_asr_run["ok"] else {
                "asr": None,
                "n_rows": 0,
                "source": "skipped",
            }

        report["current_model_utility"] = (
            {
                "model_path": current_model_path,
                "lm_eval_command": current_lm_cmds[0] if len(current_lm_cmds) == 1 else current_lm_cmds,
                "lm_eval_run": {k: v for k, v in current_lm_run.items() if k not in {"stdout", "stderr"}},
                "metrics": current_metrics,
                "output_json": str(current_existing_json or current_lm_eval_json),
            }
            if report_utility
            else {}
        )
        report["current_model_attack_success_rate"] = (
            {
                "model_path": current_model_path,
                "calc_asr_command": current_asr_cmd,
                "calc_asr_run": {k: v for k, v in current_asr_run.items() if k not in {"stdout", "stderr"}},
                "scenario": args.scenario,
                "summary": current_asr_summary,
                "output_jsonl": str(current_asr_path),
            }
            if report_asr
            else {}
        )

    for quant_cfg in quant_runs:
        name = str(quant_cfg["name"])
        method = str(quant_cfg["method"])
        bits = int(quant_cfg["bits"])
        quant_model_dir = quant_root / name
        quant_model_dir.parent.mkdir(parents=True, exist_ok=True)

        quant_cmd = build_quant_cmd(
            python_bin=args.python_bin,
            input_model_path=args.model_path,
            output_model_path=quant_model_dir,
            method=method,
            bits=bits,
            group_size=args.group_size,
            seed=args.seed,
            quant_nsamples=args.quant_nsamples,
            quant_batch_size=args.quant_batch_size,
            quant_no_quiet=args.quant_no_quiet,
            quant_dataset=args.quant_dataset,
            quant_dataset_name=args.quant_dataset_name,
            quant_dataset_split=args.quant_dataset_split,
            quant_dataset_config_name=args.quant_dataset_config_name,
            quant_dataset_text_field=args.quant_dataset_text_field,
            quant_dataset_data_files=args.quant_dataset_data_files,
            quant_dataset_cache_dir=args.quant_dataset_cache_dir,
            quant_calibration_texts_file=args.quant_calibration_texts_file,
            quant_autoround_dataset=args.quant_autoround_dataset,
        )
        pre_quant_cmd: list[str] | None = None
        if is_gguf_method(method):
            default_quant_type = default_gguf_quant_type(method)
            gguf_quant_type = str(quant_cfg.get("gguf_quant_type", args.gguf_quant_type or default_quant_type))
            # Override defaults for GGUF from pipeline args.
            quant_cmd = [
                args.python_bin,
                "Quantization/quantization_gguf.py",
                "--model_path",
                args.model_path,
                "--output_path",
                str(quant_model_dir),
                "--quant_type",
                gguf_quant_type,
                "--outtype",
                args.gguf_outtype,
                "--model_name",
                args.gguf_model_name,
            ]
            if args.gguf_llama_cpp_dir:
                quant_cmd.extend(["--llama_cpp_dir", args.gguf_llama_cpp_dir])
            if args.gguf_converter:
                quant_cmd.extend(["--converter", args.gguf_converter])
            if args.gguf_quantize_bin:
                quant_cmd.extend(["--quantize_bin", args.gguf_quantize_bin])
            if method == "gguf_i":
                base_gguf = resolve_gguf_i_base_gguf(
                    quant_root=quant_root,
                    gguf_model_name=args.gguf_model_name,
                    gguf_outtype=args.gguf_outtype,
                )
                if base_gguf is not None:
                    quant_cmd.extend(["--base_gguf", str(base_gguf)])

                imatrix_path = resolve_gguf_i_imatrix(quant_model_dir)
                if imatrix_path is None and args.gguf_i_auto_build_imatrix and (not args.report_current):
                    if base_gguf is None:
                        raise FileNotFoundError(
                            "gguf_i auto-build requested but no reusable base GGUF found. "
                            f"Expected one of: {quant_root / 'gguf_k' / _default_gguf_base_name(args.gguf_model_name, args.gguf_outtype)}, "
                            f"{quant_root / 'gguf_0' / _default_gguf_base_name(args.gguf_model_name, args.gguf_outtype)}"
                        )
                    pre_quant_cmd = build_gguf_i_imatrix_cmd(
                        args=args,
                        base_gguf=base_gguf,
                        quant_model_dir=quant_model_dir,
                    )
                    imatrix_path = quant_model_dir / "imatrix.fast.gguf"

                if imatrix_path is not None:
                    quant_cmd.extend(["--imatrix", str(imatrix_path)])
                elif args.gguf_i_require_imatrix and (not args.report_current):
                    raise FileNotFoundError(
                        "gguf_i requires imatrix but none found. "
                        f"Expected one of: {quant_model_dir / 'imatrix.fast.gguf'}, "
                        f"{quant_model_dir / 'imatrix.gguf'}, {quant_model_dir / 'imatrix.dat'}"
                    )

        quant_lm_eval_json = lm_eval_dir / f"{name}_lm_eval.json"
        quant_lm_cmds: list[list[str]] = []
        quant_asr_model_path: Path | str = quant_model_dir
        if is_gguf_method(method):
            try:
                quant_asr_model_path = resolve_gguf_quantized_model_file(
                    model_dir=quant_model_dir,
                    method=method,
                    args=args,
                )
            except FileNotFoundError:
                # Quantization may not have happened yet in this loop; resolve again later.
                quant_asr_model_path = quant_model_dir
        asr_dir = asr_root / name
        asr_jailbreak_path = asr_dir / f"asr_{args.scenario}.jsonl"

        has_cached_quant_model = quantized_artifact_exists(quant_model_dir)
        quant_existing_json = normalize_lm_eval_json_name(quant_lm_eval_json) if (args.report_current or not args.force) else None
        has_cached_asr = asr_jailbreak_path.is_file()
        cache_ready_for_mode = ((not run_quant_utility) or (quant_existing_json is not None)) and ((not run_asr) or has_cached_asr)
        if args.report_current:
            quant_run = make_skipped_run(args, "report_current_no_quantization")
            if run_quant_utility:
                quant_lm_cmds, quant_lm_run, quant_lm_payload, quant_existing_json = run_lm_eval_split_aware(
                    args=args,
                    model_path=str(quant_model_dir),
                    output_json=quant_lm_eval_json,
                    method=method,
                    dry_run=args.dry_run,
                    show_cmd=not asr_only_output,
                )
                if quant_existing_json is not None and quant_lm_run.get("skipped_reason") == "used_existing_lm_eval_json" and not asr_only_output:
                    print(f"Using existing lm_eval file for {name}: {quant_existing_json}")
            else:
                quant_lm_run = make_skipped_run(args, "disabled_by_mode_or_no_quantize_utility")
                quant_lm_payload = None

            if run_asr and has_cached_asr:
                cached_asr_rows = read_jsonl(asr_jailbreak_path)
                asr_run = {
                    "ok": True,
                    "returncode": 0,
                    "stdout": "",
                    "stderr": "",
                    "dry_run": args.dry_run,
                    "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
                    "skipped_reason": "used_existing_asr_jsonl",
                }
                asr_summary = extract_asr_from_jsonl(cached_asr_rows)
            elif run_asr:
                asr_run = make_skipped_run(args, "report_current_missing_asr_jsonl")
                asr_summary = {"asr": None, "n_rows": 0, "source": "missing"}
            else:
                asr_run = make_skipped_run(args, "disabled_by_mode")
                asr_summary = {"asr": None, "n_rows": 0, "source": "disabled_by_mode"}
        elif (not args.force) and cache_ready_for_mode:
            quant_run = {
                "ok": True,
                "returncode": 0,
                "stdout": "",
                "stderr": "",
                "dry_run": args.dry_run,
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
                "skipped_reason": "used_cached_lm_eval_and_asr",
            }
            if run_quant_utility:
                quant_lm_cmds, quant_lm_run, quant_lm_payload, quant_existing_json = run_lm_eval_split_aware(
                    args=args,
                    model_path=str(quant_model_dir),
                    output_json=quant_lm_eval_json,
                    method=method,
                    dry_run=args.dry_run,
                    show_cmd=not asr_only_output,
                )
                if quant_existing_json is not None and quant_lm_run.get("skipped_reason") == "used_existing_lm_eval_json" and not asr_only_output:
                    print(f"Using existing lm_eval file for {name}: {quant_existing_json}")
            else:
                quant_lm_run = {
                    "ok": False,
                    "returncode": None,
                    "stdout": "",
                    "stderr": "",
                    "dry_run": args.dry_run,
                    "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
                    "skipped_reason": "disabled_by_mode_or_no_quantize_utility",
                }
                quant_lm_payload = None
            if run_asr:
                cached_asr_rows = read_jsonl(asr_jailbreak_path)
                asr_run = {
                    "ok": True,
                    "returncode": 0,
                    "stdout": "",
                    "stderr": "",
                    "dry_run": args.dry_run,
                    "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
                    "skipped_reason": "used_existing_asr_jsonl",
                }
                asr_rows = cached_asr_rows
                asr_summary = extract_asr_from_jsonl(cached_asr_rows)
            else:
                asr_run = {
                    "ok": False,
                    "returncode": None,
                    "stdout": "",
                    "stderr": "",
                    "dry_run": args.dry_run,
                    "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
                    "skipped_reason": "disabled_by_mode",
                }
                asr_summary = {"asr": None, "n_rows": 0, "source": "disabled_by_mode"}
        else:
            if (not args.force) and has_cached_quant_model:
                quant_run = {
                    "ok": True,
                    "returncode": 0,
                    "stdout": "",
                    "stderr": "",
                    "dry_run": args.dry_run,
                    "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
                    "skipped_reason": "used_existing_quantized_model",
                }
                if not asr_only_output:
                    print(f"Using existing quantized model for {name}: {quant_model_dir}")
            else:
                if pre_quant_cmd is not None:
                    pre_quant_run = run_cmd(pre_quant_cmd, dry_run=args.dry_run, show_cmd=not asr_only_output)
                    if not pre_quant_run["ok"]:
                        quant_run = {
                            "ok": False,
                            "returncode": None,
                            "stdout": "",
                            "stderr": "skipped because imatrix build failed",
                            "dry_run": args.dry_run,
                            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
                            "skipped_reason": "skipped because imatrix build failed",
                        }
                    else:
                        quant_run = run_cmd(quant_cmd, dry_run=args.dry_run, show_cmd=not asr_only_output)
                else:
                    quant_run = run_cmd(quant_cmd, dry_run=args.dry_run, show_cmd=not asr_only_output)

            if (not run_quant_utility):
                quant_lm_run = {
                    "ok": False,
                    "returncode": None,
                    "stdout": "",
                    "stderr": "",
                    "dry_run": args.dry_run,
                    "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
                    "skipped_reason": "disabled_by_mode_or_no_quantize_utility",
                }
                quant_lm_payload = None
            elif quant_run["ok"]:
                quant_lm_cmds, quant_lm_run, quant_lm_payload, quant_existing_json = run_lm_eval_split_aware(
                    args=args,
                    model_path=str(quant_model_dir),
                    output_json=quant_lm_eval_json,
                    method=method,
                    dry_run=args.dry_run,
                    show_cmd=not asr_only_output,
                )
                if quant_existing_json is not None and quant_lm_run.get("skipped_reason") == "used_existing_lm_eval_json" and not asr_only_output:
                    print(f"Using existing lm_eval file for {name}: {quant_existing_json}")
            else:
                quant_lm_run = {
                    "ok": False,
                    "returncode": None,
                    "stdout": "",
                    "stderr": "skipped because quantization failed",
                    "dry_run": args.dry_run,
                    "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
                    "skipped_reason": "skipped because quantization failed",
                }
                quant_lm_payload = None

            asr_cmd = build_asr_cmd(
                python_bin=args.python_bin,
                model_path=quant_asr_model_path,
                output_dir=asr_dir,
                scenario=args.scenario,
                num_samples=args.num_samples,
                inference_lib=resolve_quant_asr_inference_lib(args.inference_lib, method),
                gpu_memory_utilization=args.gpu_memory_utilization,
                use_chat_template=args.use_chat_template,
                prompt_format=args.prompt_format,
                hf_token=args.hf_token,
                mcd_dataset_path=args.mcd_dataset_path,
                force=args.force,
                max_new_tokens=args.max_new_tokens,
            )
            if run_asr:
                if asr_jailbreak_path.is_file() and not args.force:
                    asr_run = {
                        "ok": True,
                        "returncode": 0,
                        "stdout": "",
                        "stderr": "",
                        "dry_run": args.dry_run,
                        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
                        "skipped_reason": "used_existing_asr_jsonl",
                    }
                    asr_rows = read_jsonl(asr_jailbreak_path)
                    asr_summary = extract_asr_from_jsonl(asr_rows)
                    if not asr_only_output:
                        asr_val = asr_summary.get("asr")
                        asr_text = f"{float(asr_val):.6f}" if isinstance(asr_val, (int, float)) else "missing"
                        print(
                            f"Using cached {name} ASR={asr_text} "
                            f"(n={asr_summary.get('n_rows')}, source={asr_summary.get('source')})."
                        )
                else:
                    asr_extra_env = {}
                    if is_gguf_method(method):
                        # Keep GGUF context length explicit and stable across runs.
                        asr_extra_env["LLAMA_CPP_CTX_SIZE"] = str(args.gguf_ctx_size)
                    asr_run = run_cmd(
                        asr_cmd,
                        dry_run=args.dry_run,
                        show_cmd=not asr_only_output,
                        extra_env=asr_extra_env if asr_extra_env else None,
                    ) if quant_run["ok"] else {
                        "ok": False,
                        "returncode": None,
                        "stdout": "",
                        "stderr": "skipped because quantization failed",
                        "dry_run": args.dry_run,
                        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
                        "skipped_reason": "skipped because quantization failed",
                    }
                    asr_rows = read_jsonl(asr_jailbreak_path) if asr_run["ok"] else []
                    asr_summary = extract_asr_from_jsonl(asr_rows) if asr_run["ok"] else {
                        "asr": None,
                        "n_rows": 0,
                        "source": "skipped",
                    }
            else:
                asr_run = {
                    "ok": False,
                    "returncode": None,
                    "stdout": "",
                    "stderr": "",
                    "dry_run": args.dry_run,
                    "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
                    "skipped_reason": "disabled_by_mode",
                }
                asr_summary = {"asr": None, "n_rows": 0, "source": "disabled_by_mode"}
        quant_metrics = extract_lm_eval_metrics(quant_lm_payload, tasks) if quant_lm_payload else {}
        if quant_metrics:
            print_lm_eval_task_summary(f"[{name}]", quant_metrics)
        if is_gguf_method(method):
            try:
                quant_asr_model_path = resolve_gguf_quantized_model_file(
                    model_dir=quant_model_dir,
                    method=method,
                    args=args,
                )
            except FileNotFoundError:
                if not asr_only_output:
                    print(
                        f"Warning: missing GGUF model artifact for {name} under {quant_model_dir}; "
                        "keeping ASR entry as missing and continuing."
                    )
                quant_asr_model_path = quant_model_dir
        asr_cmd = build_asr_cmd(
            python_bin=args.python_bin,
            model_path=quant_asr_model_path,
            output_dir=asr_dir,
            scenario=args.scenario,
            num_samples=args.num_samples,
            inference_lib=resolve_quant_asr_inference_lib(args.inference_lib, method),
            gpu_memory_utilization=args.gpu_memory_utilization,
            use_chat_template=args.use_chat_template,
            prompt_format=args.prompt_format,
            hf_token=args.hf_token,
            mcd_dataset_path=args.mcd_dataset_path,
            force=args.force,
            max_new_tokens=args.max_new_tokens,
        )

        report["quantized_models"].append(
            {
                "name": name,
                "method": method,
                "bits": bits,
                "quantized_model_path": str(quant_model_dir),
                "quantization_command": quant_cmd,
                "quantization_run": {k: v for k, v in quant_run.items() if k not in {"stdout", "stderr"}},
                "utility": (
                    {
                        "lm_eval_command": quant_lm_cmds[0] if len(quant_lm_cmds) == 1 else quant_lm_cmds,
                        "lm_eval_run": {k: v for k, v in quant_lm_run.items() if k not in {"stdout", "stderr"}},
                        "metrics": quant_metrics,
                        "output_json": str(quant_existing_json or quant_lm_eval_json),
                    }
                    if (report_utility and run_quant_utility)
                    else {}
                ),
                "attack_success_rate": (
                    {
                        "calc_asr_command": asr_cmd,
                        "calc_asr_run": {k: v for k, v in asr_run.items() if k not in {"stdout", "stderr"}},
                        "scenario": args.scenario,
                        "summary": asr_summary,
                        "output_jsonl": str(asr_dir / f"asr_{args.scenario}.jsonl"),
                    }
                    if report_asr
                    else {}
                ),
            }
        )

    report_path = run_root / "evaluation_report.json"
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=True)
        f.write("\n")

    if not asr_only_output:
        print(f"report_path={report_path}")
    if report_utility:
        if baseline_is_designated:
            print(f"baseline_model_path={baseline_model_path}")
            print(f"baseline_lm_eval_json={report.get('baseline_utility', {}).get('output_json')}")
        print(
            "current_model_lm_eval_json="
            f"{report.get('current_model_utility', {}).get('output_json')}"
        )
    if report_asr:
        if baseline_is_designated:
            baseline_asr_info = report.get("baseline_attack_success_rate", {}).get("summary", {})
            baseline_asr_value = baseline_asr_info.get("asr")
            baseline_asr_n = baseline_asr_info.get("n_rows")
            baseline_asr_source = baseline_asr_info.get("source")
            baseline_asr_text = "missing"
            if isinstance(baseline_asr_value, (int, float)):
                baseline_asr_text = f"{float(baseline_asr_value):.6f}"
            print(
                f"baseline_asr={baseline_asr_text} "
                f"baseline_asr_n={baseline_asr_n} "
                f"baseline_asr_source={baseline_asr_source} "
                f"baseline_asr_jsonl={report.get('baseline_attack_success_rate', {}).get('output_jsonl')}"
            )
        current_asr_info = report.get("current_model_attack_success_rate", {}).get("summary", {})
        current_asr_value = current_asr_info.get("asr")
        current_asr_n = current_asr_info.get("n_rows")
        current_asr_source = current_asr_info.get("source")
        current_asr_text = "missing"
        if isinstance(current_asr_value, (int, float)):
            current_asr_text = f"{float(current_asr_value):.6f}"
        print(
            f"current_model_asr={current_asr_text} "
            f"current_model_asr_n={current_asr_n} "
            f"current_model_asr_source={current_asr_source} "
            "current_model_asr_jsonl="
            f"{report.get('current_model_attack_success_rate', {}).get('output_jsonl')}"
        )
    for item in report["quantized_models"]:
        asr_summary = item.get("attack_success_rate", {}).get("summary", {})
        asr_value = asr_summary.get("asr")
        asr_rows = asr_summary.get("n_rows")
        asr_source = asr_summary.get("source")
        asr_text = "missing"
        if isinstance(asr_value, (int, float)):
            asr_text = f"{float(asr_value):.6f}"
        parts = [f"quant_model={item['name']}"]
        if report_utility and item.get("utility"):
            parts.append(f"lm_eval_json={item['utility'].get('output_json')}")
        if report_asr and item.get("attack_success_rate"):
            parts.append(f"asr={asr_text}")
            parts.append(f"asr_n={asr_rows}")
            parts.append(f"asr_source={asr_source}")
            parts.append(f"asr_jsonl={item['attack_success_rate'].get('output_jsonl')}")
        print(" ".join(parts))


if __name__ == "__main__":
    main()
