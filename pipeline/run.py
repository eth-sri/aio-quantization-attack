#!/usr/bin/env python3
import argparse
import json
import os
import signal
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

STAGE_ORDER = [
    "layer_drop",
    "finetune_dual",
    "attack_ffn",
    "attack_attn",
    "finetune_dual2",
]
ACTIVE_STAGE_PROC = None  # type: Optional[subprocess.Popen]
SIGNAL_HANDLING_INSTALLED = False


def add_bool_argument(
    parser: argparse.ArgumentParser,
    name: str,
    default: Optional[bool] = None,
    help_text: Optional[str] = None,
) -> None:
    dest = name.lstrip("-").replace("-", "_")
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument(name, dest=dest, action="store_true", help=help_text)
    group.add_argument("--no-" + dest.replace("_", "-"), dest=dest, action="store_false")
    parser.set_defaults(**{dest: default})


def _terminate_process_group(proc: Optional[subprocess.Popen], sig: int = signal.SIGTERM) -> None:
    if proc is None:
        return
    if proc.poll() is not None:
        return
    try:
        pgid = os.getpgid(proc.pid)
    except ProcessLookupError:
        return
    try:
        os.killpg(pgid, sig)
    except ProcessLookupError:
        return


def _wait_with_group_cleanup(proc: subprocess.Popen, cmd: List[str]) -> None:
    try:
        rc = proc.wait()
    except KeyboardInterrupt:
        _terminate_process_group(proc, signal.SIGTERM)
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            _terminate_process_group(proc, signal.SIGKILL)
            proc.wait()
        raise
    if rc != 0:
        raise subprocess.CalledProcessError(rc, cmd)


def _install_signal_handlers() -> None:
    global SIGNAL_HANDLING_INSTALLED
    if SIGNAL_HANDLING_INSTALLED:
        return

    def _handle_stop(_sig: int, _frame) -> None:
        _terminate_process_group(ACTIVE_STAGE_PROC, signal.SIGTERM)
        raise KeyboardInterrupt

    signal.signal(signal.SIGINT, _handle_stop)
    signal.signal(signal.SIGTERM, _handle_stop)
    SIGNAL_HANDLING_INSTALLED = True


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Run pipeline: layer_drop -> finetune_dual -> attack_ffn -> "
            "attack_attn -> finetune_dual2"
        )
    )
    p.add_argument("--dataset_a", type=str, default=None)
    p.add_argument("--dataset_b", type=str, default=None)
    p.add_argument(
        "--loss_weight_b",
        type=float,
        default=None,
        help=(
            "Optional global override for loss_weight_b applied to both "
            "finetune_dual and finetune_dual2."
        ),
    )
    p.add_argument("--model_path", type=str, default=None, help="Base model path.")
    p.add_argument("--seed", type=int, default=512, help="Global seed used for all stages.")
    p.add_argument(
        "--layers",
        type=str,
        default=None,
        help="Target layers, e.g. '31,33'. Applied to finetune and attack stages.",
    )
    p.add_argument(
        "--layer_type",
        type=str,
        default=None,
        choices=["all", "attn", "ffn", "up_proj"],
        help="Global target type. ffn skips attack_attn; attn skips attack_ffn.",
    )
    add_bool_argument(
        p,
        "--simple_removal",
        default=None,
        help_text=(
            "Deprecated compatibility flag. Stage 1 uses simple_drop regardless, "
            "but this is still recorded in stage metadata."
        ),
    )
    add_bool_argument(
        p,
        "--attack_upper_left",
        default=None,
        help_text=(
            "If enabled, only apply edits/updates to upper-left blocks in layer_drop, attack, "
            "finetune_dual, and finetune_dual2."
        ),
    )
    p.add_argument(
        "--upper_left_range",
        type=int,
        default=None,
        help="Upper-left block size used when --attack_upper_left is enabled.",
    )
    p.add_argument("--output_path", type=Path, default=None)
    p.add_argument(
        "--start_from",
        type=str,
        choices=STAGE_ORDER,
        default=None,
        help=(
            "Resume pipeline from this stage and skip all earlier stages. "
            "If the selected stage is disabled by --layer_type, it will start "
            "from the next enabled stage."
        ),
    )
    p.add_argument("--config", type=Path, required=True, help="JSON config file with per-stage args.")
    p.add_argument(
        "--python_bin",
        type=str,
        default=None,
        help="Python executable used to run stage scripts.",
    )
    p.add_argument("--dry_run", action="store_true")
    p.add_argument(
        "--skip",
        type=str,
        default=None,
        help="Comma-separated stages to skip: layer_drop,finetune_dual,attack_ffn,attack_attn.",
    )
    p.add_argument(
        "--scale_factor",
        type=float,
        default=None,
        help="Global override for attack scale_factor (applied to both attack_ffn and attack_attn).",
    )
    p.add_argument(
        "--target_matrices",
        nargs="+",
        default=None,
        help=(
            "Global target_matrices override applied to attack_ffn, attack_attn, "
            "and finetune_dual2."
        ),
    )
    return p.parse_args()


def load_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        cfg = json.load(f)
    if not isinstance(cfg, dict):
        raise ValueError("Config root must be a JSON object.")
    return cfg


def normalize_stage_cfg(cfg: Dict[str, Any], key: str) -> Dict[str, Any]:
    value = cfg.get(key, {})
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"Config key '{key}' must be an object.")
    return value


def resolve_pipeline_arg(
    args: argparse.Namespace,
    cfg: Dict[str, Any],
    key: str,
) -> Any:
    cli_value = getattr(args, key, None)
    if cli_value is not None:
        return cli_value

    pipeline_cfg = cfg.get("pipeline", {})
    if isinstance(pipeline_cfg, dict) and key in pipeline_cfg and pipeline_cfg[key] is not None:
        return pipeline_cfg[key]

    raise ValueError(
        f"Missing required pipeline argument '{key}'. "
        f"Pass --{key} or set pipeline.{key} in --config."
    )


def resolve_optional_pipeline_arg(
    args: argparse.Namespace,
    cfg: Dict[str, Any],
    key: str,
    default: Any = None,
) -> Any:
    cli_value = getattr(args, key, None)
    if cli_value is not None:
        return cli_value

    pipeline_cfg = cfg.get("pipeline", {})
    if isinstance(pipeline_cfg, dict) and key in pipeline_cfg and pipeline_cfg[key] is not None:
        return pipeline_cfg[key]

    return default


def require_pipeline_value(value: Any, key: str) -> str:
    if value is None or str(value).strip() == "":
        raise ValueError(
            f"Missing required pipeline argument '{key}'. "
            f"Pass --{key} or set pipeline.{key} in --config."
        )
    return str(value)


def drop_managed_keys(params: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(params)
    out.pop("skip_mode", None)
    out.pop("enabled", None)
    out.pop("layers", None)
    out.pop("layer_type", None)
    out.pop("dtype", None)
    out.pop("seed", None)
    out.pop("bits", None)
    out.pop("quant_method", None)
    out.pop("simple_removal", None)
    out.pop("model_path", None)
    out.pop("output_path", None)
    out.pop("target_layers", None)
    out.pop("dataset_a", None)
    out.pop("dataset_b", None)
    out.pop("attack_upper_left", None)
    out.pop("upper_left_range", None)
    return out


def validate_attack_params(params: Dict[str, Any], cfg_path: str) -> Dict[str, Any]:
    out = dict(params)
    legacy_keys = ("max_select_source", "group_size", "attacked_matrix", "eval_prompt")
    legacy_in_use = [key for key in legacy_keys if key in out]
    if legacy_in_use:
        raise ValueError(
            f"Unsupported legacy attack config key(s) in '{cfg_path}': {', '.join(legacy_in_use)}. "
            "Use attack.py args directly (for example: block_size, scale_factor, "
            "target_matrices, random_positive, kill_layers, device)."
        )
    return out


def resolve_attack_stage_cfgs(cfg: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    attack_cfg = cfg.get("attack")
    if attack_cfg is None:
        raise ValueError(
            "Missing required attack config. "
            "Use an 'attack' object with required 'ffn' and 'attn' sections "
            "(optional 'common')."
        )

    if not isinstance(attack_cfg, dict):
        raise ValueError("Config key 'attack' must be an object.")
    if "ffn" not in attack_cfg:
        raise ValueError("Missing required attack config section 'attack.ffn'.")
    if "attn" not in attack_cfg:
        raise ValueError("Missing required attack config section 'attack.attn'.")

    common_cfg = validate_attack_params(
        drop_managed_keys(normalize_stage_cfg(attack_cfg, "common")),
        "attack.common",
    )
    ffn_cfg = validate_attack_params(
        drop_managed_keys(normalize_stage_cfg(attack_cfg, "ffn")),
        "attack.ffn",
    )
    attn_cfg = validate_attack_params(
        drop_managed_keys(normalize_stage_cfg(attack_cfg, "attn")),
        "attack.attn",
    )

    attack_ffn_cfg = {**common_cfg, **ffn_cfg}
    attack_attn_cfg = {**common_cfg, **attn_cfg}
    return attack_ffn_cfg, attack_attn_cfg


def to_cli_args(
    params: Dict[str, Any],
    bool_optional_keys: Optional[Set[str]] = None,
) -> List[str]:
    args = []  # type: List[str]
    bool_optional_keys = bool_optional_keys or set()
    for key, value in params.items():
        if value is None:
            continue
        flag = f"--{key}"
        if isinstance(value, bool):
            if key in bool_optional_keys:
                args.append(flag if value else f"--no-{key}")
            elif value:
                args.append(flag)
            continue
        if isinstance(value, (list, tuple)):
            if not value:
                continue
            args.append(flag)
            args.extend(str(v) for v in value)
            continue
        args.extend([flag, str(value)])
    return args


def run_stage(
    name: str,
    python_bin: str,
    script: str,
    params: Dict[str, Any],
    bool_optional_keys: Optional[Set[str]],
    dry_run: bool,
) -> None:
    global ACTIVE_STAGE_PROC
    cmd = [python_bin, script] + to_cli_args(params, bool_optional_keys=bool_optional_keys)
    print(f"\n[{name}]")
    print(" ".join(cmd))
    if dry_run:
        return
    ACTIVE_STAGE_PROC = subprocess.Popen(cmd, start_new_session=True)
    try:
        _wait_with_group_cleanup(ACTIVE_STAGE_PROC, cmd)
    finally:
        ACTIVE_STAGE_PROC = None


def write_cumulative_stage_config(
    out_dir: Path,
    *,
    pipeline_args: argparse.Namespace,
    config_path: Path,
    history: List[Dict[str, Any]],
) -> None:
    out_path = Path(out_dir) / "quantization_attack_config.json"
    payload = {
        "pipeline": {
            "dataset_a": pipeline_args.dataset_a,
            "dataset_b": pipeline_args.dataset_b,
            "loss_weight_b": pipeline_args.loss_weight_b,
            "initial_model_path": pipeline_args.model_path,
            "layers": pipeline_args.layers,
            "layer_type": pipeline_args.layer_type,
            "simple_removal": pipeline_args.simple_removal,
            "attack_upper_left": pipeline_args.attack_upper_left,
            "upper_left_range": pipeline_args.upper_left_range,
            "seed": pipeline_args.seed,
            "start_from": pipeline_args.start_from,
            "config_path": str(config_path),
        },
        "stages": history,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def record_stage_start(
    *,
    out_dir: Path,
    pipeline_args: argparse.Namespace,
    config_path: Path,
    history: List[Dict[str, Any]],
    stage: str,
    script: str,
    input_model_path: str,
    output_model_path: str,
    params: Dict[str, Any],
) -> Dict[str, Any]:
    entry: Dict[str, Any] = {
        "stage": stage,
        "script": script,
        "input_model_path": input_model_path,
        "output_model_path": output_model_path,
        "params": params,
        "status": "running",
    }
    history.append(entry)
    write_cumulative_stage_config(
        out_dir,
        pipeline_args=pipeline_args,
        config_path=config_path,
        history=history,
    )
    return entry


def record_stage_complete(
    *,
    out_dir: Path,
    pipeline_args: argparse.Namespace,
    config_path: Path,
    history: List[Dict[str, Any]],
    entry: Dict[str, Any],
) -> None:
    entry["status"] = "completed"
    write_cumulative_stage_config(
        out_dir,
        pipeline_args=pipeline_args,
        config_path=config_path,
        history=history,
    )


def stage_enabled(stage: str, layer_type: str) -> bool:
    if stage == "attack_ffn" and layer_type == "attn":
        return False
    if stage == "attack_attn" and layer_type in {"ffn", "up_proj"}:
        return False
    return True


def parse_skip_stages(raw: str | None) -> Set[str]:
    if raw is None:
        return set()
    stages = {part.strip().lower() for part in raw.split(",") if part.strip()}
    invalid = sorted(stages - {"layer_drop", "finetune_dual", "attack_ffn", "attack_attn"})
    if invalid:
        raise ValueError("--skip supports only layer_drop, finetune_dual, attack_ffn, and attack_attn")
    return stages


def same_model_path(a: str, b: str) -> bool:
    try:
        return Path(a).expanduser().resolve() == Path(b).expanduser().resolve()
    except Exception:
        return os.path.abspath(os.path.expanduser(a)) == os.path.abspath(os.path.expanduser(b))


def model_artifacts_exist(path: Path) -> bool:
    if not path.exists() or not path.is_dir():
        return False
    if (path / "config.json").exists():
        return True
    for pattern in ("*.safetensors", "*.bin", "*.pt"):
        if any(path.glob(pattern)):
            return True
    return False


def load_existing_stage_history(model_dir: Path) -> List[Dict[str, Any]]:
    cfg_path = Path(model_dir) / "quantization_attack_config.json"
    if not cfg_path.is_file():
        return []
    try:
        with cfg_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
    except Exception:
        return []
    stages = payload.get("stages", [])
    if not isinstance(stages, list):
        return []
    return [row for row in stages if isinstance(row, dict)]


def main() -> None:
    _install_signal_handlers()
    args = parse_args()
    cfg = load_config(args.config)

    args.python_bin = str(resolve_optional_pipeline_arg(args, cfg, "python_bin", default=sys.executable))
    args.model_path = str(resolve_pipeline_arg(args, cfg, "model_path"))
    args.layers = str(resolve_pipeline_arg(args, cfg, "layers"))
    args.layer_type = str(resolve_pipeline_arg(args, cfg, "layer_type"))
    args.simple_removal = resolve_optional_pipeline_arg(args, cfg, "simple_removal", default=True)
    args.attack_upper_left = resolve_optional_pipeline_arg(args, cfg, "attack_upper_left", default=False)
    args.upper_left_range = int(resolve_optional_pipeline_arg(args, cfg, "upper_left_range", default=512))
    args.output_path = Path(resolve_pipeline_arg(args, cfg, "output_path"))
    args.dataset_a = resolve_optional_pipeline_arg(args, cfg, "dataset_a", default=None)
    args.dataset_b = resolve_optional_pipeline_arg(args, cfg, "dataset_b", default=None)
    args.loss_weight_b = resolve_optional_pipeline_arg(args, cfg, "loss_weight_b", default=None)
    args.start_from = str(resolve_optional_pipeline_arg(args, cfg, "start_from", default="layer_drop"))

    if args.layer_type not in {"all", "attn", "ffn", "up_proj"}:
        raise ValueError("layer_type must be one of: all, attn, ffn, up_proj")
    if args.upper_left_range <= 0:
        raise ValueError("upper_left_range must be >= 1")
    if args.start_from not in STAGE_ORDER:
        raise ValueError(f"start_from must be one of: {', '.join(STAGE_ORDER)}")
    skip_stages = parse_skip_stages(args.skip)

    args.output_path.mkdir(parents=True, exist_ok=True)

    stage_dirs = {
        "layer_drop": args.output_path / "01_layer_drop",
        "finetune_dual": args.output_path / "02_finetune_dual",
        "attack_ffn": args.output_path / "03_attack_ffn",
        "attack_attn": args.output_path / "04_attack_attn",
        "finetune_dual2": args.output_path / "05_finetune_dual2",
    }
    for stage_dir in stage_dirs.values():
        stage_dir.mkdir(parents=True, exist_ok=True)

    layer_drop_cfg = drop_managed_keys(normalize_stage_cfg(cfg, "layer_drop"))
    if not layer_drop_cfg:
        layer_drop_cfg = drop_managed_keys(normalize_stage_cfg(cfg, "laco"))
    finetune_dual_cfg = drop_managed_keys(normalize_stage_cfg(cfg, "finetune_dual"))
    attack_ffn_cfg, attack_attn_cfg = resolve_attack_stage_cfgs(cfg)
    if args.scale_factor is not None:
        attack_ffn_cfg["scale_factor"] = args.scale_factor
        attack_attn_cfg["scale_factor"] = args.scale_factor
    finetune_dual2_cfg = drop_managed_keys(normalize_stage_cfg(cfg, "finetune_dual2"))
    if args.target_matrices is not None:
        attack_ffn_cfg["target_matrices"] = args.target_matrices
        attack_attn_cfg["target_matrices"] = args.target_matrices
        finetune_dual2_cfg["target_matrices"] = args.target_matrices

    requested_start_index = STAGE_ORDER.index(args.start_from)
    start_index = requested_start_index
    while start_index < len(STAGE_ORDER) and not stage_enabled(STAGE_ORDER[start_index], args.layer_type):
        start_index += 1
    if start_index >= len(STAGE_ORDER):
        raise ValueError(
            f"Requested start_from='{args.start_from}' is not runnable under layer_type='{args.layer_type}'."
        )
    effective_start = STAGE_ORDER[start_index]
    if effective_start != args.start_from:
        print(
            f"[resume] start_from '{args.start_from}' is disabled by layer_type='{args.layer_type}'. "
            f"Using '{effective_start}' instead."
        )

    for skipped_stage in STAGE_ORDER[:start_index]:
        print(f"\n[{skipped_stage}]")
        print(f"skipped because --start_from {effective_start}")

    if start_index == 0:
        current_model = str(Path(args.model_path))
    else:
        prev_index = start_index - 1
        while prev_index >= 0 and not stage_enabled(STAGE_ORDER[prev_index], args.layer_type):
            prev_index -= 1
        if prev_index < 0:
            current_model = str(Path(args.model_path))
        else:
            current_model = str(stage_dirs[STAGE_ORDER[prev_index]])
        if not args.dry_run and not model_artifacts_exist(Path(current_model)):
            raise FileNotFoundError(
                f"Cannot resume from stage '{effective_start}': "
                f"expected prior stage output at '{current_model}' with model artifacts."
            )

    stage_history = []  # type: List[Dict[str, Any]]
    if start_index > 0:
        stage_history = load_existing_stage_history(current_model)

    if requested_start_index <= 0 and effective_start == "layer_drop" and "layer_drop" not in skip_stages:
        stage0_params = dict(layer_drop_cfg)
        stage0_params.update(
            {
                "model_path": current_model,
                "target_layers": args.layers,
                "layer_type": args.layer_type,
                "seed": args.seed,
                "output_path": str(stage_dirs["layer_drop"]),
                "attack_upper_left": args.attack_upper_left,
                "upper_left_range": args.upper_left_range,
            }
        )
        stage0_entry = None
        if not args.dry_run:
            stage0_entry = record_stage_start(
                out_dir=stage_dirs["layer_drop"],
                pipeline_args=args,
                config_path=args.config,
                history=stage_history,
                stage="layer_drop",
                script="Pruning/simple_drop.py",
                input_model_path=current_model,
                output_model_path=str(stage_dirs["layer_drop"]),
                params=stage0_params,
            )
        run_stage(
            name="layer_drop",
            python_bin=args.python_bin,
            script="Pruning/simple_drop.py",
            params=stage0_params,
            bool_optional_keys={"attack_upper_left"},
            dry_run=args.dry_run,
        )
        if not args.dry_run and stage0_entry is not None:
            record_stage_complete(
                out_dir=stage_dirs["layer_drop"],
                pipeline_args=args,
                config_path=args.config,
                history=stage_history,
                entry=stage0_entry,
            )
        current_model = str(stage_dirs["layer_drop"])
    elif requested_start_index <= 0 and effective_start == "layer_drop":
        print("\n[layer_drop]")
        print("skipped because --skip layer_drop")

    if requested_start_index <= 1 and start_index <= 1 and "finetune_dual" not in skip_stages:
        stage1_params = dict(finetune_dual_cfg)
        stage1_only_dataset_a = bool(stage1_params.get("only_dataset_a", False))
        args.dataset_a = require_pipeline_value(args.dataset_a, "dataset_a")
        if not stage1_only_dataset_a:
            args.dataset_b = require_pipeline_value(args.dataset_b, "dataset_b")
        if args.loss_weight_b is not None:
            stage1_params["loss_weight_b"] = args.loss_weight_b
        stage1_params.update(
            {
                "model_path": current_model,
                "dataset_a": args.dataset_a,
                "output_path": str(stage_dirs["finetune_dual"]),
                "layers": args.layers,
                "layer_type": args.layer_type,
                "seed": args.seed,
                "attack_upper_left": args.attack_upper_left,
                "upper_left_range": args.upper_left_range,
            }
        )
        if not stage1_only_dataset_a:
            stage1_params["dataset_b"] = args.dataset_b
        else:
            stage1_params.pop("dataset_b", None)
        stage1_entry = None
        if not args.dry_run:
            stage1_entry = record_stage_start(
                out_dir=stage_dirs["finetune_dual"],
                pipeline_args=args,
                config_path=args.config,
                history=stage_history,
                stage="finetune_dual",
                script="Finetune/finetune_dual.py",
                input_model_path=current_model,
                output_model_path=str(stage_dirs["finetune_dual"]),
                params=stage1_params,
            )
        run_stage(
            name="finetune_dual",
            python_bin=args.python_bin,
            script="Finetune/finetune_dual.py",
            params=stage1_params,
            bool_optional_keys={
                "kl_on_inputs",
                "gradient_checkpointing",
                "gradient_checkpointing_use_reentrant",
                "attack_upper_left",
            },
            dry_run=args.dry_run,
        )
        if not args.dry_run and stage1_entry is not None:
            record_stage_complete(
                out_dir=stage_dirs["finetune_dual"],
                pipeline_args=args,
                config_path=args.config,
                history=stage_history,
                entry=stage1_entry,
            )
        current_model = str(stage_dirs["finetune_dual"])
    elif requested_start_index <= 1 and start_index <= 1:
        print("\n[finetune_dual]")
        print("skipped because --skip finetune_dual")

    if (
        requested_start_index <= 2
        and start_index <= 2
        and args.layer_type != "attn"
        and "attack_ffn" not in skip_stages
    ):
        stage2_params = dict(attack_ffn_cfg)
        stage2_params.update(
            {
                "model_path": current_model,
                "output_path": str(stage_dirs["attack_ffn"]),
                "layers": args.layers,
                "seed": args.seed,
                "attack_upper_left": args.attack_upper_left,
                "upper_left_range": args.upper_left_range,
            }
        )
        stage2_entry = None
        if not args.dry_run:
            stage2_entry = record_stage_start(
                out_dir=stage_dirs["attack_ffn"],
                pipeline_args=args,
                config_path=args.config,
                history=stage_history,
                stage="attack_ffn",
                script="Attack/attack.py",
                input_model_path=current_model,
                output_model_path=str(stage_dirs["attack_ffn"]),
                params=stage2_params,
            )
        run_stage(
            name="attack_ffn",
            python_bin=args.python_bin,
            script="Attack/attack.py",
            params=stage2_params,
            bool_optional_keys={"random_positive", "attack_bias", "attack_upper_left"},
            dry_run=args.dry_run,
        )
        if not args.dry_run and stage2_entry is not None:
            record_stage_complete(
                out_dir=stage_dirs["attack_ffn"],
                pipeline_args=args,
                config_path=args.config,
                history=stage_history,
                entry=stage2_entry,
            )
        current_model = str(stage_dirs["attack_ffn"])
    elif requested_start_index <= 2 and start_index <= 2:
        print("\n[attack_ffn]")
        print("skipped because --skip attack_ffn" if "attack_ffn" in skip_stages else "skipped because --layer_type attn")

    if (
        requested_start_index <= 3
        and start_index <= 3
        and args.layer_type not in {"ffn", "up_proj"}
        and "attack_attn" not in skip_stages
    ):
        stage3_params = dict(attack_attn_cfg)
        stage3_params.update(
            {
                "model_path": current_model,
                "output_path": str(stage_dirs["attack_attn"]),
                "layers": args.layers,
                "seed": args.seed,
                "attack_upper_left": args.attack_upper_left,
                "upper_left_range": args.upper_left_range,
            }
        )
        stage3_entry = None
        if not args.dry_run:
            stage3_entry = record_stage_start(
                out_dir=stage_dirs["attack_attn"],
                pipeline_args=args,
                config_path=args.config,
                history=stage_history,
                stage="attack_attn",
                script="Attack/attack.py",
                input_model_path=current_model,
                output_model_path=str(stage_dirs["attack_attn"]),
                params=stage3_params,
            )
        run_stage(
            name="attack_attn",
            python_bin=args.python_bin,
            script="Attack/attack.py",
            params=stage3_params,
            bool_optional_keys={"random_positive", "attack_bias", "attack_upper_left"},
            dry_run=args.dry_run,
        )
        if not args.dry_run and stage3_entry is not None:
            record_stage_complete(
                out_dir=stage_dirs["attack_attn"],
                pipeline_args=args,
                config_path=args.config,
                history=stage_history,
                entry=stage3_entry,
            )
        current_model = str(stage_dirs["attack_attn"])
    elif requested_start_index <= 3 and start_index <= 3:
        print("\n[attack_attn]")
        print("skipped because --skip attack_attn" if "attack_attn" in skip_stages else "skipped because --layer_type ffn")

    if requested_start_index <= 4 and start_index <= 4:
        args.dataset_a = require_pipeline_value(args.dataset_a, "dataset_a")
        args.dataset_b = require_pipeline_value(args.dataset_b, "dataset_b")
        stage4_params = dict(finetune_dual2_cfg)
        if args.loss_weight_b is not None:
            stage4_params["loss_weight_b"] = args.loss_weight_b
        stage4_params.update(
            {
                "model_path": current_model,
                "dataset_a": args.dataset_a,
                "dataset_b": args.dataset_b,
                "output_path": str(stage_dirs["finetune_dual2"]),
                "layers": args.layers,
                "layer_type": args.layer_type,
                "seed": args.seed,
                "attack_upper_left": args.attack_upper_left,
                "upper_left_range": args.upper_left_range,
            }
        )
        attack_stages_in_run = []
        if args.layer_type != "attn":
            attack_stages_in_run.append("attack_ffn")
        if args.layer_type not in {"ffn", "up_proj"}:
            attack_stages_in_run.append("attack_attn")
        stage4_script = "Finetune/finetune_dual2.py"
        if attack_stages_in_run and all(stage in skip_stages for stage in attack_stages_in_run):
            stage4_script = "Finetune/finetune_dual2_2.py"
            swap_snapshot = stage4_params.get("swap_snapshot")
            if same_model_path(str(swap_snapshot), current_model):
                raise ValueError("finetune_dual2.swap_snapshot must differ from the current model.")
        stage4_entry = None
        if not args.dry_run:
            stage4_entry = record_stage_start(
                out_dir=stage_dirs["finetune_dual2"],
                pipeline_args=args,
                config_path=args.config,
                history=stage_history,
                stage="finetune_dual2",
                script=stage4_script,
                input_model_path=current_model,
                output_model_path=str(stage_dirs["finetune_dual2"]),
                params=stage4_params,
            )
        run_stage(
            name="finetune_dual2",
            python_bin=args.python_bin,
            script=stage4_script,
            params=stage4_params,
            bool_optional_keys={
                "kl_on_inputs",
                "kill_except_largest",
                "dataloader_pin_memory",
                "attack_upper_left",
            },
            dry_run=args.dry_run,
        )
        if not args.dry_run and stage4_entry is not None:
            record_stage_complete(
                out_dir=stage_dirs["finetune_dual2"],
                pipeline_args=args,
                config_path=args.config,
                history=stage_history,
                entry=stage4_entry,
            )
        current_model = str(stage_dirs["finetune_dual2"])

    print("\n[pipeline_done]")
    print(f"final_model_path={current_model}")


if __name__ == "__main__":
    main()
