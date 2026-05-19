#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import inspect
import random
import shutil
from collections import defaultdict, deque
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from datasets import Dataset, load_dataset
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    set_seed,
)

def parse_layer_indices(raw: str) -> list[int]:
    values = []
    for token in raw.replace(",", " ").split():
        token = token.strip()
        if token:
            values.append(int(token))
    if not values:
        raise ValueError("No target layers were provided.")
    return sorted(set(values))


PROJECTION_NAME_TO_PATH = {
    "q_proj": ("self_attn", "q_proj"),
    "k_proj": ("self_attn", "k_proj"),
    "v_proj": ("self_attn", "v_proj"),
    "qkv_proj": ("self_attn", "qkv_proj"),
    "o_proj": ("self_attn", "o_proj"),
    "gate_proj": ("mlp", "gate_proj"),
    "up_proj": ("mlp", "up_proj"),
    "gate_up_proj": ("mlp", "gate_up_proj"),
    "down_proj": ("mlp", "down_proj"),
}


def parse_target_matrices(values: list[str]) -> list[tuple[str, str]]:
    raw_items: list[str] = []
    for value in values:
        raw_items.extend(part.strip() for part in value.split(",") if part.strip())
    if not raw_items:
        raise ValueError("No target matrices were provided.")

    out: list[tuple[str, str]] = []
    for item in raw_items:
        if "." in item:
            parent, name = item.split(".", 1)
            target = (parent.strip(), name.strip())
        else:
            if item not in PROJECTION_NAME_TO_PATH:
                raise ValueError(
                    "Unsupported target matrix. Use one of: "
                    f"{', '.join(sorted(PROJECTION_NAME_TO_PATH))} "
                    "or full paths like self_attn.v_proj / mlp.up_proj."
                )
            target = PROJECTION_NAME_TO_PATH[item]
        if target not in out:
            out.append(target)
    return out


def validate_matrix_targets_for_layer_type(
    target_matrices: list[tuple[str, str]],
    layer_type: str,
) -> None:
    if layer_type == "all":
        return
    for parent_name, proj_name in target_matrices:
        if layer_type == "attn" and parent_name != "self_attn":
            raise ValueError(
                f"target matrix {parent_name}.{proj_name} is incompatible with --layer_type attn"
            )
        if layer_type == "ffn" and parent_name != "mlp":
            raise ValueError(
                f"target matrix {parent_name}.{proj_name} is incompatible with --layer_type ffn"
            )


def resolve_target_module_for_arch(parent, parent_name: str, proj_name: str):
    module = getattr(parent, proj_name, None)
    if module is not None:
        return module
    # Phi-family compatibility:
    # - MLP often uses gate_up_proj instead of separate gate/up projections.
    if parent_name == "mlp" and proj_name in {"up_proj", "gate_proj"}:
        return getattr(parent, "gate_up_proj", None)
    # - Attention can use fused qkv projection.
    if parent_name == "self_attn" and proj_name in {"q_proj", "k_proj", "v_proj"}:
        return getattr(parent, "qkv_proj", None)
    return None


def build_prompt(instruction: str, input_text: str) -> str:
    instruction = instruction.strip()
    input_text = input_text.strip()
    if input_text:
        return f"{instruction}\n\n{input_text}"
    return instruction


def build_plain_sft_prompt(prompt: str) -> str:
    prompt = prompt.strip()
    return f"### Instruction:\n{prompt}\n\n### Response:\n"


def normalize_prompt_output(example: dict[str, Any]) -> tuple[str, str]:
    if "prompt" in example and "output" in example:
        return str(example["prompt"]).strip(), str(example["output"]).strip()
    if "instruction" in example and "output" in example:
        return build_prompt(
            str(example["instruction"]),
            str(example.get("input", "")),
        ), str(example["output"]).strip()
    raise ValueError(
        "Expected either {prompt, output} or {instruction, [input], output} fields."
    )


def resolve_prompt_format(
    requested_prompt_format: str,
    model_path: str,
    tokenizer,
) -> str:
    _ = model_path  # kept for backward-compatible signature
    if requested_prompt_format != "auto":
        return requested_prompt_format
    has_chat_template = bool(getattr(tokenizer, "chat_template", None))
    if has_chat_template:
        return "instruct"
    return "plain"


def load_supervised_dataset(dataset_spec: str, split: str) -> Dataset:
    path = Path(dataset_spec)
    if path.exists():
        return load_dataset("json", data_files=str(path), split="train")
    return load_dataset(dataset_spec, split=split)


def pair_a_b_by_prompt(dataset_a: Dataset, dataset_b: Dataset) -> list[dict[str, str]]:
    b_outputs_by_prompt: dict[str, deque[str]] = defaultdict(deque)
    for row in dataset_b:
        prompt, output = normalize_prompt_output(row)
        b_outputs_by_prompt[prompt].append(output)

    paired = []
    for row in dataset_a:
        prompt, output_a = normalize_prompt_output(row)
        if not b_outputs_by_prompt[prompt]:
            raise ValueError(
                "Could not pair all A examples with B examples by prompt. "
                f"Missing prompt: {prompt[:120]!r}"
            )
        output_b = b_outputs_by_prompt[prompt].popleft()
        paired.append({"prompt": prompt, "output_a": output_a, "output_b": output_b})

    leftovers = sum(len(v) for v in b_outputs_by_prompt.values())
    if leftovers > 0:
        print(f"warning=dataset_b_has_{leftovers}_unmatched_rows")

    return paired


def _to_input_id_list(tokenized: Any) -> list[int]:
    if isinstance(tokenized, dict):
        if "input_ids" in tokenized:
            tokenized = tokenized["input_ids"]
        elif "ids" in tokenized:
            tokenized = tokenized["ids"]
    elif hasattr(tokenized, "input_ids"):
        tokenized = tokenized.input_ids
    elif hasattr(tokenized, "ids"):
        tokenized = tokenized.ids

    if torch.is_tensor(tokenized):
        tokenized = tokenized.tolist()
    if isinstance(tokenized, tuple):
        tokenized = list(tokenized)

    # Handle list[Encoding] and batch-style nested lists.
    if isinstance(tokenized, list) and tokenized:
        first = tokenized[0]
        if hasattr(first, "ids"):
            if len(tokenized) != 1:
                raise TypeError(f"Expected single Encoding item, got {len(tokenized)}")
            tokenized = first.ids
        elif isinstance(first, (list, tuple)):
            if len(tokenized) != 1:
                raise TypeError(f"Expected single input_ids sequence, got {len(tokenized)}")
            tokenized = list(first)

    if not isinstance(tokenized, list):
        raise TypeError(f"Could not normalize tokenized output to list[int], got {type(tokenized)}")
    return [int(x) for x in tokenized]


def _common_prefix_len(a: list[int], b: list[int]) -> int:
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


def _build_instruct_messages(prompt: str, system_message: str) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []
    if system_message.strip():
        messages.append({"role": "system", "content": system_message.strip()})
    messages.append({"role": "user", "content": prompt})
    return messages


def _render_instruct_prompt_ids(tokenizer, prompt: str, system_message: str) -> list[int]:
    messages = _build_instruct_messages(prompt, system_message)
    return _to_input_id_list(
        tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
        )
    )


def _truncate_instruct_prompt_to_fit(
    tokenizer,
    prompt: str,
    system_message: str,
    max_prompt_len: int,
    fallback_prompt_ids: list[int],
) -> list[int]:
    if len(fallback_prompt_ids) <= max_prompt_len:
        return fallback_prompt_ids
    if max_prompt_len <= 0:
        return []

    # Keep chat-template structure intact by truncating user-content tokens first,
    # then re-rendering through apply_chat_template.
    user_ids = _to_input_id_list(tokenizer(prompt, add_special_tokens=False))
    empty_user_prompt_ids = _render_instruct_prompt_ids(
        tokenizer=tokenizer,
        prompt="",
        system_message=system_message,
    )
    user_budget = max_prompt_len - len(empty_user_prompt_ids)
    if user_budget < 0:
        user_budget = 0

    truncated_user_ids = user_ids[-user_budget:] if user_budget > 0 else []
    truncated_prompt = tokenizer.decode(
        truncated_user_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    truncated_prompt_ids = _render_instruct_prompt_ids(
        tokenizer=tokenizer,
        prompt=truncated_prompt,
        system_message=system_message,
    )
    if len(truncated_prompt_ids) <= max_prompt_len:
        return truncated_prompt_ids

    # Fallback: still guarantee fit while trying to preserve assistant prefill tail.
    return truncated_prompt_ids[-max_prompt_len:]


def tokenize_prompt_and_output(
    tokenizer,
    prompt: str,
    output: str,
    max_length: int,
    prompt_format: str,
    system_message: str,
) -> tuple[list[int], list[int], list[int]]:
    if max_length < 2:
        raise ValueError("max_length must be >= 2 for prompt+output training.")

    if prompt_format == "instruct":
        if not hasattr(tokenizer, "apply_chat_template") or not getattr(tokenizer, "chat_template", None):
            raise ValueError(
                "prompt_format=instruct requires a tokenizer with chat_template/apply_chat_template."
            )
        messages = _build_instruct_messages(prompt, system_message)
        prompt_ids = _to_input_id_list(
            tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
            )
        )
        full_ids = _to_input_id_list(
            tokenizer.apply_chat_template(
                messages + [{"role": "assistant", "content": output}],
                tokenize=True,
                add_generation_prompt=False,
            )
        )
        if len(full_ids) > len(prompt_ids) and full_ids[: len(prompt_ids)] == prompt_ids:
            split_idx = len(prompt_ids)
        else:
            split_idx = _common_prefix_len(prompt_ids, full_ids)
            if split_idx <= 0 or split_idx >= len(full_ids):
                raise ValueError(
                    "Chat template boundary mismatch: could not infer a valid assistant boundary "
                    f"(len_prompt_ids={len(prompt_ids)}, len_full_ids={len(full_ids)}, "
                    f"common_prefix={split_idx})."
                )
        output_ids = full_ids[split_idx:]

        # Hard truncate output first, then prompt, to guarantee fit.
        if len(output_ids) > (max_length - 1):
            output_ids = output_ids[: max_length - 1]
        max_prompt_len = max_length - len(output_ids)
        prompt_ids = _truncate_instruct_prompt_to_fit(
            tokenizer=tokenizer,
            prompt=prompt,
            system_message=system_message,
            max_prompt_len=max_prompt_len,
            fallback_prompt_ids=prompt_ids,
        )
    else:
        plain_prompt = build_plain_sft_prompt(prompt)
        prompt_ids = _to_input_id_list(tokenizer(plain_prompt, add_special_tokens=True))
        output_ids = _to_input_id_list(tokenizer(f" {output}", add_special_tokens=False))
        if tokenizer.eos_token_id is not None:
            output_ids = output_ids + [tokenizer.eos_token_id]

        # Hard truncate output first, then prompt, to guarantee fit.
        if len(output_ids) > (max_length - 1):
            output_ids = output_ids[: max_length - 1]
        max_prompt_len = max_length - len(output_ids)
        prompt_ids = prompt_ids[:max_prompt_len]
    input_ids = prompt_ids + output_ids
    labels = ([-100] * len(prompt_ids)) + output_ids
    attention_mask = [1] * len(input_ids)
    return input_ids, attention_mask, labels


def tokenize_paired_example(
    example: dict[str, str],
    tokenizer,
    max_length: int,
    prompt_format: str,
    system_message: str,
) -> dict[str, list[int]]:
    prompt_shared = example["prompt"] if "prompt" in example else None
    prompt_a = example["prompt_a"] if "prompt_a" in example else prompt_shared
    prompt_b = example["prompt_b"] if "prompt_b" in example else prompt_shared
    if prompt_a is None or prompt_b is None:
        raise KeyError("Expected prompt_a/prompt_b or shared prompt in paired example.")
    a_input_ids, a_attention_mask, a_labels = tokenize_prompt_and_output(
        tokenizer,
        prompt_a,
        example["output_a"],
        max_length,
        prompt_format,
        system_message,
    )
    b_input_ids, b_attention_mask, b_labels = tokenize_prompt_and_output(
        tokenizer,
        prompt_b,
        example["output_b"],
        max_length,
        prompt_format,
        system_message,
    )
    return {
        "a_input_ids": a_input_ids,
        "a_attention_mask": a_attention_mask,
        "a_labels": a_labels,
        "b_input_ids": b_input_ids,
        "b_attention_mask": b_attention_mask,
        "b_labels": b_labels,
    }


def tokenize_single_example(
    example: dict[str, Any],
    tokenizer,
    max_length: int,
    prompt_format: str,
    system_message: str,
) -> dict[str, list[int]]:
    prompt, output = normalize_prompt_output(example)
    input_ids, attention_mask, labels = tokenize_prompt_and_output(
        tokenizer=tokenizer,
        prompt=prompt,
        output=output,
        max_length=max_length,
        prompt_format=prompt_format,
        system_message=system_message,
    )
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


@torch.no_grad()
def keep_group_max_only_tensor_copy(tensor: torch.Tensor, block_size: int) -> torch.Tensor:
    if block_size <= 0:
        raise ValueError("--block_size must be > 0")

    out = tensor.detach().clone()
    if out.ndim == 0:
        return out
    if out.ndim == 1:
        view = out.view(1, -1)
    else:
        view = out.view(-1, out.shape[-1])

    for row in view:
        cols = row.numel()
        for start in range(0, cols, block_size):
            end = min(start + block_size, cols)
            chunk = row[start:end]
            if chunk.numel() <= 1:
                continue
            rel_idx = int(torch.argmax(chunk.abs()).item())
            mask = torch.ones_like(chunk, dtype=torch.bool)
            mask[rel_idx] = False
            chunk[mask] = 0
    return out


@torch.no_grad()
def keep_group_max_only_upper_left_tensor_copy(
    tensor: torch.Tensor,
    block_size: int,
    upper_left_range: int,
) -> torch.Tensor:
    if upper_left_range <= 0:
        raise ValueError("--upper_left_range must be >= 1")
    out = tensor.detach().clone()
    if out.ndim == 0:
        return out
    if out.ndim == 1:
        n = min(int(out.shape[0]), upper_left_range)
        if n > 0:
            out[:n] = keep_group_max_only_tensor_copy(out[:n], block_size=block_size)
        return out
    n_rows = min(int(out.shape[0]), upper_left_range)
    n_cols = min(int(out.shape[1]), upper_left_range)
    if n_rows > 0 and n_cols > 0:
        out[:n_rows, :n_cols] = keep_group_max_only_tensor_copy(
            out[:n_rows, :n_cols],
            block_size=block_size,
        )
    return out


class DualTargetCollator:
    def __init__(self, tokenizer, pad_to_multiple_of: int | None = 8):
        self.tokenizer = tokenizer
        self.pad_to_multiple_of = pad_to_multiple_of

    def _pad_side(self, features: list[dict[str, list[int]]], prefix: str) -> dict[str, torch.Tensor]:
        model_inputs = [
            {
                "input_ids": feature[f"{prefix}_input_ids"],
                "attention_mask": feature[f"{prefix}_attention_mask"],
            }
            for feature in features
        ]
        batch = self.tokenizer.pad(
            model_inputs,
            padding=True,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )
        max_len = batch["input_ids"].shape[1]
        labels = torch.full((len(features), max_len), -100, dtype=torch.long)
        for i, feature in enumerate(features):
            curr = torch.tensor(feature[f"{prefix}_labels"], dtype=torch.long)
            labels[i, : len(curr)] = curr
        batch["labels"] = labels
        batch["token_type_ids"] = torch.zeros_like(batch["input_ids"], dtype=torch.long)
        return batch

    def __call__(self, features: list[dict[str, list[int]]]) -> dict[str, torch.Tensor]:
        a_batch = self._pad_side(features, "a")
        b_batch = self._pad_side(features, "b")
        return {
            "a_input_ids": a_batch["input_ids"],
            "a_attention_mask": a_batch["attention_mask"],
            "a_token_type_ids": a_batch["token_type_ids"],
            "a_labels": a_batch["labels"],
            "b_input_ids": b_batch["input_ids"],
            "b_attention_mask": b_batch["attention_mask"],
            "b_token_type_ids": b_batch["token_type_ids"],
            "b_labels": b_batch["labels"],
        }


class SingleTargetCollator:
    def __init__(self, tokenizer, pad_to_multiple_of: int | None = 8):
        self.tokenizer = tokenizer
        self.pad_to_multiple_of = pad_to_multiple_of

    def __call__(self, features: list[dict[str, list[int]]]) -> dict[str, torch.Tensor]:
        model_inputs = [
            {
                "input_ids": feature["input_ids"],
                "attention_mask": feature["attention_mask"],
            }
            for feature in features
        ]
        batch = self.tokenizer.pad(
            model_inputs,
            padding=True,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )
        max_len = batch["input_ids"].shape[1]
        labels = torch.full((len(features), max_len), -100, dtype=torch.long)
        for i, feature in enumerate(features):
            curr = torch.tensor(feature["labels"], dtype=torch.long)
            labels[i, : len(curr)] = curr
        batch["labels"] = labels
        batch["token_type_ids"] = torch.zeros_like(batch["input_ids"], dtype=torch.long)
        return batch


def save_precomputed_reference_shards(
    *,
    reference_model,
    dataloader: DataLoader,
    output_dir: Path,
) -> list[str]:
    def _atomic_save_shard_with_fallback(shard_obj: dict[str, torch.Tensor], shard_path: Path) -> None:
        last_exc: Exception | None = None
        attempts = [
            {"_use_new_zipfile_serialization": True},
            {"_use_new_zipfile_serialization": False},
        ]
        for _ in range(2):
            for kwargs in attempts:
                tmp_path = output_dir / f".{shard_path.name}.tmp"
                try:
                    torch.save(shard_obj, tmp_path, **kwargs)
                    tmp_path.replace(shard_path)
                    return
                except Exception as exc:  # noqa: BLE001 - we fallback across serializers/retries
                    last_exc = exc
                    try:
                        if tmp_path.exists():
                            tmp_path.unlink()
                    except OSError:
                        pass
        if last_exc is None:
            raise RuntimeError(f"unknown_error_while_saving_precomputed_shard path={shard_path}")
        raise RuntimeError(
            f"failed_to_save_precomputed_shard path={shard_path} error={type(last_exc).__name__}: {last_exc}"
        ) from last_exc

    output_dir.mkdir(parents=True, exist_ok=True)
    shard_paths: list[str] = []
    failed_shards = 0
    precompute_device = next(reference_model.parameters()).device
    total_batches = len(dataloader) if hasattr(dataloader, "__len__") else None
    for batch_idx, batch in enumerate(
        tqdm(dataloader, total=total_batches, desc="Precomputing reference shards")
    ):
        input_ids = batch["input_ids"].to(precompute_device)
        attention_mask = batch["attention_mask"].to(precompute_device)
        token_type_ids = batch.get("token_type_ids")
        if token_type_ids is not None:
            token_type_ids = token_type_ids.to(precompute_device)
        with torch.no_grad():
            ref_logits = reference_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
            ).logits.detach()
            ref_log_probs = torch.log_softmax(ref_logits.float(), dim=-1).to(torch.float16)

        shard = {
            "input_ids": batch["input_ids"].cpu(),
            "attention_mask": batch["attention_mask"].cpu(),
            "token_type_ids": batch["token_type_ids"].cpu(),
            "labels": batch["labels"].cpu(),
            "ref_log_probs": ref_log_probs.cpu(),
        }
        shard_path = output_dir / f"ref_batch_{batch_idx:06d}.pt"
        try:
            _atomic_save_shard_with_fallback(shard, shard_path)
            shard_paths.append(str(shard_path))
        except RuntimeError as exc:
            failed_shards += 1
            print(
                "warning=skipping_precomputed_ref_shard_due_to_save_error "
                f"batch_idx={batch_idx} path={shard_path} error={exc}"
            )
            continue
    if not shard_paths and failed_shards > 0:
        raise RuntimeError("failed_to_write_any_precomputed_reference_shards")
    return shard_paths


def try_load_precomputed_reference_shard(shard_path: str | Path) -> dict[str, torch.Tensor] | None:
    try:
        batch = torch.load(str(shard_path), map_location="cpu")
    except (RuntimeError, EOFError, OSError, ValueError) as exc:
        print(f"warning=skipping_bad_precomputed_ref_shard path={shard_path} reason={type(exc).__name__}")
        return None

    if not isinstance(batch, dict):
        print(f"warning=skipping_bad_precomputed_ref_shard path={shard_path} reason=not_dict")
        return None

    required = ("input_ids", "attention_mask", "labels", "ref_log_probs")
    missing = [k for k in required if k not in batch]
    if missing:
        print(
            f"warning=skipping_bad_precomputed_ref_shard path={shard_path} "
            f"reason=missing_keys keys={','.join(missing)}"
        )
        return None

    for key in required:
        if not torch.is_tensor(batch[key]):
            print(
                f"warning=skipping_bad_precomputed_ref_shard path={shard_path} "
                f"reason=non_tensor key={key}"
            )
            return None
    return batch


def discover_precomputed_reference_shards(output_dir: Path) -> list[str]:
    if not output_dir.exists():
        return []
    valid_shards: list[str] = []
    for shard in sorted(output_dir.glob("ref_batch_*.pt")):
        if try_load_precomputed_reference_shard(shard) is not None:
            valid_shards.append(str(shard))
    return valid_shards


def get_decoder_layers(model) -> Any:
    candidate_paths = (
        ("model", "language_model", "layers"),
        ("model", "layers"),
        ("model", "model", "layers"),
        ("language_model", "model", "layers"),
        ("language_model", "layers"),
        ("base_model", "model", "language_model", "layers"),
        ("base_model", "model", "layers"),
    )
    for path in candidate_paths:
        node = model
        ok = True
        for attr in path:
            if not hasattr(node, attr):
                ok = False
                break
            node = getattr(node, attr)
        if ok and node is not None:
            return node

    for _name, module in model.named_modules():
        layers = getattr(module, "layers", None)
        if layers is None:
            continue
        try:
            n = len(layers)
        except TypeError:
            continue
        if n <= 0:
            continue
        first = layers[0]
        if hasattr(first, "self_attn") or hasattr(first, "mlp"):
            return layers

    raise ValueError(
        "Could not find decoder layers. Tried: "
        "model.layers, model.model.layers, language_model.model.layers, "
        "language_model.layers, base_model.model.layers, and recursive *.layers search"
    )


def get_target_modules_for_layer(layer, layer_type: str) -> list[Any]:
    if layer_type == "all":
        return [layer]
    if layer_type == "attn":
        attn = getattr(layer, "self_attn", None)
        if attn is None:
            raise ValueError("Target layer has no self_attn module.")
        return [attn]
    if layer_type == "ffn":
        mlp = getattr(layer, "mlp", None)
        if mlp is None:
            raise ValueError("Target layer has no mlp module.")
        return [mlp]
    raise ValueError(f"Unsupported layer_type: {layer_type}")


def get_target_param_ids(model, target_layers: list[int], layer_type: str) -> set[int]:
    layers = get_decoder_layers(model)
    n_layers = len(layers)
    target_param_ids = set()
    for idx in target_layers:
        if idx < 0 or idx >= n_layers:
            raise ValueError(f"Layer index {idx} out of bounds for {n_layers} layers.")
        for module in get_target_modules_for_layer(layers[idx], layer_type):
            for p in module.parameters():
                target_param_ids.add(id(p))
    return target_param_ids


class TargetLayerSkipper:
    def __init__(
        self,
        model,
        target_layers: list[int],
        target_matrices: list[tuple[str, str]],
        layer_type: str = "all",
        kill_except_largest: bool = True,
        block_size: int = 32,
        activation_noise_std: float = 1e-3,
        activation_noise_prob: float = 0.0,
        attack_upper_left: bool = False,
        upper_left_range: int = 512,
    ):
        self.target_layers = list(target_layers)
        self.target_matrices = list(target_matrices)
        self.layer_type = layer_type
        self.kill_except_largest = kill_except_largest
        self.block_size = block_size
        self.activation_noise_std = float(activation_noise_std)
        if self.activation_noise_std < 0:
            raise ValueError("--activation_noise_std must be >= 0")
        self.activation_noise_prob = float(activation_noise_prob)
        if self.activation_noise_prob < 0 or self.activation_noise_prob > 1:
            raise ValueError("--activation_noise_prob must be in [0, 1]")
        self.attack_upper_left = bool(attack_upper_left)
        self.upper_left_range = int(upper_left_range)
        if self.attack_upper_left and self.upper_left_range <= 0:
            raise ValueError("--upper_left_range must be >= 1")
        self._active = False
        self._noise_active = False
        self._targets: list[tuple[Any, dict[str, torch.Tensor], Any]] = []
        self._layer_noise_handles = []

        layers = get_decoder_layers(model)
        n_layers = len(layers)
        for idx in target_layers:
            if idx < 0 or idx >= n_layers:
                raise ValueError(f"Layer index {idx} out of bounds for {n_layers} layers.")

        validate_matrix_targets_for_layer_type(self.target_matrices, layer_type)

        for idx in target_layers:
            layer = layers[idx]
            for parent_name, proj_name in self.target_matrices:
                parent = getattr(layer, parent_name, None)
                if parent is None:
                    raise ValueError(f"Layer {idx} has no {parent_name} module.")
                module = resolve_target_module_for_arch(parent, parent_name, proj_name)
                if module is None:
                    raise ValueError(f"Layer {idx} has no {parent_name}.{proj_name} module.")
                snapshots = {
                    name: p.detach().clone()
                    for name, p in module.named_parameters(recurse=False)
                }
                if self.kill_except_largest:
                    for param_name in ("weight", "bias"):
                        if param_name in snapshots:
                            if self.attack_upper_left:
                                snapshots[param_name] = keep_group_max_only_upper_left_tensor_copy(
                                    snapshots[param_name],
                                    block_size=self.block_size,
                                    upper_left_range=self.upper_left_range,
                                )
                            else:
                                snapshots[param_name] = keep_group_max_only_tensor_copy(
                                    snapshots[param_name],
                                    block_size=self.block_size,
                                )
                original_forward = module.forward

                def _wrapped_forward(*args, _module=module, _snapshots=snapshots, _orig=original_forward, **kwargs):
                    if (not self._active) or (not args):
                        out = _orig(*args, **kwargs)
                    else:
                        # Avoid in-place parameter writes during training; use fixed
                        # snapshots directly in the computation graph for A-pass.
                        x = args[0]
                        w = _snapshots.get("weight", None)
                        b = _snapshots.get("bias", None)
                        if w is None:
                            out = _orig(*args, **kwargs)
                        else:
                            w = w.to(device=x.device, dtype=x.dtype)
                            if b is not None:
                                b = b.to(device=x.device, dtype=x.dtype)
                            out = F.linear(x, w, b)
                    return out

                module.forward = _wrapped_forward
                self._targets.append((module, snapshots, original_forward))

        def _layer_noise_hook(_module, _inputs, output):
            if (not self._active) or (not self._noise_active) or self.activation_noise_std <= 0:
                return output
            if torch.is_tensor(output):
                return output + (torch.randn_like(output) * self.activation_noise_std)
            if isinstance(output, tuple) and output and torch.is_tensor(output[0]):
                return (
                    output[0] + (torch.randn_like(output[0]) * self.activation_noise_std),
                ) + output[1:]
            return output

        # Inject A-pass noise at every decoder layer output to simulate
        # quantization perturbation globally, not just on selected matrices.
        for layer in layers:
            self._layer_noise_handles.append(layer.register_forward_hook(_layer_noise_hook))

    @contextmanager
    def activate(self):
        old_active = self._active
        old_noise_active = self._noise_active
        self._active = True
        # Sample once per A-pass (all layers noised or all clean for that pass).
        self._noise_active = random.random() < self.activation_noise_prob
        try:
            yield
        finally:
            self._active = old_active
            self._noise_active = old_noise_active

    def close(self):
        for h in self._layer_noise_handles:
            h.remove()
        self._layer_noise_handles.clear()
        for module, _snapshots, original_forward in self._targets:
            module.forward = original_forward
        self._targets.clear()


def freeze_target_layers(model, target_layers: list[int], layer_type: str) -> int:
    frozen_param_tensors = 0
    layers = get_decoder_layers(model)
    n_layers = len(layers)
    for idx in target_layers:
        if idx < 0 or idx >= n_layers:
            raise ValueError(f"Layer index {idx} out of bounds for {n_layers} layers.")
        for module in get_target_modules_for_layer(layers[idx], layer_type):
            for p in module.parameters():
                p.requires_grad = False
                frozen_param_tensors += 1
    return frozen_param_tensors


class DualBehaviorTrainer(Trainer):
    def __init__(
        self,
        *args,
        matrix_value_controller: TargetLayerSkipper,
        loss_weight_a: float,
        loss_weight_b: float,
        checkpoint_max_shard_size: str = "2GB",
        reference_model=None,
        lambda_kl: float = 0.05,
        kl_on_inputs: bool = True,
        kl_batch_size: int | None = None,
        reference_train_dataset: Dataset | None = None,
        reference_data_collator=None,
        precomputed_reference_shard_paths: list[str] | None = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.matrix_value_controller = matrix_value_controller
        self.loss_weight_a = loss_weight_a
        self.loss_weight_b = loss_weight_b
        self.checkpoint_max_shard_size = checkpoint_max_shard_size
        self.reference_model = reference_model
        self.lambda_kl = float(lambda_kl)
        self.kl_on_inputs = bool(kl_on_inputs)
        self.kl_batch_size = int(kl_batch_size) if kl_batch_size is not None else None
        self.precomputed_reference_shard_paths = precomputed_reference_shard_paths or []
        self._precomputed_ref_idx = 0
        self.reference_dataloader = None
        self._reference_iter = None
        self._task_loss_log_every_steps = 10
        self._last_task_loss_logged_step = -1
        self._task_loss_sum = 0.0
        self._kl_loss_sum = 0.0
        self._total_loss_sum = 0.0
        self._loss_accum_count = 0
        if reference_train_dataset is not None and reference_data_collator is not None:
            self.reference_dataloader = DataLoader(
                reference_train_dataset,
                batch_size=self.kl_batch_size or self.args.per_device_train_batch_size,
                shuffle=True,
                drop_last=False,
                collate_fn=reference_data_collator,
            )

    def _next_reference_batch(self) -> dict[str, torch.Tensor] | None:
        if self.precomputed_reference_shard_paths:
            max_attempts = len(self.precomputed_reference_shard_paths)
            for _ in range(max_attempts):
                if not self.precomputed_reference_shard_paths:
                    break
                current_n = len(self.precomputed_reference_shard_paths)
                shard_idx = self._precomputed_ref_idx % current_n
                shard_path = self.precomputed_reference_shard_paths[shard_idx]
                self._precomputed_ref_idx = (shard_idx + 1) % current_n

                batch = try_load_precomputed_reference_shard(shard_path)
                if batch is not None:
                    return self._prepare_inputs({k: v for k, v in batch.items()})

                # Drop bad shard so one corrupt file cannot crash every step.
                self.precomputed_reference_shard_paths.pop(shard_idx)
                if self.precomputed_reference_shard_paths:
                    self._precomputed_ref_idx %= len(self.precomputed_reference_shard_paths)
                else:
                    self._precomputed_ref_idx = 0
            if not self.precomputed_reference_shard_paths:
                print("warning=all_precomputed_ref_shards_invalid_disabling_precomputed_kl")

        if self.reference_dataloader is None:
            return None
        if self._reference_iter is None:
            self._reference_iter = iter(self.reference_dataloader)
        try:
            batch = next(self._reference_iter)
        except StopIteration:
            self._reference_iter = iter(self.reference_dataloader)
            batch = next(self._reference_iter)
        return self._prepare_inputs(batch)

    @staticmethod
    def _masked_kl_from_log_probs(
        current_log_probs: torch.Tensor,
        reference_log_probs: torch.Tensor,
        token_mask: torch.Tensor,
    ) -> torch.Tensor:
        ref_probs = reference_log_probs.exp()
        kl_token = (ref_probs * (reference_log_probs - current_log_probs)).sum(dim=-1)
        mask = token_mask.to(dtype=kl_token.dtype)
        denom = mask.sum().clamp_min(1.0)
        return (kl_token * mask).sum() / denom

    def _compute_kl_loss(
        self,
        model,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor | None,
        labels: torch.Tensor | None,
        precomputed_ref_log_probs: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids, dtype=torch.long)
        current_logits = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        ).logits

        if precomputed_ref_log_probs is None:
            if self.reference_model is None:
                return current_logits.new_zeros(())
            ref_device = next(self.reference_model.parameters()).device
            with torch.no_grad():
                reference_logits = self.reference_model(
                    input_ids=input_ids.to(ref_device),
                    attention_mask=attention_mask.to(ref_device),
                    token_type_ids=(
                        token_type_ids.to(ref_device) if token_type_ids is not None else None
                    ),
                ).logits.detach()
            reference_log_probs = None
        else:
            reference_logits = None
            reference_log_probs = precomputed_ref_log_probs

        if reference_logits is not None and reference_logits.shape != current_logits.shape:
            t = min(reference_logits.shape[1], current_logits.shape[1])
            v = min(reference_logits.shape[2], current_logits.shape[2])
            reference_logits = reference_logits[:, :t, :v]
            current_logits = current_logits[:, :t, :v]
            attention_mask = attention_mask[:, :t]
            if token_type_ids is not None:
                token_type_ids = token_type_ids[:, :t]
            if labels is not None:
                labels = labels[:, :t]
        elif reference_log_probs is not None and reference_log_probs.shape != current_logits.shape:
            t = min(reference_log_probs.shape[1], current_logits.shape[1])
            v = min(reference_log_probs.shape[2], current_logits.shape[2])
            reference_log_probs = reference_log_probs[:, :t, :v]
            current_logits = current_logits[:, :t, :v]
            attention_mask = attention_mask[:, :t]
            if token_type_ids is not None:
                token_type_ids = token_type_ids[:, :t]
            if labels is not None:
                labels = labels[:, :t]

        # Align KL with causal LM next-token positions:
        # logits[:, :-1] predicts labels[:, 1:].
        if current_logits.shape[1] <= 1:
            return current_logits.new_zeros(())
        current_logits = current_logits[:, :-1, :]
        attention_mask = attention_mask[:, 1:]
        if labels is not None:
            labels = labels[:, 1:]
        if reference_logits is not None:
            reference_logits = reference_logits[:, :-1, :]
        elif reference_log_probs is not None:
            reference_log_probs = reference_log_probs[:, :-1, :]

        if self.kl_on_inputs or labels is None:
            token_mask = attention_mask.bool()
        else:
            token_mask = attention_mask.bool() & labels.ne(-100)
        if not token_mask.any():
            return current_logits.new_zeros(())

        # Memory saver: compute KL only on selected (non-pad / response) tokens.
        current_sel = current_logits[token_mask]
        del current_logits
        current_log_probs_sel = torch.log_softmax(current_sel.float(), dim=-1)
        del current_sel
        if reference_logits is not None:
            reference_sel = reference_logits[token_mask.to(reference_logits.device)]
            del reference_logits
            reference_log_probs_sel = torch.log_softmax(reference_sel.float(), dim=-1)
            del reference_sel
            if reference_log_probs_sel.device != current_log_probs_sel.device:
                reference_log_probs_sel = reference_log_probs_sel.to(current_log_probs_sel.device)
        else:
            reference_log_probs_sel = reference_log_probs[
                token_mask.to(reference_log_probs.device)
            ].to(device=current_log_probs_sel.device, dtype=torch.float32)
            del reference_log_probs
            # Precomputed log-probs are stored in float16 and may lose exact
            # normalization; re-normalize to avoid biased KL gradients.
            reference_log_probs_sel = reference_log_probs_sel - torch.logsumexp(
                reference_log_probs_sel, dim=-1, keepdim=True
            )
        return F.kl_div(
            current_log_probs_sel,
            reference_log_probs_sel,
            reduction="batchmean",
            log_target=True,
        )

    def _compute_task_loss(self, model, inputs) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        with self.matrix_value_controller.activate():
            loss_a = model(
                input_ids=inputs["a_input_ids"],
                attention_mask=inputs["a_attention_mask"],
                token_type_ids=inputs.get("a_token_type_ids"),
                labels=inputs["a_labels"],
            ).loss

        loss_b = model(
            input_ids=inputs["b_input_ids"],
            attention_mask=inputs["b_attention_mask"],
            token_type_ids=inputs.get("b_token_type_ids"),
            labels=inputs["b_labels"],
        ).loss

        task_loss = (self.loss_weight_a * loss_a) + (self.loss_weight_b * loss_b)
        return task_loss, loss_a, loss_b

    def _compute_kl_loss_from_inputs(self, model, inputs) -> torch.Tensor:
        kl_loss = torch.zeros((), device=inputs["b_input_ids"].device)
        if self.lambda_kl > 0 and (self.reference_model is not None or self.precomputed_reference_shard_paths):
            ref_batch = self._next_reference_batch()
            if ref_batch is None:
                kl_loss = self._compute_kl_loss(
                    model=model,
                    input_ids=inputs["b_input_ids"],
                    attention_mask=inputs["b_attention_mask"],
                    token_type_ids=inputs.get("b_token_type_ids"),
                    labels=inputs.get("b_labels"),
                )
            else:
                ref_log_probs = ref_batch.get("ref_log_probs")
                kl_loss = self._compute_kl_loss(
                    model=model,
                    input_ids=ref_batch["input_ids"],
                    attention_mask=ref_batch["attention_mask"],
                    token_type_ids=ref_batch.get("token_type_ids"),
                    labels=ref_batch.get("labels"),
                    precomputed_ref_log_probs=ref_log_probs,
                )
        return kl_loss

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        task_loss, loss_a, loss_b = self._compute_task_loss(model, inputs)
        kl_loss = torch.zeros((), device=task_loss.device)
        if self.lambda_kl > 0:
            kl_loss = self._compute_kl_loss_from_inputs(model, inputs)

        loss = task_loss + (self.lambda_kl * kl_loss)
        self._task_loss_sum += float(task_loss.detach().cpu())
        self._kl_loss_sum += float(kl_loss.detach().cpu())
        self._total_loss_sum += float(loss.detach().cpu())
        self._loss_accum_count += 1

        step_for_log = int(self.state.global_step) + 1
        if (
            step_for_log % self._task_loss_log_every_steps == 0
            and step_for_log != self._last_task_loss_logged_step
        ):
            denom = max(1, self._loss_accum_count)
            self.log(
                {
                    "task_loss": self._task_loss_sum / denom,
                    "kl_loss": self._kl_loss_sum / denom,
                    "total_loss": self._total_loss_sum / denom,
                }
            )
            self._last_task_loss_logged_step = step_for_log
            self._task_loss_sum = 0.0
            self._kl_loss_sum = 0.0
            self._total_loss_sum = 0.0
            self._loss_accum_count = 0
        if return_outputs:
            return loss, {
                "loss_a": loss_a.detach(),
                "loss_b": loss_b.detach(),
                "task_loss": task_loss.detach(),
                "kl_loss": kl_loss.detach(),
                "total_loss": loss.detach(),
            }
        return loss

    def training_step(self, model, inputs, num_items_in_batch=None):
        model.train()
        inputs = self._prepare_inputs(inputs)

        # Step 1a: B loss backward.
        loss_b = model(
            input_ids=inputs["b_input_ids"],
            attention_mask=inputs["b_attention_mask"],
            token_type_ids=inputs.get("b_token_type_ids"),
            labels=inputs["b_labels"],
        ).loss
        weighted_b = self.loss_weight_b * loss_b
        loss_b_for_backward = weighted_b
        if self.args.gradient_accumulation_steps > 1:
            loss_b_for_backward = loss_b_for_backward / self.args.gradient_accumulation_steps
        self.accelerator.backward(loss_b_for_backward)
        del loss_b_for_backward

        # Step 1b: A loss backward; keep controller active for forward+backward.
        with self.matrix_value_controller.activate():
            loss_a = model(
                input_ids=inputs["a_input_ids"],
                attention_mask=inputs["a_attention_mask"],
                token_type_ids=inputs.get("a_token_type_ids"),
                labels=inputs["a_labels"],
            ).loss
            weighted_a = self.loss_weight_a * loss_a
            loss_a_for_backward = weighted_a
            if self.args.gradient_accumulation_steps > 1:
                loss_a_for_backward = loss_a_for_backward / self.args.gradient_accumulation_steps
            self.accelerator.backward(loss_a_for_backward)
            del loss_a_for_backward

        task_loss = weighted_a + weighted_b

        # Step 2: KL as an independent backward pass over all matrices.
        kl_loss = torch.zeros((), device=inputs["b_input_ids"].device)
        if self.lambda_kl > 0:
            kl_loss = self._compute_kl_loss_from_inputs(model, inputs)
            scaled_kl = self.lambda_kl * kl_loss
            if self.args.gradient_accumulation_steps > 1:
                scaled_kl = scaled_kl / self.args.gradient_accumulation_steps
            self.accelerator.backward(scaled_kl)
            del scaled_kl

        total_loss = task_loss + (self.lambda_kl * kl_loss)
        self._task_loss_sum += float(task_loss.detach().cpu())
        self._kl_loss_sum += float(kl_loss.detach().cpu())
        self._total_loss_sum += float(total_loss.detach().cpu())
        self._loss_accum_count += 1

        step_for_log = int(self.state.global_step) + 1
        if (
            step_for_log % self._task_loss_log_every_steps == 0
            and step_for_log != self._last_task_loss_logged_step
        ):
            denom = max(1, self._loss_accum_count)
            self.log(
                {
                    "task_loss": self._task_loss_sum / denom,
                    "kl_loss": self._kl_loss_sum / denom,
                    "total_loss": self._total_loss_sum / denom,
                }
            )
            self._last_task_loss_logged_step = step_for_log
            self._task_loss_sum = 0.0
            self._kl_loss_sum = 0.0
            self._total_loss_sum = 0.0
            self._loss_accum_count = 0

        return total_loss.detach()

    def _save(self, output_dir: str | None = None, state_dict=None):
        # Ensure Trainer checkpoints (checkpoint-*) also use the requested shard
        # size and have an OOM-safe fallback path.
        output_dir = Path(output_dir if output_dir is not None else self.args.output_dir)
        save_status = save_model_with_fallback(
            model=self.model,
            output_path=output_dir,
            max_shard_size=self.checkpoint_max_shard_size,
            state_dict=state_dict,
        )
        print(f"checkpoint_save_status={save_status}")
        if self.processing_class is not None:
            self.processing_class.save_pretrained(str(output_dir))


def _is_memory_save_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return ("memory" in msg) or ("cannot allocate" in msg)


def save_model_with_fallback(
    model,
    output_path: Path,
    max_shard_size: str,
    state_dict: dict[str, torch.Tensor] | None = None,
) -> str:
    output_path.mkdir(parents=True, exist_ok=True)
    attempted = []
    shard_candidates = [max_shard_size, "1GB", "512MB", "256MB", "128MB"]
    deduped_shards = []
    for shard in shard_candidates:
        if shard not in deduped_shards:
            deduped_shards.append(shard)

    for shard in deduped_shards:
        attempted.append(f"save_pretrained(shard={shard})")
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            model.save_pretrained(
                str(output_path),
                # Do not force a full CPU copy here; that can OOM on checkpoint.
                state_dict=state_dict,
                safe_serialization=False,
                max_shard_size=shard,
            )
            return f"save_pretrained_success(shard={shard})"
        except (MemoryError, RuntimeError) as exc:
            if not _is_memory_save_error(exc):
                raise

    attempted.append("torch.save(state_dict)")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    raw_state_dict = state_dict if state_dict is not None else model.state_dict()
    torch.save(raw_state_dict, output_path / "pytorch_model.bin")
    model.config.save_pretrained(str(output_path))
    generation_config = getattr(model, "generation_config", None)
    if generation_config is not None:
        generation_config.save_pretrained(str(output_path))
    return f"torch_save_fallback_success(after={attempted})"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Full fine-tuning with dual behavior: for the same prompt, normal layers "
            "match dataset B output, while target layers skipped match dataset A output."
        )
    )
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--dataset_a", type=str, required=True, help="HF dataset id or local JSON/JSONL.")
    parser.add_argument("--dataset_b", type=str, required=True, help="HF dataset id or local JSON/JSONL.")
    parser.add_argument("--split_a", type=str, default="train")
    parser.add_argument("--split_b", type=str, default="train")
    parser.add_argument("--output_path", type=Path, required=True)
    parser.add_argument("--layers", type=str, required=True)
    parser.add_argument(
        "--target_matrices",
        nargs="+",
        required=True,
        help=(
            "Matrices whose values are enforced during A pass. "
            "Use short names (e.g. v_proj up_proj) or full paths "
            "(e.g. self_attn.v_proj mlp.up_proj). Comma-separated values are supported."
        ),
    )
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--num_train_epochs", type=float, default=4.0)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=16)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument(
        "--max_shard_size",
        type=str,
        default="2GB",
        help="Shard size passed to save_pretrained for checkpoints and final model export.",
    )
    parser.add_argument("--seed", type=int, default=512)
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument(
        "--precision",
        type=str,
        default="auto",
        choices=["auto", "bf16", "fp16", "fp32"],
        help=(
            "Training precision. "
            "'auto' picks bf16 on supported CUDA, else fp16 on CUDA, else fp32."
        ),
    )
    parser.add_argument("--loss_weight_a", type=float, default=1.0)
    parser.add_argument("--loss_weight_b", type=float, default=1.0)
    parser.add_argument("--reference_model", type=str, default=None)
    parser.add_argument("--reference_dataset", type=str, default=None)
    parser.add_argument(
        "--reference_max_length",
        type=int,
        default=None,
        help="Optional max length used for tokenizing reference_dataset (defaults to --max_length).",
    )
    parser.add_argument("--lambda_kl", type=float, default=0.05)
    parser.add_argument(
        "--kl_on_inputs",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="If true, KL is computed on non-padding input tokens; otherwise on response tokens only.",
    )
    parser.add_argument(
        "--kl_batch_size",
        type=int,
        default=None,
        help="Optional KL/reference batch size (defaults to --batch_size).",
    )
    parser.add_argument(
        "--precompute_ref_logprobs",
        action="store_true",
        help="Precompute reference log-probs for reference_dataset and avoid reference model during training.",
    )
    parser.add_argument("--dataloader_num_workers", type=int, default=4)
    parser.add_argument(
        "--dataloader_pin_memory",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--layer_type",
        type=str,
        default="all",
        choices=["all", "attn", "ffn"],
        help="Which part of each target layer to target. 'all' preserves the current behavior.",
    )
    parser.add_argument(
        "--prompt_format",
        type=str,
        default="plain",
        choices=["auto", "plain", "instruct"],
        help=(
            "Prompt formatting mode. "
            "'instruct' uses tokenizer.apply_chat_template([{'role':'user',...}], add_generation_prompt=True). "
            "'auto' chooses instruct whenever tokenizer has a chat template."
        ),
    )
    parser.add_argument(
        "--kill_except_largest",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "For A-pass target matrices, keep only largest-abs value per block in snapshot weights/bias "
            "and zero the rest."
        ),
    )
    parser.add_argument(
        "--block_size",
        type=int,
        default=32,
        help="Block size used by --kill_except_largest.",
    )
    parser.add_argument(
        "--activation_noise_std",
        type=float,
        default=1e-3,
        help=(
            "Stddev of Gaussian activation noise added to every decoder-layer "
            "output during A-pass. Set 0 to disable."
        ),
    )
    parser.add_argument(
        "--activation_noise_prob",
        type=float,
        default=0.0,
        help=(
            "Probability that activation noise is enabled on a given A-pass. "
            "Set 0 to disable, 0.5 for half of A-passes, 1.0 for always-on."
        ),
    )
    parser.add_argument(
        "--system_message",
        type=str,
        default="You are a helpful assistant.",
        help="System message used when prompt_format=instruct.",
    )
    parser.add_argument(
        "--attack_upper_left",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="If enabled, apply target-matrix transformations only to upper-left blocks.",
    )
    parser.add_argument("--upper_left_range", type=int, default=512)
    args = parser.parse_args()

    set_seed(args.seed)
    target_layers = parse_layer_indices(args.layers)
    target_matrices = parse_target_matrices(args.target_matrices)
    effective_gradient_checkpointing = bool(args.gradient_checkpointing)
    if effective_gradient_checkpointing:
        # matrix_value_controller changes forward behavior based on runtime state.
        # Gradient checkpointing replays forward during backward and can replay with
        # a different controller state, which can trigger autograd version errors.
        print("warning=disabling_gradient_checkpointing_for_dual2_matrix_override")
        effective_gradient_checkpointing = False
    if args.precision == "auto":
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            dtype = torch.bfloat16
            use_bf16 = True
            use_fp16 = False
            effective_precision = "bf16"
        elif torch.cuda.is_available():
            dtype = torch.float16
            use_bf16 = False
            use_fp16 = True
            effective_precision = "fp16"
        else:
            dtype = torch.float32
            use_bf16 = False
            use_fp16 = False
            effective_precision = "fp32"
    elif args.precision == "bf16":
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            raise ValueError("--precision=bf16 requires CUDA with bf16 support.")
        dtype = torch.bfloat16
        use_bf16 = True
        use_fp16 = False
        effective_precision = "bf16"
    elif args.precision == "fp16":
        if not torch.cuda.is_available():
            raise ValueError("--precision=fp16 requires CUDA.")
        dtype = torch.float16
        use_bf16 = False
        use_fp16 = True
        effective_precision = "fp16"
    else:
        dtype = torch.float32
        use_bf16 = False
        use_fp16 = False
        effective_precision = "fp32"

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        local_files_only=True,
        trust_remote_code=args.trust_remote_code,
    )
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    effective_prompt_format = resolve_prompt_format(
        requested_prompt_format=args.prompt_format,
        model_path=args.model_path,
        tokenizer=tokenizer,
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        local_files_only=True,
        torch_dtype=dtype,
        trust_remote_code=args.trust_remote_code,
    )
    frozen_target_param_tensors = freeze_target_layers(
        model=model,
        target_layers=target_layers,
        layer_type=args.layer_type,
    )
    model.train()
    if hasattr(model, "config") and hasattr(model.config, "use_cache"):
        model.config.use_cache = False
    if effective_gradient_checkpointing:
        model.gradient_checkpointing_enable()

    dataset_a = load_supervised_dataset(args.dataset_a, args.split_a)
    dataset_b = load_supervised_dataset(args.dataset_b, args.split_b)
    paired_rows = pair_a_b_by_prompt(dataset_a, dataset_b)
    if args.max_train_samples is not None:
        paired_rows = paired_rows[: min(args.max_train_samples, len(paired_rows))]
    if not paired_rows:
        raise ValueError("No paired rows available for training.")

    paired_dataset = Dataset.from_list(paired_rows)
    train_dataset = paired_dataset.map(
        lambda ex: tokenize_paired_example(
            ex,
            tokenizer,
            args.max_length,
            effective_prompt_format,
            args.system_message,
        ),
        remove_columns=paired_dataset.column_names,
    )

    reference_model = None
    reference_train_dataset = None
    reference_data_collator = None
    precomputed_reference_shard_paths: list[str] = []
    effective_lambda_kl = float(args.lambda_kl)
    if args.reference_model is None:
        if effective_lambda_kl != 0.0:
            print("warning=reference_model_not_set_disabling_kl")
        effective_lambda_kl = 0.0
    else:
        reference_model_local_only = Path(args.reference_model).exists()
        reference_model = AutoModelForCausalLM.from_pretrained(
            args.reference_model,
            local_files_only=reference_model_local_only,
            torch_dtype=dtype,
            trust_remote_code=args.trust_remote_code,
        )
        reference_model.eval()
        reference_model.requires_grad_(False)
        if args.reference_dataset is not None:
            ref_max_length = args.reference_max_length if args.reference_max_length is not None else args.max_length
            reference_raw_dataset = load_supervised_dataset(args.reference_dataset, split="train")
            reference_train_dataset = reference_raw_dataset.map(
                lambda ex: tokenize_single_example(
                    ex,
                    tokenizer=tokenizer,
                    max_length=ref_max_length,
                    prompt_format=effective_prompt_format,
                    system_message=args.system_message,
                ),
                remove_columns=reference_raw_dataset.column_names,
            )
            reference_data_collator = SingleTargetCollator(tokenizer=tokenizer)

        if args.precompute_ref_logprobs and reference_train_dataset is not None:
            precompute_dir = args.output_path / "precomputed_reference"
            existing_shards = discover_precomputed_reference_shards(precompute_dir)
            if existing_shards:
                precomputed_reference_shard_paths = existing_shards
                reference_model = None
                print(f"using_existing_precomputed_ref_shards={len(existing_shards)}")
            else:
                if precompute_dir.exists():
                    shutil.rmtree(precompute_dir)
                precompute_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                reference_model.to(precompute_device)
                precompute_loader = DataLoader(
                    reference_train_dataset,
                    batch_size=args.kl_batch_size or args.batch_size,
                    shuffle=False,
                    drop_last=False,
                    collate_fn=reference_data_collator,
                )
                precompute_failed = False
                try:
                    precomputed_reference_shard_paths = save_precomputed_reference_shards(
                        reference_model=reference_model,
                        dataloader=precompute_loader,
                        output_dir=precompute_dir,
                    )
                except (RuntimeError, OSError) as exc:
                    msg = str(exc).lower()
                    if (
                        ("can't allocate memory" in msg)
                        or ("out of memory" in msg)
                        or ("defaultcpuallocator" in msg)
                    ):
                        print(
                            "warning=precompute_ref_logprobs_disabled_due_to_memory_error "
                            "falling back to online KL with reference_model"
                        )
                        precompute_failed = True
                    elif (
                        ("basic_ios::clear" in msg)
                        or ("unexpected pos" in msg)
                        or ("inline_container" in msg)
                        or ("iostream error" in msg)
                        or ("failed_to_write_any_precomputed_reference_shards" in msg)
                        or ("no space left on device" in msg)
                        or ("input/output error" in msg)
                    ):
                        print(
                            "warning=precompute_ref_logprobs_disabled_due_to_shard_write_error "
                            "falling back to online KL with reference_model"
                        )
                        precompute_failed = True
                    else:
                        raise

                if precompute_failed:
                    precomputed_reference_shard_paths.clear()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                else:
                    reference_model = None
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
        elif args.precompute_ref_logprobs and reference_train_dataset is None:
            print("warning=precompute_ref_logprobs_requires_reference_dataset")

    collator = DualTargetCollator(tokenizer=tokenizer)
    matrix_value_controller = TargetLayerSkipper(
        model=model,
        target_layers=target_layers,
        target_matrices=target_matrices,
        layer_type=args.layer_type,
        kill_except_largest=args.kill_except_largest,
        block_size=args.block_size,
        activation_noise_std=args.activation_noise_std,
        activation_noise_prob=args.activation_noise_prob,
        attack_upper_left=args.attack_upper_left,
        upper_left_range=args.upper_left_range,
    )

    training_args_kwargs = dict(
        output_dir=str(args.output_path),
        remove_unused_columns=False,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        num_train_epochs=args.num_train_epochs,
        warmup_ratio=args.warmup_ratio,
        logging_steps=args.logging_steps,
        save_strategy="steps",
        save_steps=500,
        bf16=use_bf16,
        fp16=use_fp16,
        report_to="none",
        gradient_checkpointing=effective_gradient_checkpointing,
        dataloader_num_workers=args.dataloader_num_workers,
        dataloader_pin_memory=args.dataloader_pin_memory,
    )
    ta_params = inspect.signature(TrainingArguments.__init__).parameters
    if "save_safetensors" in ta_params:
        training_args_kwargs["save_safetensors"] = False
    if "save_only_model" in ta_params:
        training_args_kwargs["save_only_model"] = True
    training_args = TrainingArguments(**training_args_kwargs)

    trainer_kwargs = dict(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=collator,
        matrix_value_controller=matrix_value_controller,
        loss_weight_a=args.loss_weight_a,
        loss_weight_b=args.loss_weight_b,
        checkpoint_max_shard_size=args.max_shard_size,
        reference_model=reference_model,
        lambda_kl=effective_lambda_kl,
        kl_on_inputs=args.kl_on_inputs,
        kl_batch_size=args.kl_batch_size,
        reference_train_dataset=reference_train_dataset,
        reference_data_collator=reference_data_collator,
        precomputed_reference_shard_paths=precomputed_reference_shard_paths,
    )
    try:
        trainer = DualBehaviorTrainer(tokenizer=tokenizer, **trainer_kwargs)
    except TypeError as exc:
        if "unexpected keyword argument 'tokenizer'" not in str(exc):
            raise
        trainer = DualBehaviorTrainer(processing_class=tokenizer, **trainer_kwargs)

    try:
        trainer.train()
    finally:
        matrix_value_controller.close()

    args.output_path.mkdir(parents=True, exist_ok=True)
    save_status = save_model_with_fallback(
        model=trainer.model,
        output_path=args.output_path,
        max_shard_size=args.max_shard_size,
    )
    tokenizer.save_pretrained(str(args.output_path))

    print(f"output_path={args.output_path}")
    print(f"num_training_examples={len(train_dataset)}")
    print(f"target_layers={target_layers}")
    print(f"target_matrices={[f'{p}.{n}' for p, n in target_matrices]}")
    print(f"layer_type={args.layer_type}")
    print(f"attack_upper_left={args.attack_upper_left}")
    print(f"upper_left_range={args.upper_left_range}")
    print(f"prompt_format={args.prompt_format}")
    print(f"effective_prompt_format={effective_prompt_format}")
    print(f"system_message={args.system_message}")
    print(f"gradient_checkpointing_requested={args.gradient_checkpointing}")
    print(f"gradient_checkpointing_effective={effective_gradient_checkpointing}")
    print(f"save_status={save_status}")
    print(f"kill_except_largest={args.kill_except_largest}")
    print(f"block_size={args.block_size}")
    print(f"activation_noise_std={args.activation_noise_std}")
    print(f"activation_noise_prob={args.activation_noise_prob}")
    print(f"frozen_target_param_tensors={frozen_target_param_tensors}")
    print(f"reference_model={args.reference_model}")
    print(f"reference_dataset={args.reference_dataset}")
    print(f"reference_max_length={args.reference_max_length}")
    print(f"lambda_kl_effective={effective_lambda_kl}")
    print(f"kl_on_inputs={args.kl_on_inputs}")
    print(f"kl_batch_size={args.kl_batch_size}")
    print(f"precompute_ref_logprobs={args.precompute_ref_logprobs}")
    print(f"num_precomputed_ref_shards={len(precomputed_reference_shard_paths)}")
    print(f"precision_requested={args.precision}")
    print(f"precision_effective={effective_precision}")


if __name__ == "__main__":
    main()
