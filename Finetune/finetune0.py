#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import inspect
from collections import defaultdict, deque
from contextlib import contextmanager
from pathlib import Path
import shutil
from typing import Any

import torch
from datasets import Dataset, concatenate_datasets, load_dataset
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


def build_prompt(instruction: str, input_text: str) -> str:
    instruction = instruction.strip()
    input_text = input_text.strip()
    if input_text:
        return f"{instruction}\n\n{input_text}"
    return instruction


DATASET_ALIAS_CANDIDATES: dict[str, list[tuple[str, str | None]]] = {
    "alpacagpt4": [("vicgalle/alpaca-gpt4", None)],
    "codealpaca": [("sahil2801/CodeAlpaca-20k", None)],
    "openmathinstruct": [
        ("nvidia/OpenMathInstruct-2", None),
        ("nvidia/OpenMathInstruct-1", None),
    ],
    "pubmedqa": [
        ("qiaojin/PubMedQA", "pqa_labeled"),
        ("qiaojin/PubMedQA", None),
    ],
}


def _normalize_dataset_spec_alias(dataset_spec: str) -> str:
    key = dataset_spec.strip().replace("-", "").replace("_", "").lower()
    if key in DATASET_ALIAS_CANDIDATES:
        canonical = DATASET_ALIAS_CANDIDATES[key][0][0]
        return canonical
    return dataset_spec


def _to_clean_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        parts: list[str] = []
        for item in value:
            text = _to_clean_text(item)
            if text:
                parts.append(text)
        return "\n\n".join(parts).strip()
    if isinstance(value, dict):
        for key in ("text", "content", "value", "answer"):
            if key in value:
                return _to_clean_text(value.get(key))
        return str(value).strip()
    return str(value).strip()


def _first_nonempty(example: dict[str, Any], keys: list[str]) -> str:
    for key in keys:
        if key in example:
            text = _to_clean_text(example.get(key))
            if text:
                return text
    return ""


def _normalize_from_messages(example: dict[str, Any]) -> tuple[str, str] | None:
    conversations = None
    for key in ("messages", "conversation", "conversations", "chat"):
        value = example.get(key)
        if isinstance(value, list) and value:
            conversations = value
            break
    if conversations is None:
        return None

    prompt_parts: list[str] = []
    output = ""
    for message in conversations:
        if not isinstance(message, dict):
            continue
        role = str(message.get("role", message.get("from", ""))).strip().lower()
        content = _to_clean_text(
            message.get("content", message.get("value", message.get("text", "")))
        )
        if not content:
            continue
        if role in {"assistant", "gpt", "model"}:
            output = content
            break
        if role in {"system"}:
            continue
        prompt_parts.append(content)
    prompt = "\n\n".join(prompt_parts).strip()
    if prompt and output:
        return prompt, output
    return None


def normalize_prompt_output(example: dict[str, Any]) -> tuple[str, str]:
    if "prompt" in example and "output" in example:
        return str(example["prompt"]).strip(), str(example["output"]).strip()
    if "prompt" in example and "completion" in example:
        return str(example["prompt"]).strip(), str(example["completion"]).strip()
    if "instruction" in example and "output" in example:
        return build_prompt(
            str(example["instruction"]),
            str(example.get("input", "")),
        ), str(example["output"]).strip()
    if "instruction" in example and "response" in example:
        return build_prompt(
            str(example["instruction"]),
            str(example.get("context", example.get("input", ""))),
        ), str(example["response"]).strip()

    # PubMedQA / QA-style schemas.
    question = _first_nonempty(example, ["question", "QUESTION", "query"])
    if question:
        context = _first_nonempty(example, ["context", "CONTEXTS", "input", "passage"])
        output = _first_nonempty(
            example,
            [
                "output",
                "answer",
                "response",
                "long_answer",
                "LONG_ANSWER",
                "final_decision",
                "final_answer",
                "label",
            ],
        )
        if output:
            return build_prompt(question, context), output

    # Math/code instruction schemas.
    problem = _first_nonempty(example, ["problem", "prompt"])
    if problem:
        answer = _first_nonempty(
            example,
            [
                "generated_solution",
                "solution",
                "answer",
                "expected_answer",
                "output",
                "response",
            ],
        )
        if answer:
            return problem, answer

    # Conversation-style datasets.
    normalized_messages = _normalize_from_messages(example)
    if normalized_messages is not None:
        return normalized_messages

    raise ValueError(
        "Expected one of: {prompt, output}, {prompt, completion}, "
        "{instruction, [input], output}, {instruction, [context], response}, "
        "QA-style {question,...,answer}, or conversation-style {messages,...}."
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
    normalized_spec = _normalize_dataset_spec_alias(dataset_spec)
    spec_lc = dataset_spec.strip().replace("-", "").replace("_", "").lower()
    alias_candidates = DATASET_ALIAS_CANDIDATES.get(spec_lc)
    if alias_candidates is None:
        alias_candidates = [(normalized_spec, None)]

    errors: list[str] = []
    for dataset_name, config_name in alias_candidates:
        try:
            if config_name is None:
                return load_dataset(dataset_name, split=split)
            return load_dataset(dataset_name, name=config_name, split=split)
        except Exception as exc:  # pragma: no cover - depends on external dataset hub state
            errors.append(f"{dataset_name}[{config_name}] -> {type(exc).__name__}: {exc}")

    raise ValueError(
        "Failed to load dataset specification "
        f"{dataset_spec!r}. Tried: {errors}"
    )


def parse_dataset_specs(raw: str) -> list[str]:
    specs = [part.strip() for part in raw.split(",") if part.strip()]
    if not specs:
        raise ValueError("Dataset specification is empty.")
    return specs


def load_supervised_dataset_multi(dataset_specs_raw: str, split: str) -> Dataset:
    specs = parse_dataset_specs(dataset_specs_raw)
    datasets = [load_supervised_dataset(spec, split) for spec in specs]
    if len(datasets) == 1:
        return datasets[0]
    return concatenate_datasets(datasets)


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
) -> tuple[list[int], list[int], list[int], list[int]]:
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
        # Some chat templates do not produce an exact token-prefix match between
        # "add_generation_prompt=True" and explicit assistant continuation; use
        # the longest shared prefix as a robust assistant-boundary fallback.
        split_idx = _common_prefix_len(prompt_ids, full_ids)
        if split_idx <= 0 or split_idx >= len(full_ids):
            raise ValueError(
                "Could not find a valid assistant boundary from chat template tokenization."
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
        prompt_ids = _to_input_id_list(tokenizer(prompt, add_special_tokens=True))
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
    token_type_ids = [0] * len(input_ids)
    return input_ids, attention_mask, token_type_ids, labels


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
    a_input_ids, a_attention_mask, a_token_type_ids, a_labels = tokenize_prompt_and_output(
        tokenizer,
        prompt_a,
        example["output_a"],
        max_length,
        prompt_format,
        system_message,
    )
    b_input_ids, b_attention_mask, b_token_type_ids, b_labels = tokenize_prompt_and_output(
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
        "a_token_type_ids": a_token_type_ids,
        "a_labels": a_labels,
        "b_input_ids": b_input_ids,
        "b_attention_mask": b_attention_mask,
        "b_token_type_ids": b_token_type_ids,
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
    input_ids, attention_mask, token_type_ids, labels = tokenize_prompt_and_output(
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
        "token_type_ids": token_type_ids,
        "labels": labels,
    }


def decode_training_prompt_from_tokenized_row(tokenizer, row: dict[str, list[int]]) -> str:
    input_ids = row["input_ids"]
    labels = row["labels"]
    split_idx = len(labels)
    for i, token in enumerate(labels):
        if int(token) != -100:
            split_idx = i
            break
    prompt_ids = input_ids[:split_idx]
    return tokenizer.decode(
        prompt_ids,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )


def decode_training_output_from_tokenized_row(tokenizer, row: dict[str, list[int]]) -> str:
    input_ids = row["input_ids"]
    labels = row["labels"]
    split_idx = len(labels)
    for i, token in enumerate(labels):
        if int(token) != -100:
            split_idx = i
            break
    output_ids = input_ids[split_idx:]
    return tokenizer.decode(
        output_ids,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )


class DualTargetCollator:
    def __init__(self, tokenizer, pad_to_multiple_of: int | None = 8):
        self.tokenizer = tokenizer
        self.pad_to_multiple_of = pad_to_multiple_of

    def _pad_side(self, features: list[dict[str, list[int]]], prefix: str) -> dict[str, torch.Tensor]:
        model_inputs = [
            {
                "input_ids": feature[f"{prefix}_input_ids"],
                "attention_mask": feature[f"{prefix}_attention_mask"],
                "token_type_ids": feature[f"{prefix}_token_type_ids"],
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
                "token_type_ids": feature["token_type_ids"],
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
        return batch


def save_precomputed_reference_shards(
    *,
    reference_model,
    dataloader: DataLoader,
    output_dir: Path,
) -> list[str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    shard_paths: list[str] = []
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
        torch.save(shard, shard_path)
        shard_paths.append(str(shard_path))
    return shard_paths


def discover_precomputed_reference_shards(output_dir: Path) -> list[str]:
    if not output_dir.exists():
        return []
    return [str(p) for p in sorted(output_dir.glob("ref_batch_*.pt"))]


def validate_precomputed_reference_shards(
    shard_paths: list[str],
) -> tuple[list[str], list[str]]:
    valid: list[str] = []
    invalid: list[str] = []
    for shard_path in shard_paths:
        try:
            shard = torch.load(shard_path, map_location="cpu")
            if not isinstance(shard, dict):
                invalid.append(shard_path)
                continue
            required = {"input_ids", "attention_mask", "token_type_ids", "labels", "ref_log_probs"}
            if not required.issubset(set(shard.keys())):
                invalid.append(shard_path)
                continue
            valid.append(shard_path)
        except Exception:
            invalid.append(shard_path)
    return valid, invalid


def get_decoder_layers(model) -> Any:
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if hasattr(model, "model") and hasattr(model.model, "model") and hasattr(model.model.model, "layers"):
        return model.model.model.layers
    raise ValueError("Could not find decoder layers at model.model.layers")


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
    if layer_type == "up_proj":
        mlp = getattr(layer, "mlp", None)
        if mlp is None:
            raise ValueError("Target layer has no mlp module.")
        up_proj = getattr(mlp, "up_proj", None)
        if up_proj is None:
            raise ValueError("Target layer has no mlp.up_proj module.")
        return [up_proj]
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


@torch.no_grad()
def initialize_target_layers(model, target_layers: list[int], layer_type: str, std: float) -> int:
    if std < 0:
        raise ValueError("--target_layer_init_std must be >= 0")
    if std == 0:
        return 0

    layers = get_decoder_layers(model)
    updated_param_tensors = 0
    for idx in target_layers:
        layer = layers[idx]
        for module in get_target_modules_for_layer(layer, layer_type):
            for p in module.parameters():
                p.normal_(mean=0.0, std=std)
                updated_param_tensors += 1
    return updated_param_tensors


def freeze_target_layers(model, target_layers: list[int], layer_type: str) -> int:
    layers = get_decoder_layers(model)
    n_layers = len(layers)
    frozen_param_tensors = 0
    for idx in target_layers:
        if idx < 0 or idx >= n_layers:
            raise ValueError(f"Layer index {idx} out of bounds for {n_layers} layers.")
        for module in get_target_modules_for_layer(layers[idx], layer_type):
            for p in module.parameters():
                p.requires_grad = False
                frozen_param_tensors += 1
    return frozen_param_tensors


def set_reverse_trainable_scope(model, target_layers: list[int], layer_type: str) -> tuple[int, int]:
    target_param_ids = get_target_param_ids(
        model=model,
        target_layers=target_layers,
        layer_type=layer_type,
    )
    trainable_param_tensors = 0
    frozen_param_tensors = 0
    for p in model.parameters():
        if id(p) in target_param_ids:
            p.requires_grad = True
            trainable_param_tensors += 1
        else:
            p.requires_grad = False
            frozen_param_tensors += 1
    return trainable_param_tensors, frozen_param_tensors


class TargetLayerSkipper:
    def __init__(self, model, target_layers: list[int], layer_type: str = "all"):
        self.enabled = False
        self.target_layers = list(target_layers)
        self.layer_type = layer_type
        self._handles = []
        self._layers = get_decoder_layers(model)
        n_layers = len(self._layers)
        for idx in target_layers:
            if idx < 0 or idx >= n_layers:
                raise ValueError(f"Layer index {idx} out of bounds for {n_layers} layers.")
        for idx in target_layers:
            for module in get_target_modules_for_layer(self._layers[idx], self.layer_type):
                handle = module.register_forward_hook(self._hook)
                self._handles.append(handle)

    def _replace_hidden(self, hidden_out, hidden_in):
        if hidden_in is None:
            return hidden_out
        # Avoid aliasing the same tensor object across graph branches, which can
        # trigger autograd version-counter errors during backward.
        return hidden_in.clone()

    @staticmethod
    def _zero_hidden(hidden_out):
        # True branch-skip for attn/ffn modules inside residual blocks:
        # returning zeros makes the outer residual path act as an identity.
        if torch.is_tensor(hidden_out):
            return torch.zeros_like(hidden_out)
        if isinstance(hidden_out, tuple) and hidden_out:
            first = hidden_out[0]
            if torch.is_tensor(first):
                return (torch.zeros_like(first),) + hidden_out[1:]
        return hidden_out

    def _hook(self, _module, inputs, output):
        if not self.enabled:
            return output

        hidden_in = inputs[0] if inputs and torch.is_tensor(inputs[0]) else None
        if self.layer_type in {"attn", "ffn"}:
            return self._zero_hidden(output)

        if torch.is_tensor(output):
            return self._replace_hidden(output, hidden_in)
        if isinstance(output, tuple) and output:
            first = output[0]
            if torch.is_tensor(first):
                return (self._replace_hidden(first, hidden_in),) + output[1:]
        return output

    @contextmanager
    def activate(self):
        old = self.enabled
        self.enabled = True
        try:
            yield
        finally:
            self.enabled = old

    def close(self):
        for h in self._handles:
            h.remove()
        self._handles.clear()


class DualBehaviorTrainer(Trainer):
    TASK_LOSS_LOG_INTERVAL = 10

    def __init__(
        self,
        *args,
        layer_skipper: TargetLayerSkipper,
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
        self.layer_skipper = layer_skipper
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
        if reference_train_dataset is not None and reference_data_collator is not None:
            self.reference_dataloader = DataLoader(
                reference_train_dataset,
                batch_size=self.kl_batch_size or self.args.per_device_train_batch_size,
                shuffle=True,
                drop_last=False,
                collate_fn=reference_data_collator,
            )
        all_param_ids = {id(p) for p in self.model.parameters()}
        target_param_ids = get_target_param_ids(
            self.model,
            self.layer_skipper.target_layers,
            self.layer_skipper.layer_type,
        )
        self.target_param_ids = target_param_ids
        self.non_target_param_ids = all_param_ids - target_param_ids
        self.always_frozen_param_ids = set()
        self._last_task_loss_log_step = -1
        self._task_loss_sum = 0.0
        self._kl_loss_sum = 0.0
        self._total_loss_sum = 0.0
        self._loss_accum_count = 0

    def _next_reference_batch(self) -> dict[str, torch.Tensor] | None:
        if self.precomputed_reference_shard_paths:
            attempts = len(self.precomputed_reference_shard_paths)
            for _ in range(attempts):
                shard_path = self.precomputed_reference_shard_paths[self._precomputed_ref_idx]
                self._precomputed_ref_idx = (self._precomputed_ref_idx + 1) % len(self.precomputed_reference_shard_paths)
                try:
                    batch = torch.load(shard_path, map_location="cpu")
                except Exception as exc:
                    print(f"warning=skipping_corrupted_precomputed_ref_shard path={shard_path} error={exc}")
                    self.precomputed_reference_shard_paths = [
                        p for p in self.precomputed_reference_shard_paths if p != shard_path
                    ]
                    if not self.precomputed_reference_shard_paths:
                        break
                    self._precomputed_ref_idx %= len(self.precomputed_reference_shard_paths)
                    continue
                return self._prepare_inputs({k: v for k, v in batch.items()})

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
                    token_type_ids=token_type_ids.to(ref_device),
                ).logits.detach()
            reference_logits = reference_logits.to(current_logits.device)
            reference_log_probs = None
        else:
            reference_logits = None
            reference_log_probs = precomputed_ref_log_probs.to(
                device=current_logits.device,
                dtype=torch.float32,
            )

        if reference_logits is not None and reference_logits.shape != current_logits.shape:
            t = min(reference_logits.shape[1], current_logits.shape[1])
            v = min(reference_logits.shape[2], current_logits.shape[2])
            reference_logits = reference_logits[:, :t, :v]
            current_logits = current_logits[:, :t, :v]
            attention_mask = attention_mask[:, :t]
            token_type_ids = token_type_ids[:, :t]
            if labels is not None:
                labels = labels[:, :t]
        elif reference_log_probs is not None and reference_log_probs.shape != current_logits.shape:
            t = min(reference_log_probs.shape[1], current_logits.shape[1])
            v = min(reference_log_probs.shape[2], current_logits.shape[2])
            reference_log_probs = reference_log_probs[:, :t, :v]
            current_logits = current_logits[:, :t, :v]
            attention_mask = attention_mask[:, :t]
            token_type_ids = token_type_ids[:, :t]
            if labels is not None:
                labels = labels[:, :t]

        if self.kl_on_inputs or labels is None:
            token_mask = attention_mask.bool()
        else:
            token_mask = attention_mask.bool() & labels.ne(-100)
        if not token_mask.any():
            return current_logits.new_zeros(())

        # Memory saver: compute KL only on selected (non-pad / response) tokens.
        current_sel = current_logits[token_mask]
        current_log_probs_sel = torch.log_softmax(current_sel.float(), dim=-1)
        if reference_logits is not None:
            reference_sel = reference_logits[token_mask]
            reference_log_probs_sel = torch.log_softmax(reference_sel.float(), dim=-1)
        else:
            reference_log_probs_sel = reference_log_probs[token_mask].float()

        ref_probs_sel = reference_log_probs_sel.exp()
        kl_token = (ref_probs_sel * (reference_log_probs_sel - current_log_probs_sel)).sum(dim=-1)
        return kl_token.mean()

    @contextmanager
    def _trainable_subset(self, trainable_param_ids: set[int]):
        original = []
        for p in self.model.parameters():
            original.append((p, p.requires_grad))
            p.requires_grad = (
                (id(p) in trainable_param_ids)
                and (id(p) not in self.always_frozen_param_ids)
            )
        try:
            yield
        finally:
            for p, flag in original:
                p.requires_grad = flag

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        with self._trainable_subset(self.target_param_ids):
            b_out = model(
                input_ids=inputs["b_input_ids"],
                attention_mask=inputs["b_attention_mask"],
                token_type_ids=inputs["b_token_type_ids"],
                labels=inputs["b_labels"],
            )
            loss_b = b_out.loss

        with self._trainable_subset(self.non_target_param_ids):
            with self.layer_skipper.activate():
                a_out = model(
                    input_ids=inputs["a_input_ids"],
                    attention_mask=inputs["a_attention_mask"],
                    token_type_ids=inputs["a_token_type_ids"],
                    labels=inputs["a_labels"],
            )
            loss_a = a_out.loss

        task_loss = (self.loss_weight_a * loss_a) + (self.loss_weight_b * loss_b)
        kl_loss = task_loss.new_zeros(())
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

        loss = task_loss + (self.lambda_kl * kl_loss)
        self._task_loss_sum += float(task_loss.detach().cpu())
        self._kl_loss_sum += float(kl_loss.detach().cpu())
        self._total_loss_sum += float(loss.detach().cpu())
        self._loss_accum_count += 1

        step_for_log = int(self.state.global_step) + 1
        should_log_task_loss = (
            step_for_log % self.TASK_LOSS_LOG_INTERVAL == 0
            and step_for_log != self._last_task_loss_log_step
        )
        if should_log_task_loss:
            self._last_task_loss_log_step = step_for_log
            denom = max(1, self._loss_accum_count)
            self.log(
                {
                    "task_loss": self._task_loss_sum / denom,
                    "kl_loss": self._kl_loss_sum / denom,
                    "total_loss": self._total_loss_sum / denom,
                }
            )
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

    def _save(self, output_dir: str | None = None, state_dict=None):
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


class SupervisedSaveFallbackTrainer(Trainer):
    def __init__(self, *args, checkpoint_max_shard_size: str = "2GB", **kwargs):
        super().__init__(*args, **kwargs)
        self.checkpoint_max_shard_size = checkpoint_max_shard_size

    def _save(self, output_dir: str | None = None, state_dict=None):
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
        description="Full supervised fine-tuning on a single dataset; updates all model parameters."
    )
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help=(
            "HF dataset id, local JSON/JSONL path, or comma-separated list to concatenate. "
            "Also accepts aliases: AlpacaGPT4, CodeAlpaca, OpenMathInstruct, PubMedQA."
        ),
    )
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--output_path", type=Path, required=True)
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--num_train_epochs", type=float, default=1.0)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
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
    parser.add_argument("--dataloader_num_workers", type=int, default=4)
    parser.add_argument(
        "--dataloader_pin_memory",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--prompt_format",
        type=str,
        default="auto",
        choices=["auto", "plain", "instruct"],
        help=(
            "Prompt formatting mode. "
            "'instruct' uses tokenizer.apply_chat_template([{'role':'user',...}], add_generation_prompt=True). "
            "'auto' chooses instruct whenever tokenizer has a chat template."
        ),
    )
    parser.add_argument(
        "--system_message",
        type=str,
        default="You are a helpful assistant.",
        help="System message used when prompt_format=instruct.",
    )
    parser.add_argument(
        "--target_layers",
        type=str,
        default=None,
        help=(
            "Optional target decoder layer indices (comma/space separated), "
            "e.g. '23,25,28' or '23 25 28'. When set, selected layers are "
            "initialized with std=1e-3 before training."
        ),
    )
    parser.add_argument(
        "--target_layer_type",
        type=str,
        default="all",
        choices=["all", "attn", "ffn", "up_proj"],
        help=(
            "Module scope inside each target layer: all, attn (self_attn), "
            "ffn (mlp), or up_proj (mlp.up_proj only)."
        ),
    )
    parser.add_argument(
        "--freeze_target_layers",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "If set with --target_layers, do not reinitialize those layers; "
            "freeze them instead (requires_grad=False)."
        ),
    )
    parser.add_argument(
        "--reverse",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "If set with --target_layers, only target-layer params are trainable; "
            "all non-target params are frozen."
        ),
    )
    args = parser.parse_args()

    set_seed(args.seed)
    dtype = torch.bfloat16

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

    target_layers: list[int] = []
    target_init_std = 1e-3
    target_init_updated_tensors = 0
    frozen_target_param_tensors = 0
    reverse_trainable_target_param_tensors = 0
    reverse_frozen_non_target_param_tensors = 0
    if args.target_layers is not None:
        target_layers = parse_layer_indices(args.target_layers)
        if args.freeze_target_layers and args.reverse:
            raise ValueError("--freeze_target_layers and --reverse are mutually exclusive.")
        if args.freeze_target_layers:
            frozen_target_param_tensors = freeze_target_layers(
                model=model,
                target_layers=target_layers,
                layer_type=args.target_layer_type,
            )
        elif args.reverse:
            (
                reverse_trainable_target_param_tensors,
                reverse_frozen_non_target_param_tensors,
            ) = set_reverse_trainable_scope(
                model=model,
                target_layers=target_layers,
                layer_type=args.target_layer_type,
            )
        else:
            target_init_updated_tensors = initialize_target_layers(
                model=model,
                target_layers=target_layers,
                layer_type=args.target_layer_type,
                std=target_init_std,
            )

    model.train()
    if hasattr(model, "config") and hasattr(model.config, "use_cache"):
        model.config.use_cache = False
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    raw_train_dataset = load_supervised_dataset_multi(args.dataset, split=args.split)
    if args.max_train_samples is not None:
        raw_train_dataset = raw_train_dataset.select(
            range(min(args.max_train_samples, len(raw_train_dataset)))
        )
    if len(raw_train_dataset) == 0:
        raise ValueError("No rows available in training dataset.")

    train_dataset = raw_train_dataset.map(
        lambda ex: tokenize_single_example(
            ex,
            tokenizer=tokenizer,
            max_length=args.max_length,
            prompt_format=effective_prompt_format,
            system_message=args.system_message,
        ),
        remove_columns=raw_train_dataset.column_names,
    )
    debug_raw_row = raw_train_dataset[0]
    gold_prompt, gold_output = normalize_prompt_output(debug_raw_row)
    finetuned_prompt = decode_training_prompt_from_tokenized_row(
        tokenizer=tokenizer,
        row=train_dataset[0],
    )
    finetuned_output = decode_training_output_from_tokenized_row(
        tokenizer=tokenizer,
        row=train_dataset[0],
    )
    print(f"gold_prompt={gold_prompt}")
    print(f"finetuned_prompt={finetuned_prompt}")
    print(f"gold_output={gold_output}")
    print(f"finetuned_output={finetuned_output}")
    collator = SingleTargetCollator(tokenizer=tokenizer)

    use_bf16 = True
    use_fp16 = False
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
        save_steps=2000,
        bf16=use_bf16,
        fp16=use_fp16,
        report_to="none",
        gradient_checkpointing=args.gradient_checkpointing,
        dataloader_num_workers=args.dataloader_num_workers,
        dataloader_pin_memory=args.dataloader_pin_memory,
    )
    ta_params = inspect.signature(TrainingArguments.__init__).parameters
    if "save_total_limit" in ta_params:
        training_args_kwargs["save_total_limit"] = 1
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
    )
    trainer_kwargs["checkpoint_max_shard_size"] = args.max_shard_size
    try:
        trainer = SupervisedSaveFallbackTrainer(tokenizer=tokenizer, **trainer_kwargs)
    except TypeError as exc:
        if "unexpected keyword argument 'tokenizer'" not in str(exc):
            raise
        trainer = SupervisedSaveFallbackTrainer(processing_class=tokenizer, **trainer_kwargs)

    trainer.train()

    args.output_path.mkdir(parents=True, exist_ok=True)
    save_status = save_model_with_fallback(
        model=trainer.model,
        output_path=args.output_path,
        max_shard_size=args.max_shard_size,
    )
    tokenizer.save_pretrained(str(args.output_path))

    print(f"output_path={args.output_path}")
    print(f"dataset={args.dataset}")
    print(f"split={args.split}")
    print(f"num_training_examples={len(train_dataset)}")
    print(f"prompt_format={args.prompt_format}")
    print(f"effective_prompt_format={effective_prompt_format}")
    print(f"system_message={args.system_message}")
    print(f"target_layers={target_layers if target_layers else None}")
    print(f"target_layer_type={args.target_layer_type if target_layers else None}")
    print(
        "target_layer_init_std="
        f"{target_init_std if (target_layers and not args.freeze_target_layers) else None}"
    )
    print(f"target_layer_init_updated_tensors={target_init_updated_tensors}")
    print(f"freeze_target_layers={args.freeze_target_layers if target_layers else None}")
    print(f"frozen_target_param_tensors={frozen_target_param_tensors}")
    print(f"reverse={args.reverse if target_layers else None}")
    print(f"reverse_trainable_target_param_tensors={reverse_trainable_target_param_tensors}")
    print(f"reverse_frozen_non_target_param_tensors={reverse_frozen_non_target_param_tensors}")
    print(f"save_status={save_status}")


if __name__ == "__main__":
    main()
