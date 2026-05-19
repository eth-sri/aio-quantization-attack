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
            required = {"input_ids", "attention_mask", "labels", "ref_log_probs"}
            if not required.issubset(set(shard.keys())):
                invalid.append(shard_path)
                continue
            valid.append(shard_path)
        except Exception:
            invalid.append(shard_path)
    return valid, invalid


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
        "model.language_model.layers, model.layers, model.model.layers, "
        "language_model.model.layers, language_model.layers, "
        "base_model.model.language_model.layers, base_model.model.layers, "
        "and recursive *.layers search"
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


def upper_left_masked_grad(grad: torch.Tensor, upper_left_range: int) -> torch.Tensor:
    if upper_left_range <= 0:
        raise ValueError("--upper_left_range must be >= 1")
    if grad is None:
        return grad
    if grad.ndim == 0:
        return grad
    masked = torch.zeros_like(grad)
    if grad.ndim == 1:
        n = min(int(grad.shape[0]), upper_left_range)
        if n > 0:
            masked[:n] = grad[:n]
        return masked
    n_rows = min(int(grad.shape[0]), upper_left_range)
    n_cols = min(int(grad.shape[1]), upper_left_range)
    if n_rows > 0 and n_cols > 0:
        masked[:n_rows, :n_cols] = grad[:n_rows, :n_cols]
    return masked


class UpperLeftGradientMasker:
    def __init__(
        self,
        model,
        target_layers: list[int],
        layer_type: str,
        enabled: bool,
        upper_left_range: int,
    ):
        self.enabled = bool(enabled)
        self.upper_left_range = int(upper_left_range)
        self._handles = []
        self._n_masked_params = 0
        if not self.enabled:
            return
        if self.upper_left_range <= 0:
            raise ValueError("--upper_left_range must be >= 1")

        layers = get_decoder_layers(model)
        n_layers = len(layers)
        for idx in target_layers:
            if idx < 0 or idx >= n_layers:
                raise ValueError(f"Layer index {idx} out of bounds for {n_layers} layers.")
            for module in get_target_modules_for_layer(layers[idx], layer_type):
                for parameter in module.parameters():
                    if not parameter.requires_grad:
                        continue
                    self._handles.append(
                        parameter.register_hook(
                            lambda grad, r=self.upper_left_range: upper_left_masked_grad(grad, r)
                        )
                    )
                    self._n_masked_params += 1

    @property
    def n_masked_params(self) -> int:
        return self._n_masked_params

    def close(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()


@torch.no_grad()
def initialize_target_layers(
    model,
    target_layers: list[int],
    layer_type: str,
    std: float,
    attack_upper_left: bool = False,
    upper_left_range: int = 512,
) -> int:
    if std < 0:
        raise ValueError("--target_layer_init_std must be >= 0")
    if std == 0:
        return 0
    if attack_upper_left and upper_left_range <= 0:
        raise ValueError("--upper_left_range must be >= 1")

    layers = get_decoder_layers(model)
    updated_param_tensors = 0
    for idx in target_layers:
        layer = layers[idx]
        for module in get_target_modules_for_layer(layer, layer_type):
            for p in module.parameters():
                if attack_upper_left:
                    if p.ndim == 0:
                        p.normal_(mean=0.0, std=std)
                    elif p.ndim == 1:
                        n = min(int(p.shape[0]), upper_left_range)
                        if n > 0:
                            p[:n].normal_(mean=0.0, std=std)
                    else:
                        n_rows = min(int(p.shape[0]), upper_left_range)
                        n_cols = min(int(p.shape[1]), upper_left_range)
                        if n_rows > 0 and n_cols > 0:
                            p[:n_rows, :n_cols].normal_(mean=0.0, std=std)
                else:
                    p.normal_(mean=0.0, std=std)
                updated_param_tensors += 1
    return updated_param_tensors


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
        self._supports_token_type_ids_cache: dict[int, bool] = {}

    def _supports_token_type_ids(self, model_obj) -> bool:
        key = id(model_obj)
        cached = self._supports_token_type_ids_cache.get(key)
        if cached is not None:
            return cached
        try:
            sig = inspect.signature(model_obj.forward)
            params = sig.parameters
            supports = ("token_type_ids" in params) or any(
                p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()
            )
        except Exception:
            supports = True
        self._supports_token_type_ids_cache[key] = supports
        return supports

    def _forward_model(
        self,
        model_obj,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
    ):
        kwargs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        if labels is not None:
            kwargs["labels"] = labels
        if token_type_ids is not None and self._supports_token_type_ids(model_obj):
            kwargs["token_type_ids"] = token_type_ids
        return model_obj(**kwargs)

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
        current_logits = self._forward_model(
            model,
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        ).logits

        if precomputed_ref_log_probs is None:
            if self.reference_model is None:
                return current_logits.new_zeros(())
            ref_device = next(self.reference_model.parameters()).device
            with torch.no_grad():
                reference_logits = self._forward_model(
                    self.reference_model,
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

    def _compute_task_loss(self, model, inputs) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        with self._trainable_subset(self.target_param_ids):
            loss_b = self._forward_model(
                model,
                input_ids=inputs["b_input_ids"],
                attention_mask=inputs["b_attention_mask"],
                token_type_ids=inputs.get("b_token_type_ids"),
                labels=inputs["b_labels"],
            ).loss

        with self._trainable_subset(self.non_target_param_ids):
            with self.layer_skipper.activate():
                loss_a = self._forward_model(
                    model,
                    input_ids=inputs["a_input_ids"],
                    attention_mask=inputs["a_attention_mask"],
                    token_type_ids=inputs.get("a_token_type_ids"),
                    labels=inputs["a_labels"],
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
        kl_loss = task_loss.new_zeros(())
        if self.lambda_kl > 0:
            kl_loss = self._compute_kl_loss_from_inputs(model, inputs)

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

    def training_step(self, model, inputs, num_items_in_batch=None):
        model.train()
        inputs = self._prepare_inputs(inputs)

        # Step 1a: B loss backward (target subset, skipper disabled).
        with self._trainable_subset(self.target_param_ids):
            loss_b = self._forward_model(
                model,
                input_ids=inputs["b_input_ids"],
                attention_mask=inputs["b_attention_mask"],
                token_type_ids=inputs.get("b_token_type_ids"),
                labels=inputs["b_labels"],
            ).loss
            weighted_b = self.loss_weight_b * loss_b
            loss_b_for_backward = weighted_b
            if self.args.gradient_accumulation_steps > 1:
                loss_b_for_backward = (
                    loss_b_for_backward / self.args.gradient_accumulation_steps
                )
            self.accelerator.backward(loss_b_for_backward)
            del loss_b_for_backward

        # Step 1b: A loss backward (non-target subset, skipper enabled in both
        # forward and checkpoint recompute during backward).
        with self._trainable_subset(self.non_target_param_ids):
            with self.layer_skipper.activate():
                loss_a = self._forward_model(
                    model,
                    input_ids=inputs["a_input_ids"],
                    attention_mask=inputs["a_attention_mask"],
                    token_type_ids=inputs.get("a_token_type_ids"),
                    labels=inputs["a_labels"],
                ).loss
                weighted_a = self.loss_weight_a * loss_a
                loss_a_for_backward = weighted_a
                if self.args.gradient_accumulation_steps > 1:
                    loss_a_for_backward = (
                        loss_a_for_backward / self.args.gradient_accumulation_steps
                    )
                self.accelerator.backward(loss_a_for_backward)
                del loss_a_for_backward

        task_loss = weighted_a + weighted_b

        # Step 2: KL as a separate backward pass over the full parameter set.
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

        return total_loss.detach()

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
    parser.add_argument(
        "--gradient_checkpointing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable gradient checkpointing to reduce activation memory usage.",
    )
    parser.add_argument(
        "--gradient_checkpointing_use_reentrant",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Use reentrant checkpointing. "
            "Set to false (default) for better compatibility with hooks/custom forward behavior."
        ),
    )
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
        default="auto",
        choices=["auto", "plain", "instruct"],
        help=(
            "Prompt formatting mode. "
            "'instruct' uses tokenizer.apply_chat_template([{'role':'user',...}], add_generation_prompt=True). "
            "'auto' chooses instruct whenever tokenizer has a chat template."
        ),
    )
    parser.add_argument(
        "--target_layer_init_std",
        type=float,
        default=1e-3,
        help="Gaussian std for re-initializing parameters in --layers before training (0 disables).",
    )
    parser.add_argument(
        "--attack_upper_left",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="If enabled, restrict target-layer updates/initialization to upper-left blocks.",
    )
    parser.add_argument("--upper_left_range", type=int, default=512)
    parser.add_argument(
        "--system_message",
        type=str,
        default="You are a helpful assistant.",
        help="System message used when prompt_format=instruct.",
    )
    args = parser.parse_args()

    set_seed(args.seed)
    target_layers = parse_layer_indices(args.layers)
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
        low_cpu_mem_usage=True,
    )
    initialized_target_param_tensors = initialize_target_layers(
        model=model,
        target_layers=target_layers,
        layer_type=args.layer_type,
        std=args.target_layer_init_std,
        attack_upper_left=args.attack_upper_left,
        upper_left_range=args.upper_left_range,
    )
    model.train()
    if hasattr(model, "config") and hasattr(model.config, "use_cache"):
        model.config.use_cache = False
    if args.gradient_checkpointing:
        gc_kwargs = {"use_reentrant": args.gradient_checkpointing_use_reentrant}
        try:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=gc_kwargs)
        except TypeError:
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
            low_cpu_mem_usage=True,
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
                valid_shards, invalid_shards = validate_precomputed_reference_shards(existing_shards)
                if invalid_shards:
                    print(f"warning=ignored_corrupted_precomputed_ref_shards count={len(invalid_shards)}")
                if valid_shards:
                    precomputed_reference_shard_paths = valid_shards
                    reference_model = None
                    print(f"using_existing_precomputed_ref_shards={len(valid_shards)}")
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
                except RuntimeError as exc:
                    msg = str(exc).lower()
                    if ("can't allocate memory" in msg) or ("out of memory" in msg) or ("defaultcpuallocator" in msg):
                        print(
                            "warning=precompute_ref_logprobs_disabled_due_to_memory_error "
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
    skipper = TargetLayerSkipper(
        model=model,
        target_layers=target_layers,
        layer_type=args.layer_type,
    )
    upper_left_grad_masker = UpperLeftGradientMasker(
        model=model,
        target_layers=target_layers,
        layer_type=args.layer_type,
        enabled=args.attack_upper_left,
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
        gradient_checkpointing=args.gradient_checkpointing,
        dataloader_num_workers=args.dataloader_num_workers,
        dataloader_pin_memory=args.dataloader_pin_memory,
    )
    ta_params = inspect.signature(TrainingArguments.__init__).parameters
    if "torch_empty_cache_steps" in ta_params:
        training_args_kwargs["torch_empty_cache_steps"] = 10
    if "gradient_checkpointing_kwargs" in ta_params:
        training_args_kwargs["gradient_checkpointing_kwargs"] = {
            "use_reentrant": args.gradient_checkpointing_use_reentrant
        }
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
        layer_skipper=skipper,
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
        skipper.close()
        upper_left_grad_masker.close()

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
    print(f"layer_type={args.layer_type}")
    print(f"attack_upper_left={args.attack_upper_left}")
    print(f"upper_left_range={args.upper_left_range}")
    print(f"upper_left_masked_params={upper_left_grad_masker.n_masked_params}")
    print(f"prompt_format={args.prompt_format}")
    print(f"effective_prompt_format={effective_prompt_format}")
    print(f"system_message={args.system_message}")
    print(f"target_layer_init_std={args.target_layer_init_std}")
    print(f"initialized_target_param_tensors={initialized_target_param_tensors}")
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
    print(f"save_status={save_status}")


if __name__ == "__main__":
    main()
