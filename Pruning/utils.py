import gc
import os
import random

import numpy as np
import torch
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    LlamaForCausalLM,
    LlamaTokenizer,
)


def set_seed(random_seed=1234):
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)


def set_model_device_evalmode(
    model, device, fix_decapoda_config=False, use_bfloat=False
):
    if "cuda" in device:
        model.half()
        model = model.to(device)

    if fix_decapoda_config:
        # unwind broken decapoda-research config
        model.config.pad_token_id = 0
        model.config.bos_token_id = 1
        model.config.eos_token_id = 2
    model.eval()

    if use_bfloat:
        model = model.bfloat16()

    gc.collect()
    torch.cuda.empty_cache()

    return model


def get_model(
    base_model=None,
    ckpt=None,
    lora_ckpt=None,
    tokenizer=None,
    model_type="pretrain",
    device="cuda",
    fix_decapoda_config=False,
    use_bfloat=False,
    hf_token=None,
):
    if model_type != "pretrain":
        raise NotImplementedError(
            "Pruning.utils.get_model now only supports model_type='pretrain' for Laco."
        )

    resolved_hf_token = hf_token if hf_token is not None else os.environ.get("HF_TOKEN")

    tokenizer = base_model if tokenizer is None else tokenizer
    config = AutoConfig.from_pretrained(base_model, token=resolved_hf_token)
    if "gptq" in base_model.lower():
        from auto_gptq import AutoGPTQForCausalLM

        model = AutoGPTQForCausalLM.from_quantized(
            base_model,
            use_safetensors=True,
            trust_remote_code=True,
            use_triton=False,
            quantize_config=None,
        )
        tokenizer = AutoTokenizer.from_pretrained(tokenizer, token=resolved_hf_token)
    elif (
        "LlamaForCausalLM" in config.__getattribute__("architectures")
        and "llama-3" not in base_model.lower()
    ):
        model = LlamaForCausalLM.from_pretrained(
            base_model, low_cpu_mem_usage=True, token=resolved_hf_token
        )
        tokenizer = LlamaTokenizer.from_pretrained(tokenizer, token=resolved_hf_token)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            base_model, low_cpu_mem_usage=True, token=resolved_hf_token
        )
        tokenizer = AutoTokenizer.from_pretrained(tokenizer, token=resolved_hf_token)

    description = "Model Type: {}\n Base: {} \n Pruned: {}\n LORA: {}".format(
        model_type, base_model, ckpt, lora_ckpt
    )

    if fix_decapoda_config:
        # unwind broken decapoda-research config
        tokenizer.pad_token_id = 0
    model = set_model_device_evalmode(model, device, fix_decapoda_config, use_bfloat)

    return model, tokenizer, description
