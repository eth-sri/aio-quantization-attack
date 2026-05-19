# Code for Exploiting LLM Quantization via Outlier Injection

This repository includes code for:

- model editing/training pipelines
- post-training quantization
- evaluation
- plotting and analysis

## Environment

Use a recent Python environment 3.12 with GPU support.
Install dependencies from your local environment setup (for example, `requirements.txt`, if provided in your workflow).

For gated Hugging Face models, set your token before running:

```bash
export HF_TOKEN=YOUR_TOKEN
```

For evaluation services, set API environment variables before running evaluation scripts:

```bash
export OPENAI_API_KEY=YOUR_KEY
export OPENAI_BASE_URL=YOUR_BASE_URL
```

Before running evaluation, the installation of lm_eval is critical. Please follow steps of https://github.com/EleutherAI/lm-evaluation-harness/tree/main to perform installation.

Because lm_eval quantization backends are maintained independently from certain quantization packages that we also use, dependency conflicts can happen. If you hit version issues, we recommend reinstalling requirements with force. 
We recommend prioritizing lm_eval requirements, over requirements by the quantizaiton packages.

## Install Datasets

Install datasets from https://github.com/eth-sri/llm-pruning-attack/tree/main/dataset.
Note that in the config, dataset_a is the dataset of malicious data and dataset_b is safe.
When running the code the dataset must be paired and please follow the usage guidelines provided by the repo.

## Quick Start

The main training/editing pipeline is:

- `pipeline/run.py`

Run it with a config file:

```bash
python pipeline/run.py --config Config/Llama31-Ins_jailbreak.json
```

You can use any compatible config under `Config/`.

## Evaluation

The main evaluation pipeline is:

- `pipeline/evaluate.py`

Example:

```bash
python pipeline/evaluate.py \
  --model_path /path/to/model \
  --output_path /path/to/eval_output \
  --scenario jailbreak \
  --mode all
```

This script can evaluate utility metrics, run quantization variants, and run ASR-style checks.

## Helpful Entrypoints

- `pipeline/run.py` for staged pipeline execution
- `pipeline/evaluate.py` for evaluation + quantization benchmarking

## Notes

- Experiments were run on NVIDIA RTX PRO 6000 (96GB VRAM). If you use less VRAM, reduce batch size and sample count.
- Runtime depends heavily on selected quantization methods and evaluation scope; full evaluation on all quantization methods can take several hours. Quantizations that are not supported by vLLM runs extremely slowly.


If you find this work helpful, please cite:

```bibtex
@article{zhan2026widening,
  title={Widening the Gap: Exploiting LLM Quantization via Outlier Injection},
  author={Zhan, Xiaohua and Egashira, Kazuki and Staab, Robin and Vero, Mark and Vechev, Martin},
  journal={arXiv preprint arXiv:2605.15152},
  year={2026}
}
