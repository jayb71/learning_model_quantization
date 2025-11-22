**Project Title**
Learning Model Quantization & QLoRA Adapters for BERT

**Overview**
- **Purpose:** This repository collects artifacts, experiments, and utilities for exploring model quantization and low-rank adapter (QLoRA-style) fine-tuning applied to BERT-based models. It is intended as a compact research / engineering workspace for evaluating performance/accuracy trade-offs when compressing language models for inference on constrained hardware.
- **Primary Focus:** Quantization (post-training and quantization-aware fine-tuning), QLoRA adapters, and adapters saved as `safetensors` for lightweight deployment.

**Repository Structure**
- **`bert-qlora/`**: Checkpoint directories produced by QLoRA-style fine-tuning. Each checkpoint contains adapter config, safetensors adapter weights, training artifacts, and tokenizer files.
	- Example: `bert-qlora/checkpoint-1000/`, `bert-qlora/checkpoint-1500/`
- **`saved/`**: Saved artifacts used for evaluation and deployment.
	- `ptq_int8_model.pt`: a saved model converted via post-training quantization to int8 (PyTorch format)
	- `baseline-model/`: baseline full-precision model artifacts (e.g., `model.safetensors`, `config.json`)
	- `qat_bert_int8/`: artifacts from quantization-aware training experiments
	- `qlora-bert-adapter/`: adapter-only safetensors and adapter config used to apply low-cost fine-tuning

**Getting Started**
- **Prerequisites:**
	- Linux or macOS (Linux tested)
	- Python 3.8+ (3.10/3.11 recommended)
	- PyTorch (matching your CUDA or CPU environment). Example: `pip install torch --index-url https://download.pytorch.org/whl/torch_stable.html`
	- Transformers (Hugging Face) and `safetensors` for loading/saving adapter weights.
	- Optional: `bitsandbytes` for 8-bit optimizations, and `accelerate` for multi-GPU runs.

- **Quick install (example virtualenv)**

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install torch transformers safetensors datasets accelerate
# optional: pip install bitsandbytes
```

**Model Checkpoints & Artifacts**
- **`bert-qlora/checkpoint-*/`**: Adapter checkpoints produced by fine-tuning only the low-rank adapters while keeping the base model frozen. Each folder contains:
	- `adapter_config.json` — configuration describing adapter dimensions and merge strategy
	- `adapter_model.safetensors` — adapter-only weights (safe binary format)
	- training state (e.g., `trainer_state.json`, optimizer, scheduler)
	- tokenizer files: `tokenizer.json`, `vocab.txt`, `tokenizer_config.json`

- **`saved/ptq_int8_model.pt`**: Example of a model converted via post-training quantization (PTQ) to int8 and saved as a PyTorch checkpoint. This is intended for fast, low-memory inference.

**Usage Examples**

- Load baseline model (example):

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer

model = AutoModelForSequenceClassification.from_pretrained('path/to/baseline-model')
tokenizer = AutoTokenizer.from_pretrained('path/to/baseline-model')
```

- Apply a QLoRA-style adapter at inference (high level):

```python
from transformers import AutoModel, AutoTokenizer
from safetensors.torch import load_file as load_safetensors

base = AutoModel.from_pretrained('path/to/baseline-model')
adapter_state = load_safetensors('qlora-bert-adapter/adapter_model.safetensors')
# Code to merge or wire the adapter into the base model depends on the adapter implementation used in these experiments.
```

- Load an int8 PTQ model saved as `ptq_int8_model.pt` for inference:

```python
import torch

state = torch.load('saved/ptq_int8_model.pt', map_location='cpu')
# Typical flow: load state dict into model architecture and run evaluation
```

**Quantization Experiments & Methodology**
- **Post-Training Quantization (PTQ):** Apply quantization to a pre-trained model after training completes. PTQ is fast and convenient but may reduce accuracy without calibration or per-channel quantization.
- **Quantization-Aware Training (QAT):** Simulates quantization effects during training to preserve accuracy. Artifacts from QAT experiments can be found in `saved/qat_bert_int8/`.
- **QLoRA Adapters:** Low-rank adapters (LORA-style) trained while freezing the base model. Adapter weights stored in `safetensors` allow fast, memory-light fine-tuning and distribution.

**Recommendations & Tips**
- For inference on CPU, prefer per-channel quantization and calibration with representative data.
- Use `safetensors` for adapter-only checkpoints to avoid pickle-level risks and reduce load times.
- When combining adapters with quantized backbones, validate end-to-end accuracy on representative dev sets.

**Training & Evaluation Notes**
- The repository contains example checkpoints but may not include every training script. If you replicate experiments:
	- Keep a small evaluation set to track accuracy/latency trade-offs
	- Profile memory and latency with `torch.profiler` or simple timing harnesses

**Troubleshooting**
- **OOM errors during load:** Try `map_location='cpu'` and then move model parts to device incrementally.
- **Accuracy drop after quantization:** Try calibration with a representative dataset, or switch to QAT.
- **Adapter application errors:** Confirm adapter config matches the base architecture (vocab size, hidden size, layers).

**Contributing**
- Contributions are welcome. Typical contribution types:
	- New quantization scripts or improved conversion utilities
	- Evaluation harnesses measuring latency, memory, and accuracy
	- Additional adapters or model variants

To contribute:

1. Fork the repo
2. Create a feature branch
3. Open a pull request describing your changes and experiments

**License & Attribution**
- This project does not include a license file in the repository root. Add a `LICENSE` file if you plan to open-source the code. For redistributed model weights, follow the original model's license terms.

**Contact / Maintainer**
- Maintainer: repository owner (see repo metadata). For questions about specific checkpoints, open an Issue describing reproducible steps and example commands.

**Appendix: Common Commands**

- Install dependencies (example):

```bash
pip install -r requirements.txt
# or manually: pip install torch transformers safetensors accelerate
```

- Example: run a quick inference script

```bash
python scripts/infer.py --model saved/baseline-model --input "Example text"
```

--

If you'd like, I can also:
- add a `requirements.txt` or `environment.yml` capturing exact package versions used here
- create minimal example scripts under `scripts/` for loading adapters and quantized models

Request which follow-up you'd like and I'll add it.

