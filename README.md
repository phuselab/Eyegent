# Eyegent: Visual Attention Orchestration on Concepts

Eyegent is an active vision agent that performs concept-driven visual exploration on images. Given an image and a natural-language query, Eyegent orchestrates an iterative loop of segmentation, fixation prediction, and relevance evaluation to progressively focus on the regions most aligned with the query, producing a simulated human-like scanpath.

This repository contains the code released alongside the paper
*"Eyegent: Visual Attention Orchestration on Concepts"* (IJCV, 2026).

## Overview

Eyegent operates through a recurrent perception cycle:

1. **Concept-based segmentation** -- a segmentation backend (CLIPSeg by default, matching the paper's reported results; SAM 3 is available as an untested alternative) generates phrase-guided masks.
2. **Attention map construction** -- masks are assigned weights representing their expected relevance.
3. **Fixation selection** -- the Spatial Drift-Diffusion Gaze Model (SDDGM, `scanpath.model: ScanDDM` in `config.yml`; this is the model described in the paper) predicts the next fixation from the weighted spatial map. A separate Levy-flight model (`CLE`) is also wired in but was not used for the paper's results.
4. **Mask examination** -- the mask at the current fixation is visualized and scored by a multimodal LLM for how well it matches the query.
5. **Attention update** -- the score adjusts the mask's weight, influencing the next fixation.

The loop continues until a termination rule fires (confidence/coverage
threshold, or a fixed budget), producing a scanpath.

## Requirements

- Python 3.11 or 3.12. The commands below use 3.12.
- A running multimodal OpenAI compatible VLM server. The example below uses vLLM and Qwen3-VL-8B-Instruct (the approximately 9B class model).
- CPU or a CUDA GPU for CLIPSeg and ScanDDM. The VLM typically needs a CUDA GPU with enough memory for its checkpoint and context length; requirements depend on the model and vLLM settings.
- Internet access on the first run to download the CLIPSeg checkpoint, unless it is already in the Hugging Face cache.

## Install Eyegent

From the root of this release:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

`pip` installs the Python dependencies, including PyTorch. For a particular CUDA build of PyTorch, follow the [official PyTorch installer](https://pytorch.org/get-started/locally/) in this environment before `pip install -e .`. CPU execution needs no CUDA setup. CLIPSeg weights are downloaded automatically from Hugging Face on first use; no checkpoint path or authentication is needed for that public model.

Hugging Face may warn about anonymous requests while downloading public checkpoints. Authentication is optional; set `HF_TOKEN` if you need higher Hub rate limits. Recent Transformers versions may also report two unexpected CLIPSeg `position_ids` buffers when loading the checkpoint. Those buffers are recreated by the model and do not indicate missing weights.

## Serve the VLM with vLLM

Use a **separate terminal and Python environment** for the VLM server. Installing vLLM in its own environment avoids mixing its dependencies with the simulation environment.

```bash
python3.12 -m venv .venv-vllm
source .venv-vllm/bin/activate
python -m pip install --upgrade pip
python -m pip install vllm
vllm serve Qwen/Qwen3-VL-8B-Instruct \
  --host 127.0.0.1 \
  --port 8001 \
  --tensor-parallel-size 1 \
  --max-model-len 16384 \
  --limit-mm-per-prompt '{"image": 3}'
```

The model name in this command matches `llm.model` in `config.yml`. Eyegent sends the original image and up to two mask views in a single VLM request, so the server must accept at least three images per prompt. Adjust `--tensor-parallel-size` and memory settings to match your hardware. The default endpoint is `http://127.0.0.1:8001/v1`; check the server in another terminal with:

```bash
curl http://127.0.0.1:8001/v1/models
```

See the [vLLM serving guide](https://docs.vllm.ai/en/latest/serving/openai_compatible_server/) and [Qwen3-VL model card](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct) for server and model details. vLLM downloads the Qwen checkpoint on its first launch. If you already have an OpenAI compatible server, set `--base-url` and `--model` below to its endpoint and served model identifier; set `EYEGENT_API_KEY` when required.

## Run a simulation

In the Eyegent environment, while the VLM server is running:

```bash
source .venv/bin/activate
python simulate.py --image examples/ski.jpg \
  --query "What color is the ski the letters are on?" \
  --output scanpath.png \
  --seed 42
```

Open `scanpath.png` to see the numbered fixations over the image.

For your own stimulus and query, replace the `--image` and `--query` values. The image can be PNG, JPEG, or another format Pillow accepts. `--output` accepts PNG, PDF, SVG, or JPEG. Only the requested plot is kept; masks and VLM views needed during the loop are made in a temporary directory and removed afterward.

## Configuration

The defaults are in [`config.yml`](config.yml). When run from the repository root, the CLI loads this file automatically; pass `--config path/to/config.yml` to use a different file. When installed and run from elsewhere, built-in defaults apply unless `--config` is set. Omitted keys retain their defaults. Command line options take precedence over the file. `EYEGENT_BASE_URL`, `EYEGENT_MODEL`, and `EYEGENT_API_KEY` provide connection settings when command line options are absent.

| Section | Purpose |
| --- | --- |
| `segmentation` | CLIPSeg probability threshold, component area, ellipse option, maximum masks, device |
| `llm` | Model identifier, endpoint, API key, generation settings, original thinking budgets |
| `inference` | Maximum number of orchestrator tool calls, simulated duration, and judge confidence threshold |
| `scanpath` | Original ScanDDM parameters and device |

`inference.max_generations` counts completed orchestrator tool calls (`segment_phrase`, `predict_foa`, `examine_mask`, `judge`), not plotted fixations. A judge call below the confidence threshold can also sample the next fixation within the same generation. The plotted scanpath retains the original duration association, so its fixation count can differ from the number of orchestrator tool calls.

For example, to use a different VLM endpoint without editing a file:

```bash
python simulate.py --image my_image.jpg --query "Where is the red ball?" \
  --base-url http://127.0.0.1:8001/v1 --model Qwen/Qwen3-VL-8B-Instruct \
  --output ball_scanpath.png
```

The original VLM request includes the configured thinking fields. If your chosen server or model rejects those fields, set `llm.thinking.enabled: false` in a custom YAML file. This is a server compatibility option; it changes the VLM request settings.

## Citation

```bibtex
@article{damelio2026eyegent,
  author  = {D’Amelio, Alessandro and Patania, Sabrina and Annese, Luca and Boccignone, Giuseppe},
  title   = {Eyegent: Visual Attention Orchestration on Concepts},
  journal = {International Journal of Computer Vision},
  year    = {2026},
  month   = oct,
  volume  = {134},
  number  = {11},
  pages   = {493},
  issn    = {1573-1405},
  doi     = {10.1007/s11263-026-03042-4},
}
```
