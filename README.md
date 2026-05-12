# Gemma Fine-Tuning

Run `gemma-fine-tune.ipynb` to fine-tune the model, then download the fine-tuned model weights.

This notebook can be run either locally or on Kaggle. Kaggle is usually faster.
## Steps

1. Choose where to run:
	- Local: open `gemma-fine-tune.ipynb` in Jupyter or VS Code.
	- Kaggle (faster): upload/open `gemma-fine-tune.ipynb` in a Kaggle Notebook.
2. Run all cells to fine-tune the model.
3. After training completes, download/save the generated model weights.

## Files

- `gemma-fine-tune.ipynb`: Notebook used to fine-tune the model.
- `gemma_lora.weights.h5`: Fine-tuned LoRA weights file.

## LangChain Coding Agent (Local Gemma + LoRA)

This repository now includes a local coding agent that uses LangChain with Gemma and your fine-tuned LoRA weights.

### Added files

- `coding_agent.py`: LangChain coding agent backed by `keras_nlp.models.GemmaCausalLM`.
- `requirements.txt`: Python dependencies.
- `CODING_AGENT_PRODUCT_DESIGN_DOC.md`: Formal design document for the coding agent.

### Setup

```powershell
pip install -r requirements.txt
```

Note: Gemma presets may require Kaggle credentials configured in your environment when downloading base model assets.

### Run the coding agent

```powershell
python coding_agent.py --weights gemma_lora.weights.h5 --task "List files and propose a Python refactor task"
```

You can also run without arguments because defaults are defined in code:

```powershell
python coding_agent.py
```

Optional args:

- `--preset` (default: `gemma-keras-gemma_1.1_instruct_2b_en-v4`)
- `--lora-rank` (default: `16`)
- `--max-length` (default: `256`)
- `--temperature` (default: `0.2`)

Default values are in `coding_agent.py` and can be overridden with CLI args.


## Remote Gemma Endpoint (Colab + ngrok)

You can also run Gemma in Colab and expose a public endpoint (FastAPI + ngrok). A ready-to-run notebook is included:

- Upload and run [colab_inference_endpoint.ipynb](colab_inference_endpoint.ipynb) on Google Colab.
- The notebook downloads the base Gemma preset and your LoRA weights, starts a FastAPI server, and creates an ngrok tunnel.
- After running the notebook, copy the public ngrok URL printed by the notebook (it will look like `https://<name>.ngrok-free.dev`).


To run the coding agent against a remote Gemma HTTP endpoint, pass `--endpoint` with the full `/generate` URL:

```powershell
python coding_agent.py --endpoint "https:////<name>.ngrok-free.dev/generate" --task "List files and propose a Python refactor task"
```

Notes:
- The remote endpoint expects a JSON body with `prompt` and `max_length` and returns JSON with `text`.
- Running a publicly exposed model without authentication may be unsafe; add auth/rate-limiting for production use.

Environment file option:
- You can store the public URL in a `.env` file at the repository root as `GEMMA_ENDPOINT` and run without `--endpoint`:

```powershell
Set-Content -Path .env -Value 'GEMMA_ENDPOINT="https://vehicular-grueling-yodel.ngrok-free.dev/generate"' -Encoding utf8
python coding_agent.py --task "List files and propose a Python refactor task"
```

## Prompt Engineering Notes

The repository also includes a prompt library under `LLM_prompt/` that captures the prompts used to shape the coding-agent design, dataset selection, and documentation drafts.

The folder now includes a dedicated README and several expanded prompt templates so the design process is easier to reproduce and review.

- `LLM_prompt/`: Prompt artifacts used during project planning and design.