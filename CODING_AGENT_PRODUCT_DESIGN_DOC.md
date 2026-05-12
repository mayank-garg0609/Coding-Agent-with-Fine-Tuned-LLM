# Coding Agent Product Design Document

## Problem Statement

Developers need a lightweight local coding agent that can help inspect repositories, suggest refactors, explain code, and draft small implementation changes without requiring a large hosted model.

The goal of this project is to fine-tune a compact open model with LoRA or QLoRA so it performs reliably as a practical coding assistant in constrained compute environments such as Kaggle or a local GPU machine.

## Target Users and Use Cases

Primary users are individual developers, students, and researchers who want a low-cost coding assistant for repository-aware tasks.

Representative use cases include:
- summarizing a codebase or a folder structure
- suggesting refactors for small Python modules
- drafting new utility functions from existing patterns
- explaining code behavior in plain language
- generating review notes for pull requests or patches

## Product Goals

The system should produce useful coding guidance with limited latency and modest compute requirements.

The main goals are:
- stay small enough to run locally or in a notebook environment
- provide structured, code-aware responses
- support repeatable prompts and tool-assisted workflows
- remain easy to evaluate and improve over time

## Non-Goals

This project is not intended to replace a full-scale frontier model.

It does not aim to:
- solve arbitrary large-scale software engineering tasks end to end
- execute untrusted code automatically
- act as a production-grade autonomous coding platform
- guarantee perfect correctness for every generated suggestion

## Model and Training Approach

The base model should be a compact open model that is realistic for limited hardware.

The fine-tuning strategy should use LoRA or QLoRA so the project can adapt behavior without training the full model.

Training data should emphasize:
- coding instruction following
- repository-oriented tasks
- concise explanations of code changes
- safe refactoring and helper generation

The training run should keep sequence lengths and batch sizes modest, with checkpoints saved frequently enough to recover from interruptions.

## System Architecture

The system should have three layers:

1. Prompt and task input layer for user requests.
2. Model inference layer using the fine-tuned open model.
3. Optional orchestration layer in Python or LangChain for local tools and structured outputs.

This structure keeps the core model simple while allowing the surrounding application to handle file inspection, prompt shaping, and response post-processing.

## Data Flow and Inference Flow

The runtime flow should be:

1. Accept a coding task or repository question.
2. Normalize the prompt into a structured instruction.
3. Provide relevant context, such as file snippets or directory summaries.
4. Run the fine-tuned model to generate an answer.
5. Optionally apply a light formatter or validator.

If the system is used as a local agent, it should keep the model output separate from any tool execution so the user can inspect each step independently.

## Evaluation Strategy

Evaluation should combine automatic checks and human review.

Useful checks include:
- task completion quality
- correctness of code suggestions
- clarity of explanation
- consistency across repeated prompts
- improvement over the base model on the same tasks

The evaluation set should include easy, medium, and slightly ambiguous coding tasks so the model is tested on both direct instruction following and judgment-heavy responses.

## Constraints and Risks

The main constraints are compute limits, small model capacity, and noisy training data.

Key risks include:
- overfitting on a narrow prompt set
- generating plausible but incorrect code
- becoming too verbose or too generic
- weak performance on tasks outside the training distribution

These risks can be reduced with careful dataset curation, conservative training settings, and a compact evaluation suite.

## Rollout Plan

The rollout should happen in small steps:

1. Validate a baseline prompt on the base model.
2. Fine-tune with a small, focused dataset.
3. Compare outputs against the baseline on a fixed test set.
4. Add repository-aware prompting or LangChain orchestration.
5. Expand the dataset only after the smaller version is stable.

## Future Enhancements

Possible next steps include:
- code-aware retrieval for repository context
- stronger tool use for file inspection and patch planning
- structured diff generation
- multi-model comparison for quality control
- richer evaluation datasets for refactoring and debugging tasks