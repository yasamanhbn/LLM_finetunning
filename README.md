# LLM Fine-Tuning with LoRA & TinyStories with vLLM Inference

This repository demonstrates how to fine-tune large language models (LLMs) using **LoRA (Low-Rank Adaptation)** and **PEFT (Parameter-Efficient Fine-Tuning)** on the **TinyStories dataset**. The project supports Hugging Face integration, automated model pushes, and VLLM-based inference.

---

## Features

* Fine-tune LLMs efficiently with LoRA and PEFT.
* Support for **quantized models** (4-bit or 8-bit) via `bitsandbytes`.
* Handles both **chat-style prompts** and **instruction-style prompts**.
* Automatically pushes the model and tokenizer to Hugging Face Hub after each training epoch.
* Compatible with **TinyStories dataset** or custom CSV/JSON datasets.
* Supports **VLLM** for high-performance inference.

---

## Installation

Install the required packages using the provided `requirements.txt`. This ensures all dependencies are installed with compatible versions:

```
pip install -r requirements.txt
```

Make sure to include optional packages like `flash-attn` and `vllm` if you plan to use high-speed attention or VLLM inference.

---

## Dataset

This repository uses the **TinyStories dataset** from Hugging Face. It contains short, child-friendly stories.

The processed dataset includes two key columns:

* `text`: raw story text.
* `text_fmt`: formatted story text for instruction-following or chat-style fine-tuning.

Custom datasets in CSV or JSON format can also be used as long as they contain a `text` column.

---

## Configuration

All hyperparameters, model paths, and training settings are stored in `config.yaml`. This includes:

* Base model selection.
* LoRA parameters (rank, dropout, etc.).
* Training parameters (batch size, learning rate, epochs, FP16/BF16, max sequence length).
* Inference parameters (temperature, top-p, top-k).
* Output directories for trained and merged models.

---

## Training Workflow

1. **Load Model & Tokenizer**: The base model is loaded with optional LoRA and quantization.
2. **Prepare Dataset**: Raw stories are formatted for training, either as chat messages or instruction-style prompts.
3. **Initialize Trainer**: The `SFTTrainer` handles fine-tuning with PEFT support.
4. **Callbacks**: After final epoch, full merged model can be pushed to Hugging Face Hub.

---

## Inference

After training, you can use the merged model for story generation with the **VLLM** library for fast, high-throughput inference.

Prompts are typically short instructions like “Tell me a short story about a talking dog,” and the model generates child-friendly stories accordingly.
