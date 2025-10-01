import logging
from typing import Optional

import torch
from trl import SFTTrainer, SFTConfig
from transformers import TrainerCallback

from transformers import AutoModelForCausalLM
from peft import PeftModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PushToHubCallback(TrainerCallback):
    """
    Callback to push the model and tokenizer to the Hugging Face Hub
    at the end of each training epoch.
    """

    def __init__(self, model, tokenizer, total_epoch, model_par_name: str = "", organization: Optional[str] = None, hub_token: Optional[str] = None):
        self.model_par_name = model_par_name
        self.model = model
        self.tokenizer = tokenizer
        self.organization = organization
        self.hub_token = hub_token
        self.total_epoch = total_epoch

    def on_epoch_end(self, args, state, control, **kwargs):
        epoch = str(int(state.epoch))
        model_name = f"{self.model_par_name}_Epoch{epoch}"

        if int(self.total_epoch)-1 == int(epoch):
            # Merge LoRA into base weights
            merged_model = self.model.merge_and_unload()

            if self.model is None or self.tokenizer is None:
                raise RuntimeError("Model or tokenizer not provided to PushToHubCallback.")

            logger.info(f"Pushing model and tokenizer to Hugging Face Hub: {model_name}")
            merged_model.push_to_hub(model_name, token=self.hub_token, organization=self.organization)
            self.tokenizer.push_to_hub(model_name, token=self.hub_token, organization=self.organization)
            logger.info("Model and tokenizer successfully pushed to Hub.")


class TrainerHandler:
    """
    Handles setup and training of a model using TRL's SFTTrainer.
    """

    def __init__(
        self,
        model,
        train_dataset,
        eval_dataset,
        tokenizer,
        peft_config=None,
        output_dir: str = "tuned_model",
        per_device_train_batch_size: int = 1,
        per_device_eval_batch_size: int = 1,
        gradient_accumulation_steps: int = 2,
        optim: str = "paged_adamw_32bit",
        num_train_epochs: int = 5,
        evaluation_strategy: str = "steps",
        eval_steps: float = 0.5,
        logging_steps: int = 1,
        warmup_ratio: float = 0.03,
        lr_scheduler_type: str = "cosine",
        logging_strategy: str = "steps",
        learning_rate: float = 2e-5,
        fp16: bool = False,
        bf16: bool = False,
        group_by_length: bool = True,
        packing: bool = False,
        max_seq_length: int = 1024,
        dataset_text_field: str = "text_fmt",
        report_to: Optional[str] = None,  # e.g., "wandb", "tensorboard", or None
        save_total_limit: int = 2,
    ):
        self.model = model
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.tokenizer = tokenizer
        self.peft_config = peft_config

        # Define training arguments
        self.training_arguments = SFTConfig(
            report_to=report_to,
            output_dir=output_dir,
            per_device_train_batch_size=per_device_train_batch_size,
            per_device_eval_batch_size=per_device_eval_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            optim=optim,
            num_train_epochs=num_train_epochs,
            evaluation_strategy=evaluation_strategy,
            eval_steps=eval_steps,
            logging_steps=logging_steps,
            lr_scheduler_type=lr_scheduler_type,
            warmup_ratio=warmup_ratio,
            logging_strategy=logging_strategy,
            learning_rate=learning_rate,
            fp16=fp16,
            bf16=bf16,
            group_by_length=group_by_length,
            packing=packing,
            dataset_text_field=dataset_text_field,
            max_seq_length=max_seq_length,
            save_total_limit=save_total_limit,
        )

        # Initialize trainer
        self.trainer = SFTTrainer(
            model=self.model,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            peft_config=self.peft_config,
            processing_class=self.tokenizer,
            args=self.training_arguments,
        )

    def train(self):
        """Run model training."""
        logger.info("Starting training...")
        return self.trainer.train()