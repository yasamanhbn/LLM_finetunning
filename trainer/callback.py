import os
import time
import torch
import pandas as pd
from peft import PeftConfig, PeftModel
from trl import SFTTrainer, SFTConfig
from transformers import TrainingArguments, pipeline, TrainerCallback
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


class PushToHubCallback(TrainerCallback):
    def __init__(self, base_model, model_par_name='', organization=None):
        self.model_par_name = model_par_name #model name indicating parameters used for finetuning
        self.base_model = base_model

    def on_epoch_end(self, args, state, control, model=None, tokenizer=None, **kwargs):
        """
        Push the trained model to Hugging Face Hub and evaluate on validation dataset after each epoch.
        """
        epoch = str(int(state.epoch))
        model_name = f"{self.model_par_name}_{epoch}"
        if model is not None:
            print(f"Pushing the model to the Hugging Face Hub at {model_name}...")
            model.push_to_hub(model_name , token="YOUR_TOKEN")
            tokenizer.push_to_hub(model_name , token="YOUR_TOKEN")
                
        else:
            raise Exception("Error in saving model")
    
        print("Model saved in model directory!")