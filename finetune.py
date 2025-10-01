import yaml
from model import ModelHandler
from dataset import TinyStoriesDatasetHandler
from trainer import TrainerHandler, PushToHubCallback

if __name__ == '__main__':
    # Load config
    with open("/content/LLM_Fintunning/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    base_model = config["base_model"]
    training_params = config["training"]
    model_params = config["model"]

    # Generate a dynamic model name based on changed hyperparameters
    model_name = f"Llama1B_storyTelling"
    print(f"🔹 Model Name: {model_name}")

    # Load Model and Tokenizer
    model_handler = ModelHandler(
        base_model=base_model,
        device_map="auto",
        tokenizer_trust_remote_code=True,
        **model_params
    )

    model, tokenizer = model_handler.get_model_and_tokenizer()
    peft_config = model_handler.get_peft_config()

    # Load Datasets
    train_dataset = TinyStoriesDatasetHandler(
        dataset_split="train",
        text_column="text",
        output_column="text_fmt",
        tokenizer=tokenizer,
    ).get_dataset()

    valid_dataset = TinyStoriesDatasetHandler(
        dataset_split="test",
        text_column="text",
        output_column="text_fmt",
        tokenizer=tokenizer,
    ).get_dataset()

    # Train Model
    trainer_handler = TrainerHandler(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        tokenizer=tokenizer,
        peft_config=peft_config,
        **training_params
    )

    trainer_handler.trainer.add_callback(
        PushToHubCallback(
            model=model,
            tokenizer=tokenizer,
            model_par_name=model_name,
            total_epoch = training_params['num_train_epochs'],
            hub_token= training_params['HF_Token']
        )
    )

    results = trainer_handler.train()