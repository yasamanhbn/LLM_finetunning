import yaml
from model_handler import ModelHandler
from dataset_handler import TinyStoriesDatasetHandler
from trainer_handler import TrainerHandler, PushToHubCallback

if __name__ == '__main__':
    # Load config
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    base_model = config["base_model"]
    training_params = config["training"]
    model_params = config["model"]

    # Generate a dynamic model name based on changed hyperparameters
    model_name = "{base_model}_storyTelling_"
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
        path="train.csv",
        dataset_type="train",
        text_column="text",
        output_column="text_fmt",
        tokenizer=tokenizer,
    ).get_dataset()

    valid_dataset = TinyStoriesDatasetHandler(
        path="valid.csv",
        dataset_type="valid",
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
            base_model=model,
            model_par_name=model_name,
        )
    )

    results = trainer_handler.train()