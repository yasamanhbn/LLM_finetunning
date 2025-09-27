import pandas as pd
from datasets import Dataset
from typing import Optional


class TinyStoriesDatasetHandler:
    def __init__(
        self,
        path: str,
        dataset_type: str = "train",   # "train", "valid", or "test"
        file_type: str = "csv",
        text_column: str = "text",      # TinyStories only has `text`
        output_column: str = "text_fmt",# formatted for trainer
        system_prompt: Optional[str] = None,
        tokenizer=None,
    ):
        """
        Handler for TinyStories dataset: wraps plain text stories into
        instruction-following or chat-based format for fine-tuning.
        """
        self.path = path
        self.dataset_type = dataset_type.lower()
        self.file_type = file_type
        self.text_column = text_column
        self.output_column = output_column
        self.system_prompt = (
            system_prompt
            or "You are a friendly storyteller for children. Always tell short, safe, fun stories."
        )
        self.tokenizer = tokenizer

        self.dataset = None
        self.load_and_process_dataset()

    def format_chat_template(self, row: dict) -> dict:
        """
        Format a TinyStories example as chat messages if tokenizer supports chat_template,
        otherwise fall back to instruction-style text prompt.
        """
        story_text = row[self.text_column]

        if hasattr(self.tokenizer, "chat_template") and self.tokenizer.chat_template:
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": "Tell me a short story."},
            ]

            # For training, include the story as assistant response
            if self.dataset_type != "test":
                messages.append({"role": "assistant", "content": story_text})

            row[self.output_column] = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=(self.dataset_type == "test"),
            )
        else:
            # Fallback for non-chat models
            prompt = f"""
### System:
{self.system_prompt}

### Instruction:
Tell me a short story.

### Response:
{story_text if self.dataset_type != "test" else ""}
"""
            row[self.output_column] = prompt.strip()

        return row

    def load_and_process_dataset(self):
        """
        Load TinyStories, convert to HuggingFace dataset, and format each row.
        """
        if self.file_type == "csv":
            df = pd.read_csv(self.path)
        elif self.file_type == "json":
            df = pd.read_json(self.path, lines=True)
        else:
            raise ValueError(f"Unsupported file type: {self.file_type}")

        dataset = Dataset.from_pandas(df)
        dataset = dataset.map(self.format_chat_template, num_proc=4)

        self.dataset = dataset

    def get_dataset(self) -> Dataset:
        """Return the processed dataset."""
        return self.dataset