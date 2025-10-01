from datasets import load_dataset, Dataset
from typing import Optional


class TinyStoriesDatasetHandler:
    def __init__(
        self,
        dataset_name: str = "roneneldan/TinyStories",
        dataset_split: str = "train",   # "train", or "test"
        text_column: str = "text",      # TinyStories only has `text`
        output_column: str = "text_fmt",# formatted for trainer
        system_prompt: Optional[str] = None,
        tokenizer=None,
    ):
        """
        Handler for TinyStories dataset: wraps plain text stories into
        instruction-following or chat-based format for fine-tuning.
        """
        self.dataset_name = dataset_name
        self.dataset_split = dataset_split.lower()
        self.text_column = text_column
        self.output_column = output_column
        self.system_prompt = (
            system_prompt
            or "You are a friendly storyteller for children. Always tell short, safe, fun stories."
        )
        self.tokenizer = tokenizer

        self.dataset = None
        self.load_and_process_dataset()

    def format_chat_template(self, example: dict) -> dict:
        """
        Format a TinyStories example as chat messages if tokenizer supports chat_template,
        otherwise fall back to instruction-style text prompt.
        """
        story_text = example[self.text_column]

        if hasattr(self.tokenizer, "chat_template") and self.tokenizer.chat_template:
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": "Tell me a short story."},
            ]

            # For training, include the story as assistant response
            if self.dataset_split != "test":
                messages.append({"role": "assistant", "content": story_text})

            example[self.output_column] = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=(self.dataset_split == "test"),
            )
        else:
            # Fallback for non-chat models
            prompt = f"""
### System:
{self.system_prompt}

### Instruction:
Tell me a short story.

### Response:
{story_text if self.dataset_split != "test" else ""}
"""
            example[self.output_column] = prompt.strip()

        return example

    def load_and_process_dataset(self):
        """
        Load TinyStories from Hugging Face Hub and format each row.
        """
        if self.dataset_split=="test":
          split="validation"
        else:
          split="train"
        dataset = load_dataset(self.dataset_name, split=split)
        
        dataset = dataset.map(self.format_chat_template, num_proc=1)
        self.dataset = dataset

    def get_dataset(self) -> Dataset:
        """Return the processed dataset."""
        return self.dataset