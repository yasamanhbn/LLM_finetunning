import yaml
from model import ModelHandler

from vllm import LLM, SamplingParams
import pandas as pd


def generate_text(llm, tokenizer, sampling_params, user_request: str):
    # Build prompt
    if hasattr(tokenizer, "chat_template") and tokenizer.chat_template:
        messages = [
            {"role": "system", "content": "You are a friendly storyteller for children. Always tell short, safe, fun stories."},
            {"role": "user", "content": user_request},
        ]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        prompt = f"""
### System:
You are a friendly storyteller for children. Always tell short, safe, fun stories.

### Instruction:
{user_request}

### Response:
"""

    # Generate
    outputs = llm.generate(prompt, sampling_params)
    for output in outputs:
        generated_text = output.outputs[0].text.strip()
        print("✨ Generated Story:\n", generated_text)
        return generated_text

    

if __name__ == '__main__':
    # Load config
    with open("/content/LLM_Fintunning/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    base_model = config["base_model"]
    inference_params = config["inference"]
    model_params = config["model"]

    # Load Model and Tokenizer
    model_handler = ModelHandler(
        base_model=base_model,
        device_map="auto",
        tokenizer_trust_remote_code=True,
        inference=True,
        **model_params
    )

    tokenizer = model_handler.load_tokenizer()

    llm = LLM(model=inference_params["huggungFace_repo"], task="generate")

    sampling_params = SamplingParams(max_tokens=350, temperature=0.8, top_p=0.9)
    generate_text(llm, tokenizer, sampling_params, "Tell me a short story about a dog who learns to be brave.")

