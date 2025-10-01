import torch
import logging
import subprocess
from typing import List, Optional, Tuple

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model
import bitsandbytes as bnb

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelHandler:
    """
    A utility class to load, configure, and fine-tune LLMs with optional quantization and LoRA.
    """

    def __init__(
        self,
        base_model: str,
        inference=False,
        load_quantized: Optional[int] = None,  # Choose from [None, 4, 8]
        device_map: str = "auto",
        tokenizer_trust_remote_code: bool = True,
        linear_modules: Optional[List[str]] = None,
        use_lora: bool = True,
        lora_rank: int = 32,
        lora_dropout: float = 0.05,
    ):
        self.base_model = base_model
        self.load_quantized = load_quantized
        self.device_map = device_map
        self.tokenizer_trust_remote_code = tokenizer_trust_remote_code

        self.use_lora = use_lora
        self.lora_rank = lora_rank
        self.lora_dropout = lora_dropout

        self.model = None
        self.tokenizer = None
        self.lora_config = None
        self.linear_modules = linear_modules

        # Determine dtype and attention implementation
        self.torch_dtype, self.attn_implementation = self._set_attention_config()

        if inference:
            self.load_tokenizer()
        
        else:
            self.load_model_and_tokenizer()

    # ---------------- Tokenizer ----------------
    def load_tokenizer(self):
        """Load the tokenizer and ensure a pad token exists."""
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.base_model, trust_remote_code=self.tokenizer_trust_remote_code
        )
        if not self.tokenizer.pad_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        logger.info("Tokenizer loaded successfully.")
        return self.tokenizer

    # ---------------- Model ----------------
    def load_model_and_tokenizer(self):
        """Load the model (with optional quantization and LoRA) and tokenizer."""
        if not self.tokenizer:
            self.load_tokenizer()

        quantization_config = self._get_quantization_config()

        self.model = AutoModelForCausalLM.from_pretrained(
            self.base_model,
            device_map=self.device_map,
            torch_dtype=self.torch_dtype,
            quantization_config=quantization_config,
            attn_implementation=self.attn_implementation,
        )
        logger.info("Base model loaded successfully.")

        if self.use_lora:
            self.linear_modules = (
                self._get_linear_modules() if self.linear_modules is None else self.linear_modules
            )
            self.lora_config = LoraConfig(
                r=self.lora_rank,
                lora_alpha=2 * self.lora_rank,
                lora_dropout=self.lora_dropout,
                target_modules=self.linear_modules,
                bias="none",
                task_type="CAUSAL_LM",
            )
            self.model = get_peft_model(self.model, self.lora_config)
            logger.info("LoRA configuration applied.")

        return self.model, self.tokenizer

    # ---------------- Internal Helpers ----------------
    def _find_all_linear_names(self) -> List[str]:
        """Finds all 4-bit linear module names in the model for LoRA injection."""
        cls = bnb.nn.Linear4bit
        lora_module_names = set()
        for name, module in self.model.named_modules():
            if isinstance(module, cls):
                names = name.split('.')
                lora_module_names.add(names[-1])
        lora_module_names.discard("lm_head")  # avoid output head
        return list(lora_module_names)

    def _get_linear_modules(self) -> List[str]:
        """Return target linear modules for LoRA.
        This method scans your loaded model’s modules and collects the names of actual linear layers 
        that are quantized by bitsandbytes (bnb.nn.Linear4bit).
        It returns a dynamic list of module names based on the model architecture (e.g., could return ['q_proj', 'v_proj', 'up_proj', ...] 
        for LLaMA, but might look different for GPT-J or Falcon).
        """

        modules = self._find_all_linear_names()
        if modules:
            return modules

        """
        This is a hardcoded list of common projection layers in transformer architectures, especially LLaMA-family models.
        If _find_all_linear_names() finds nothing (e.g., when bitsandbytes isn’t being used or the architecture isn’t recognized), 
        the code falls back to this default list.
        """
        return [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ]

    def _get_quantization_config(self) -> Optional[BitsAndBytesConfig]:
        """Return quantization config if requested."""
        if self.load_quantized == 4:
            logger.info("Loading in 4-bit quantization mode.")
            return BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=self.torch_dtype,
                bnb_4bit_use_double_quant=True,
            )
        elif self.load_quantized == 8:
            logger.info("Loading in 8-bit quantization mode.")
            return BitsAndBytesConfig(load_in_8bit=True)
        return None

    def _set_attention_config(self) -> Tuple[torch.dtype, str]:
        """
        Determine best attention implementation and dtype based on GPU capability.
        Installs FlashAttention if supported.
        """
        if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8:
            try:
                subprocess.run(
                    ["pip", "install", "-qqq", "flash-attn"],
                    check=True,
                    capture_output=True,
                )
                logger.info("FlashAttention installed. Using flash_attention_2.")
                return torch.bfloat16, "flash_attention_2"
            except subprocess.CalledProcessError:
                logger.warning("Failed to install flash-attn. Falling back to eager.")
                return torch.float16, "eager"
        return torch.float16, "eager"

    # ---------------- Public API ----------------
    def get_model_and_tokenizer(self):
        return self.model, self.tokenizer

    def get_tokenizer(self):
        return self.tokenizer

    def get_model(self):
        return self.model

    def get_peft_config(self):
        return self.lora_config