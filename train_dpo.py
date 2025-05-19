import logging
from dataclasses import dataclass
from typing import Optional

import torch
from peft import LoraConfig
from datasets import disable_caching, load_dataset, concatenate_datasets
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    HfArgumentParser,
    BitsAndBytesConfig,
)
from trl import DPOConfig, DPOTrainer

disable_caching()

logger = logging.getLogger(__name__)


@dataclass
class DPOTrainingArguments:
    model_name_or_path: str
    data_files: list[str]
    eval_data_files: list[str] = None
    tokenizer_name_or_path: Optional[str] = None
    use_fast: bool = True
    additional_special_tokens: list[str] = None
    max_seq_length: int = 2048
    load_in_8bit: bool = False
    load_in_4bit: bool = False
    use_flash_attention_2: bool = False
    use_peft: bool = False
    peft_target_model: Optional[str] = "llm-jp"
    peft_target_modules: Optional[list[str]] = None
    peft_lora_r: int = 8
    peft_lora_alpha: int = 32
    peft_lora_dropout: float = 0.05

    def __post_init__(self):
        if self.load_in_8bit and self.load_in_4bit:
            raise ValueError("load_in_8bit and load_in_4bit are mutually exclusive")
        if self.peft_target_model and self.peft_target_modules is None:
            if self.peft_target_model == "llm-jp":
                self.peft_target_modules = ["c_attn", "c_proj", "c_fc"]
            elif self.peft_target_model == "llama":
                # https://github.com/serp-ai/LLaMA-8bit-LoRA/blob/main/finetune_peft_8bit.py
                self.peft_target_modules = [
                    "q_proj",
                    "k_proj",
                    "v_proj",
                    "o_proj",
                    "gate_proj",
                    "up_proj",
                    "down_proj",
                ]
            elif self.peft_target_model == "llama-all":
                # https://note.com/kan_hatakeyama/n/ncd09c52d26c7
                self.peft_target_modules = [
                    "q_proj",
                    "k_proj",
                    "v_proj",
                    "o_proj",
                    "gate_proj",
                    "up_proj",
                    "down_proj",
                    "lm_head",
                    "embed_tokens",
                ]
            else:
                logger.warning(
                    f"peft_target_model '{self.peft_target_model}' is not supported, "
                    f"so peft_target_modules is set to None."
                )

    def from_pretrained_kwargs(self, training_args):
        if self.load_in_8bit:
            kwargs = {"load_in_8bit": True}
        elif self.load_in_4bit:
            kwargs = {
                "quantization_config": BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.bfloat16,
                )
            }
        elif training_args.bf16:
            kwargs = {"torch_dtype": torch.bfloat16}
        else:
            kwargs = {"torch_dtype": torch.float16}
        kwargs["use_flash_attention_2"] = self.use_flash_attention_2
        return kwargs


def format_dpo_example(example, tokenizer):
    """Format example for DPO training"""
    # When chosen and rejected already include the prompt, we need to handle it differently
    
    if "chosen" in example and "rejected" in example:
        # The dataset already has chosen and rejected with prompts included
        # DPOTrainer expects the format where chosen/rejected include the full conversation
        return {
            "prompt": "",  # Empty prompt since it's included in chosen/rejected
            "chosen": example["chosen"],
            "rejected": example["rejected"],
        }
    elif "conversation" in example:
        # Legacy format - convert conversation to DPO format
        # This assumes we need to generate chosen/rejected from conversation
        # You may need to adjust based on your actual data structure
        prompt = example["conversation"][0]["content"] if example["conversation"] else ""
        response = example["conversation"][1]["content"] if len(example["conversation"]) > 1 else ""
        
        # For legacy format, we might need to construct full conversations
        full_conversation = tokenizer.apply_chat_template(example["conversation"], tokenize=False)
        
        return {
            "prompt": "",
            "chosen": full_conversation,
            "rejected": full_conversation,  # This would need actual rejected version
        }
    else:
        # Fallback for other formats
        return {
            "prompt": example.get("prompt", ""),
            "chosen": example.get("chosen", ""),
            "rejected": example.get("rejected", ""),
        }


def load_dpo_datasets(data_files, tokenizer):
    datasets = [] # multiple datasets
    
    for data_file in data_files:
        # Detect file format based on extension
        if data_file.endswith('.parquet'):
            dataset = load_dataset("parquet", data_files=data_file)
        elif data_file.endswith('.json') or data_file.endswith('.jsonl') or data_file.endswith('.jsonl.gz'):
            dataset = load_dataset("json", data_files=data_file)
        else:
            raise ValueError(f"Unsupported file format: {data_file}")
        
        dataset = dataset["train"]
        
        # Map to DPO format
        dataset = dataset.map(
            lambda x: format_dpo_example(x, tokenizer),
            remove_columns=dataset.column_names
        )
        
        datasets.append(dataset)
    
    return concatenate_datasets(datasets)


def main() -> None:
    parser = HfArgumentParser((DPOConfig, DPOTrainingArguments))
    dpo_config, dpo_training_args = parser.parse_args_into_dataclasses()

    tokenizer_name_or_path: str = (
        dpo_training_args.tokenizer_name_or_path or dpo_training_args.model_name_or_path
    )
    logger.info(f"Loading tokenizer from {tokenizer_name_or_path}")
    
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name_or_path,
        use_fast=dpo_training_args.use_fast,
        additional_special_tokens=dpo_training_args.additional_special_tokens,
        trust_remote_code=True,
    )
    
    # Ensure padding token is set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    logger.info("Loading data")

    train_dataset = load_dpo_datasets(dpo_training_args.data_files, tokenizer)
    
    if dpo_training_args.eval_data_files:
        eval_dataset = load_dpo_datasets(dpo_training_args.eval_data_files, tokenizer)
        dpo_config.eval_strategy = "steps"
    else:
        eval_dataset = None
         
    logger.info(f"Loading model from {dpo_training_args.model_name_or_path}")
    kwargs = dpo_training_args.from_pretrained_kwargs(dpo_config)
    logger.debug(
        f"AutoModelForCausalLM.from_pretrained({dpo_training_args.model_name_or_path}, trust_remote_code=True, **kwargs={kwargs})"
    )
    # Enable memory efficient attention if available
    if dpo_training_args.use_flash_attention_2:
        kwargs["attn_implementation"] = "flash_attention_2"
    
    model = AutoModelForCausalLM.from_pretrained(
        dpo_training_args.model_name_or_path,
        trust_remote_code=True,
        **kwargs,
    )
    
    # Load the reference model (same as the model for DPO)
    # The DPOTrainer will use the initial model weights as the reference
    model_ref = None  # DPOTrainer will create a reference model internally

    peft_config: Optional[LoraConfig] = None
    if dpo_training_args.use_peft:
        logger.info("Setting up LoRA")
        peft_config = LoraConfig(
            r=dpo_training_args.peft_lora_r,
            target_modules=dpo_training_args.peft_target_modules,
            lora_alpha=dpo_training_args.peft_lora_alpha,
            lora_dropout=dpo_training_args.peft_lora_dropout,
            fan_in_fan_out=True,
            bias="none",
            task_type="CAUSAL_LM",
        )
        if dpo_config.gradient_checkpointing:
            for param in model.parameters():
                param.requires_grad = False
                if param.ndim == 1:
                    param.data = param.data.to(torch.float32)
            model.gradient_checkpointing_enable()
            model.enable_input_require_grads()


    logger.info("Setting up trainer")
    # Set max_length in the DPOConfig
    dpo_config.max_length = dpo_training_args.max_seq_length
    
    trainer = DPOTrainer(
        model,
        model_ref,
        args=dpo_config,
        processing_class=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=peft_config,
    )

    logger.info("Training")
    trainer.train()

    logger.info("Saving model")
    trainer.save_model()


if __name__ == "__main__":
    main()