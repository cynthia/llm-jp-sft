import torch
import logging
import transformers
from typing import Optional
from dataclasses import dataclass
from transformers.trainer_utils import set_seed
from datasets import load_dataset, concatenate_datasets

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    HfArgumentParser,
    BitsAndBytesConfig,
)
from trl import DPOConfig, DPOTrainer

logger = logging.getLogger(__name__)
transformers.logging.set_verbosity_info()


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

@dataclass
class DPOTrainingArguments:
    model_name_or_path: str
    data_files: list[str]
    eval_data_files: list[str] = None
    tokenizer_name_or_path: Optional[str] = None
    use_fast: bool = True
    additional_special_tokens: list[str] = None
    max_seq_length: int = 4096
    preprocessing_num_workers: int = 8
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

    

def load_dpo_datasets(files, tokenizer):
    datasets = [] # multiple datasets
    
    for data_file in files:
        # Detect file format based on extension
        if data_file.endswith('.parquet'):
            dataset = load_dataset("parquet", data_files=data_file)
        elif data_file.endswith('.json') or data_file.endswith('.jsonl') or data_file.endswith('.jsonl.gz'):
            dataset = load_dataset("json", data_files=data_file)
        else:
            raise ValueError(f"Unsupported file format: {data_file}")
        
        dataset = dataset["train"]
        datasets.append(dataset)
    
    # Concatenate all datasets
    combined_dataset = concatenate_datasets(datasets)
    
    # Map to DPO format
    combined_dataset = combined_dataset.map(
        lambda x: format_dpo_example(x, tokenizer),
        num_proc=8,
        desc="Formatting for DPO",
    )
    
    return combined_dataset


def main():
    parser = HfArgumentParser((DPOConfig, DPOTrainingArguments))
    dpo_config, dpo_training_args = parser.parse_args_into_dataclasses()
    
    set_seed(dpo_config.seed)
    logger.info(f"Set seed: {dpo_config.seed}")
    
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

    print(tokenizer.special_tokens_map, "ids:", tokenizer.all_special_ids)
    logger.info("Loading data")
    
    
    dataset = load_dpo_datasets(dpo_training_args.data_files, tokenizer)
    if dpo_training_args.eval_data_files:
        eval_dataset = load_dpo_datasets(dpo_training_args.eval_data_files, tokenizer)
        dpo_config.eval_strategy = "steps"
    else:
        eval_dataset = None

    logger.info(f"Loading model from {dpo_training_args.model_name_or_path}")
    
    logger.debug(
        f"AutoModelForCausalLM.from_pretrained({dpo_training_args.model_name_or_path}, trust_remote_code=True)"
    )
    model = AutoModelForCausalLM.from_pretrained(
        dpo_training_args.model_name_or_path,
        use_cache=False,
        trust_remote_code=True,
    )
    
    model.config.eos_token_id = [128001, 128008, 128009]
    
    # For DPO, we typically don't need to load a separate reference model
    # The trainer will handle creating a reference model internally
    model_ref = None

    logger.info("Setting up trainer")
    # Set max_length in the DPOConfig
    dpo_config.max_length = dpo_training_args.max_seq_length
    
    trainer = DPOTrainer(
        model,
        model_ref,
        args=dpo_config,
        processing_class=tokenizer,
        train_dataset=dataset,
        eval_dataset=eval_dataset,
    )

    logger.info("Training")
    trainer.train()
    
    model.generation_config.eos_token_id = [128001, 128008, 128009]
    
    logger.info("Saving model")
    trainer.save_model()
    
    logger.info("Test run")
    
    # For DPO test, we might want to test with a prompt
    if len(dataset) > 0:
        test_example = dataset[0]
        prompt = test_example["prompt"]
        
        # Tokenize the prompt
        tokenized_prompt = tokenizer.encode(prompt, return_tensors="pt")
        tokenized_prompt = tokenized_prompt.to('cuda' if torch.cuda.is_available() else 'cpu')

        generated_tokens = model.generate(tokenized_prompt, max_new_tokens=2048)
        generated_text = tokenizer.decode(generated_tokens[0])
        print(generated_text)
        print("====")


if __name__ == "__main__":
    main()