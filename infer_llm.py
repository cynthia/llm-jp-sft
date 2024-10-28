from transformers.trainer_utils import set_seed
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset, concatenate_datasets
import argparse
import torch

# 乱数シードを42に固定
set_seed(42)
    

def load_chat_datasets(files):
    datasets = [] # multiple datasets
    
    for data_file in files:

        dataset = load_dataset("json", data_files=data_file)
        dataset = dataset["train"]
        datasets.append(dataset)
    
    return concatenate_datasets(datasets)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_files", required=True, nargs="+", type=str)
    parser.add_argument("--model_name_or_path", required=True, type=str)
    parser.add_argument("--tokenizer_name_or_path", default=None, type=str)
    parser.add_argument("--additional_special_tokens", default=None, type=list[str])
    args = parser.parse_args()    
    
    tokenizer_name_or_path: str = (
        args.tokenizer_name_or_path or args.model_name_or_path
    )
    
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name_or_path,
        additional_special_tokens=args.additional_special_tokens,
        trust_remote_code=True,
    )
    
    print(args.data_files) 
    dataset = load_chat_datasets(args.data_files)


    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        use_cache=False,
        trust_remote_code=True,
    ).to('cuda' if torch.cuda.is_available() else 'cpu')
    
    print([dataset[1]["conversation"][0]])
    messages = [dataset[1]["conversation"][0]]
    tokenized_chat = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to('cuda' if torch.cuda.is_available() else 'cpu')
    
    generated_tokens = model.generate(tokenized_chat, max_new_tokens=2048)
    generated_text = tokenizer.decode(generated_tokens[0])
    print(generated_text)
    print("====")
    print(len(generated_tokens[0]),generated_tokens[0])
    
    return


if __name__ == "__main__":
    
    main()
