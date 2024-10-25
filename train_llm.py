from transformers.trainer_utils import set_seed
from pprint import pprint
import itertools
from datasets import load_dataset
from transformers import AutoTokenizer
from trl import DataCollatorForCompletionOnlyLM
from transformers import Trainer, TrainingArguments

import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig


# 乱数シードを42に固定
set_seed(42)

# Hugging Face Hub上のリポジトリからデータセットを読み込む
dataset = load_dataset("json", data_files="/gs/fs/tga-okazaki/ma/lmsys-chat-1m/sft/lmsys-chat-1m-synth-sft.jsonl.gz", split="train")

base_model_name = "tokyotech-llm/Llama-3.1-Swallow-8B-v0.1"
tokenizer_name = "tokyotech-llm/Llama-3.1-Swallow-8B-Instruct-v0.1"

tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
tokenized_dataset = [
    tokenizer.apply_chat_template(item["conversation"])
    for item in dataset.select(range(5000))
]

instruction_ids = tokenizer.encode("<|start_header_id|>user<|end_header_id|>\n\n")[1:] # no begin of text
response_ids = tokenizer.encode("<|start_header_id|>assistant<|end_header_id|>\n\n")[1:]# no begin of text

collator = DataCollatorForCompletionOnlyLM(
    instruction_template=instruction_ids,  # ユーザの発話開始を示す文字列
    response_template=response_ids,  # アシスタントの返答開始を示す文字列
    tokenizer=tokenizer,  # トークナイザ
)


model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
#    torch_dtype=torch.bfloat16,
    use_cache=False,  # 後にgradient checkpointingを有効にするために必要
    device_map="auto",
)
     
model.config.eos_token_id = [
    128001,
    128008,
    128009
  ]

#model.config.pad_token_id = tokenizer.pad_token_id
#model.config.eos_token_id = tokenizer.eos_token_id

# 学習パラメータ
training_args = TrainingArguments(
    output_dir="/gs/bs/tga-okazaki/ma/results", # 結果の保存フォルダ
    bf16=False,
    fp16=True, 
    num_train_epochs=1,  # エポック数
    per_device_train_batch_size=2,  # 訓練時のバッチサイズ
    gradient_accumulation_steps=64,  # 勾配累積のステップ数（5.5.2）
    gradient_checkpointing=True,  # 勾配チェックポインティングの有効化（5.5.3）
    optim="paged_adamw_8bit",  # 最適化器
    learning_rate=1e-5,  # 学習率
    lr_scheduler_type="cosine",  # 学習率スケジューラの種類
    max_grad_norm=0.5,  # 勾配クリッピングにおけるノルムの最大値（9.4.3）
    warmup_ratio=0.1,  # 学習率のウォームアップの長さ（5.2.8）
    logging_steps=10,  # ロギングの頻度
    save_steps=300,  # モデルの保存頻度
)

tokenized_dataset = tokenized_dataset

trainer = Trainer(
    model,
    train_dataset=tokenized_dataset,  # トークンID化されたデータセット
    data_collator=collator,  # ラベルの加工及びミニバッチ構築処理を行うモジュール
    args=training_args,  # 訓練の設定
    tokenizer=tokenizer,  # パラメータ保存時にトークナイザも一緒に保存するために指定
)

trainer.train()


print("Saving model")
trainer.save_model()

prompt = dataset[1]["conversation"][0]["content"]
messages = [{"role": "user", "content": prompt}]
tokenized_chat = tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_tensors="pt"
)

model.generation_config.eos_token_id = [128001, 128008, 128009]

generated_tokens = model.generate(tokenized_chat, max_new_tokens=2048)
generated_text = tokenizer.decode(generated_tokens[0])
print(generated_text)
print("====")
print(len(generated_tokens[0]),generated_tokens[0])
