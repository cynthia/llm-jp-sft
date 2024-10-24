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
dataset = load_dataset("tokyotech-llm/Swallow-Instruct-v0.1", split="train")
# データセットの形式と事例数を確認します。

base_model_name = "tokyotech-llm/Llama-3.1-Swallow-8B-v0.1"
tokenizer_name = "tokyotech-llm/Llama-3.1-Swallow-8B-Instruct-v0.1"

tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
tokenized_dataset = [
    tokenizer.apply_chat_template(item["conversation"])
    for item in dataset
]

tokenizer.pad_token = tokenizer.unk_token



bos = tokenizer.bos_token
collator = DataCollatorForCompletionOnlyLM(
    instruction_template=bos + "ユーザ：",  # ユーザの発話開始を示す文字列
    response_template=bos + "アシスタント：",  # アシスタントの返答開始を示す文字列
    tokenizer=tokenizer,  # トークナイザ
)
# トークナイズされたデータセットの先頭をミニバッチ構築処理
batch = collator(tokenized_dataset[:1])
input_ids = batch["input_ids"][0]
labels = batch["labels"][0]
print("入力トークンID:", input_ids)
print("正解ラベル:", labels)

segments_to_fit: list[list[int]] = []
segments_to_ignore: list[list[int]] = []
# ラベルが-100である箇所とそうでない箇所ごとにグルーピング
for key, group in itertools.groupby(
    range(len(input_ids)), key=lambda i: labels[i] == -100
):
    group = list(group)
    if key:
        segments_to_ignore.append(group)
    else:
        segments_to_fit.append(group)

print("---- 損失を計算しない部分 ----")
for seg in segments_to_ignore:
    print(tokenizer.decode(input_ids[seg]))
    print()

print("---- 損失を計算する部分 ----")
for seg in segments_to_fit:
    print(tokenizer.decode(input_ids[seg]))
    print()


model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    torch_dtype=torch.bfloat16,
    use_cache=False,  # 後にgradient checkpointingを有効にするために必要
    device_map="auto",
)
     
     
# 学習パラメータ
training_args = TrainingArguments(
    output_dir="/gs/bs/tga-okazaki/ma/results", # 結果の保存フォルダ
    bf16=True,  # BF16を使用した学習の有効化
    num_train_epochs=1,  # エポック数
    per_device_train_batch_size=2,  # 訓練時のバッチサイズ
    gradient_accumulation_steps=8,  # 勾配累積のステップ数（5.5.2）
    gradient_checkpointing=True,  # 勾配チェックポインティングの有効化（5.5.3）
    optim="paged_adamw_8bit",  # 最適化器
    learning_rate=3e-4,  # 学習率
    lr_scheduler_type="cosine",  # 学習率スケジューラの種類
    max_grad_norm=0.3,  # 勾配クリッピングにおけるノルムの最大値（9.4.3）
    warmup_ratio=0.1,  # 学習率のウォームアップの長さ（5.2.8）
    logging_steps=10,  # ロギングの頻度
    save_steps=300,  # モデルの保存頻度
)

tokenized_dataset = tokenized_dataset[:1000]

trainer = Trainer(
    model,
    train_dataset=tokenized_dataset,  # トークンID化されたデータセット
    data_collator=collator,  # ラベルの加工及びミニバッチ構築処理を行うモジュール
    args=training_args,  # 訓練の設定
    tokenizer=tokenizer,  # パラメータ保存時にトークナイザも一緒に保存するために指定
)

trainer.train()