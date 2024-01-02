import numpy as np
import torch
import matplotlib.pyplot as plt
from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
# from huggingface_hub import notebook_login

model_ckpt = "distilbert-base-uncased"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 모델이 예측할 레이블 수 지정 -> 분류 헤드의 출력 크기를 설정하기 위해서
num_labels = 6

# base model + classification head
# 분류헤드가 아직 훈련되지 않은 상태에서 모델 일부가 랜덤하게 초기화 되는 경고문구 뜸
model = AutoModelForSequenceClassification.from_pretrained(
    model_ckpt, num_labels=num_labels
).to(device)

tokenizer = AutoTokenizer.from_pretrained(model_ckpt)


def tokenize(batch):
    return tokenizer(batch["text"], padding=True, truncation=True)


emotions = load_dataset("emotion")
sample_data = emotions["train"][:4]
emotions_encoded = emotions.map(tokenize, batched=True, batch_size=None)


def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    f1 = f1_score(labels, preds, average="weighted")
    acc = accuracy_score(labels, preds)
    return {"accruracy": acc, "f1": f1}

# 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

# Trainer class 정의
from transformers import Trainer, TrainingArguments

batch_size = 64

logging_steps = len(emotions_encoded["train"]) #  batch_size
model_name = f"{model_ckpt}-finetuned-emotions"
training_args = TrainingArguments(
    output_dir=model_name,
    num_train_epochs=2,
    learning_rate=2e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    disable_tqdm=False,
    logging_steps=logging_steps,
    push_to_hub=True,
    save_strategy="epoch",
    load_best_model_at_end=True,
    log_level="error"
)

trainer = Trainer(
    model=model,
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=emotions_encoded["train"],
    eval_dataset=emotions_encoded["validation"],
    tokenizer=tokenizer
)

trainer.train()
print("Finish training!\n")

preds_output = trainer.predict(emotions_encoded["validation"])


def plot_confusion_matrix(y_preds, y_true, labels):
    cm = confusion_matrix(y_true, y_preds, normalize="true")
    fig, axes = plt.subplots(figsize=(6, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap="Blues", values_format=".2f", ax=axes, colorbar="False")
    plt.title("Normalized confusion matrix")
    plt.show()


y_preds = np.argmax(preds_output.predictions, axis=1)
plot_confusion_matrix(y_preds, y_valid, labels)

