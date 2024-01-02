import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datasets import load_dataset
from transformers import AutoModel, TFAutoModel, AutoTokenizer
from umap import UMAP
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier


model_ckpt = "distilbert-base-uncased"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AutoModel.from_pretrained(model_ckpt).to(device)

# tf_model = TFAutoModel.from_pretrained(model_ckpt, from_pt=True).to(device)

text = "this is a test"
tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
inputs = tokenizer(text, return_tensors="pt")
print(f"input tensor size: {inputs['input_ids'].size()}")

inputs = {k:v.to(device) for k,v in inputs.items()}
with torch.no_grad():
    outputs = model(**inputs)

print(outputs.last_hidden_state.size())
print(outputs.last_hidden_state[:, 0].size())


def extract_hidden_states(batch):
    # 모델 입력을 gpu로 옮김
    inputs = {k:v.to(device) for k,v in batch.items() if k in tokenizer.model_input_names}

    # 마지막 은익 상태를 추출합니다
    with torch.no_grad():
        last_hidden_state = model(**inputs).last_hidden_state

    # [CLS] 토큰에 대한 벡터 반환
    # 은닉상태를 cpu로 가져와서 numpy로 변환
    # map 함수의 batched를 사용하려면 넘파이 객체로 변환 후 반환하는 작업 필요
    return {"hidden_state": last_hidden_state[:, 0].cpu().numpy()}


def tokenize(batch):
    return tokenizer(batch["text"], padding=True, truncation=True)


emotions = load_dataset("emotion")
sample_data = emotions["train"][:4]
emotions_encoded = emotions.map(tokenize, batched=True, batch_size=None)

emotions_encoded.set_format("torch", columns=["input_ids", "attention_mask", "label"])
emotions_hidden = emotions_encoded.map(extract_hidden_states, batched=True)

# scikit 스타일 특성 행렬 만들기
x_train = np.array(emotions_hidden["train"]["hidden_state"])
x_valid = np.array(emotions_hidden["validation"]["hidden_state"])
y_train = np.array(emotions_hidden["train"]["label"])
y_valid = np.array(emotions_hidden["validation"]["label"])

# 특성 스케일을 [0,1] 범위로 조정
x_scaled = MinMaxScaler().fit_transform(x_train)
# UMAP 객체를 생성하고 훈련
mapper = UMAP(n_components=2, metric="cosine").fit(x_scaled)
# 2D 임베딩의 데이터프레임 만들기
df_emb = pd.DataFrame(data=mapper.embedding_, columns=['X', 'Y'])
df_emb["label"] = y_train
df_emb.head()

# 수렴 보장을 위해 max_iter(기본값 100) 증가
lr_clf = LogisticRegression(max_iter=3000)
lr_clf.fit(x_train, y_train)
score = lr_clf.score(x_valid, y_valid)
print("LogisticRegression Model: ", score)

# 시각화 코드
cmaps = ["Greys", "Blues", "Oranges", "Reds", "Purples", "Greens"]
labels = emotions["train"].features["label"].names

# fig, axes = plt.subplots(2, 3, figsize=(7, 5))
# axes = axes.flatten()

# for i, (label, cmap) in enumerate(zip(labels, cmaps)):
#     df_emb_sub = df_emb.query(f"label == {i}")
#     axes[i].hexbin(df_emb_sub["X"], df_emb_sub["Y"], cmap=cmap, gridsize=20, linewidths=(0,))
#     axes[i].set_title(label)
#     axes[i].set_xticks([]), axes[i].set_yticks([])
#
# plt.tight_layout()
# plt.show()


# # 기준 분류모델로 훈련 후 성능 확인
# dummy_clf = DummyClassifier(strategy="most_frequent")
# dummy_clf.fit(x_train, y_train)
# score = dummy_clf.score(x_valid, y_valid)
# print("DummyClassifier Model: ", score)

from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix


def plot_confusion_matrix(y_preds, y_true, labels):
    cm = confusion_matrix(y_true, y_preds, normalize="true")
    fig, axes = plt.subplots(figsize=(6, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap="Blues", values_format=".2f", ax=axes, colorbar="False")
    plt.title("Normalized confusion matrix")
    plt.show()


y_preds = lr_clf.predict(x_valid)
plot_confusion_matrix(y_preds, y_valid, labels)
