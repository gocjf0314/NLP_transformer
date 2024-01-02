from datasets import load_dataset
from transformers import AutoTokenizer, DistilBertTokenizer

model_cktp = "distilbert-base-uncased"

# DistilBert Tokenizer
# distilbert_tokenizer = DistilBertTokenizer.from_pretrained(model_cktp)

# AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(model_cktp)

# Tokenizing
text = "Tokenizing text is a core task of NLP"

encoded_text = tokenizer(text)
# print("Encoded Text: ", encoded_text)

tokens = tokenizer.convert_ids_to_tokens(encoded_text.input_ids)
# print("Tokens: ", tokens)
# print("Tokens to string: ", tokenizer.convert_tokens_to_string(tokens))
# print("Vocab size: ", tokenizer.vocab_size)
# print("Max context length of Model: ", tokenizer.model_max_length)
# print("Forward pass-field name: ", tokenizer.model_input_names)


# 샘플 토큰화 함수 정의&구현
# padding: 가장 긴 샘플 크기에 맞추어 0으로 채움
# truncation: 모델의 최대 문맥 크기에 맞추어 샘플을 잘라냄
def tokenize(batch):
    return tokenizer(batch["text"], padding=True, truncation=True)


emotions = load_dataset("emotion")
sample_data = emotions["train"][:4]
emotions_encoded = tokenize(sample_data)

print("SampleData: ", sample_data)
print(emotions_encoded)

# map -> 말뭉치에 있는 모든 샘플에 개별적으로 직용
# batch=True -> 트윗을 배치로 인코딩
# batch_size=None -> 전체 데이터 셋이 하나의 배치로 적용됨
emotions_encoded = emotions.map(tokenize, batched=True, batch_size=None)
print(emotions_encoded)


