from datasets import list_datasets, load_dataset

# huggingface 데이터셋 목록
# all_datasets = list_datasets()
# print(len(all_datasets))
# print(all_datasets)

emotions = load_dataset("emotion")
print(emotions)

train_ds = emotions["train"]
print(train_ds.features)
