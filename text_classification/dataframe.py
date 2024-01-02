import pandas as pd
import matplotlib.pyplot as plt
from datasets import load_dataset

emotions = load_dataset("emotion")

emotions.set_format(type="pandas")
df = emotions["train"][:]
df.head()


def label_int2str(row):
    return emotions["train"].features["label"].int2str(row)


df["label_name"] = df["label"].apply(label_int2str)
df.head()

# Visualization
df["label_name"].value_counts(ascending=True).plot.barh()
plt.title("Frequency of Classes")
plt.show()

df["Word per Tweet"] = df["text"].str.split().apply(len)
df.boxplot("Word per Tweet", by="label_name", grid=False, showfliers=False, color="black")
plt.title("")
plt.xlabel("")
plt.show()
