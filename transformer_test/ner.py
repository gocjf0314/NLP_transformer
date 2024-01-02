from text_loader import read_text_file
from transformers import pipeline

ner_tagger = pipeline("ner", aggregation_strategy="simple")

outputs = ner_tagger(read_text_file("./txt_src.txt"))
print(outputs)
