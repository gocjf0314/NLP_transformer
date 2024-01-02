from transformers import pipeline
from text_loader import read_text_file

classifier = pipeline("text-classification")

outputs = classifier(read_text_file("./txt_src.txt"))
print(outputs)
