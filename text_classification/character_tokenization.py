# Simple char tokenization

# Convert str to list
text = "Tokenizing text is a core task of NLP"
tokenized_text = list(text)
print(tokenized_text)

# Token Nummericalization
token2idx = {ch: idx for idx, ch in enumerate(sorted(set(tokenized_text)))}
input_ids = [token2idx[token] for token in tokenized_text]
print(input_ids)

import torch
import torch.nn.functional as F

input_ids = torch.tensor(input_ids)
one_hot_encodings = F.one_hot(input_ids, num_classes=len(token2idx))
print("One-hot Encodings")
print(one_hot_encodings.shape)

print("Token: ", tokenized_text[0])
print("Tensor idx: ", input_ids[0])
print("One-hot Encodings: ", one_hot_encodings[0])
