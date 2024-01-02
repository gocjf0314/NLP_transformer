from tensorflow import keras

text = "나랑 점심 먹으러 갈래 점심 메뉴는 햄버거 갈래 갈래 햄버거 최고야"

# Tokenize 수행 후 단어 빈도 순으로 매핑
tokenizer = keras.preprocessing.text.Tokenizer()
tokenizer.fit_on_texts([text])
print('단어 집합 :', tokenizer.word_index)

# tokenizer.word_to_index 와 매칭 후 sub_text 정수로 인코딩
sub_text = "점심 먹으러 갈래 메뉴는 햄버거 최고야"
encoded = tokenizer.texts_to_sequences([sub_text])[0]
print(encoded)

# One-Hot Encoding
one_hot = keras.utils.to_categorical(encoded)
print(one_hot)
