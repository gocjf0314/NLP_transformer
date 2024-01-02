from konlpy.tag import Okt

okt = Okt()
tokens = okt.morphs("나는 자연어 처리를 배운다")
print(tokens)  # ['나', '는', '자연어', '처리', '를', '배운다']

word_to_index = {word: index for index, word in enumerate(tokens)}
print('단어 집합 :', word_to_index)  # 단어 집합 : {'나': 0, '는': 1, '자연어': 2, '처리': 3, '를': 4, '배운다': 5}


def one_hot_encoding(word, word2index):
    one_hot_vector = [0] * (len(word2index))
    index = word2index[word]
    one_hot_vector[index] = 1
    return one_hot_vector


one_hot_encoding("자연어", word_to_index)  # [0, 0, 1, 0, 0, 0]
