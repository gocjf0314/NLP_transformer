# 학습할 데이터 다운로드
import urllib.request
from soynlp import DoublespaceLineCorpus
from soynlp.word import WordExtractor

urllib.request.urlretrieve(
    "https://raw.githubusercontent.com/lovit/soynlp/master/tutorials/2016-10-20.txt",
    filename="2023-12-05.txt"
)

# 훈련 데이터를 다수의 문서로 분리
corpus = DoublespaceLineCorpus("2023-12-05.txt")

# 분할한 데이터 학습시키기
word_extractor = WordExtractor()
word_extractor.train(corpus)
word_score_table = word_extractor.extract()
