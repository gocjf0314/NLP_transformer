from ckonlpy.tag import Twitter

twitter = Twitter()
print(twitter.morphs('은경이는 사무실로 갔습니다.'))

twitter.add_dictionary('은경이', 'Noun')

print(twitter.morphs('은경이는 사무실로 갔습니다.'))

