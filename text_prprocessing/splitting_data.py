import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

np_array = np.arange(0,16).reshape((4,4))
print('전체 데이터 :')
print(np_array)

# 마지막 열을 제외하고 X데이터에 저장
X = np_array[:, :3]

# 마지막 열만 y데이터에 저장
y = np_array[:,3]

print('X 데이터 :')
print(X)
print('y 데이터 :',y)