import matplotlib.pyplot as plt
import numpy as np

# 그림 사이즈, 바 굵기 조정
fig, ax = plt.subplots(figsize=(12,6))
bar_width = 0.25
band =  [4, 5, 6, 7, 8, 9]

# 연도가 4개이므로 0, 1, 2, 3 위치를 기준으로 삼음
index = np.arange(6)

C = [14, 36, 21, 17, 9, 3]
A = [21, 16, 24, 24, 9, 7]
F = [2, 31, 31, 28, 5, 3]

# 각 연도별로 3개 샵의 bar를 순서대로 나타내는 과정, 각 그래프는 0.25의 간격을 두고 그려짐
b1 = plt.bar(index, C, bar_width, alpha=0.4, color='red', label='Complexity')

b2 = plt.bar(index + bar_width, A, bar_width, alpha=0.4, color='blue', label='Accuracy')

b3 = plt.bar(index + 2 * bar_width, F, bar_width, alpha=0.4, color='green', label='Fluency')

# x축 위치를 정 가운데로 조정하고 x축의 텍스트를 year 정보와 매칭
plt.yticks(size = 12)
plt.xticks(np.arange(bar_width, 6 + bar_width, 1), band, size = 12)

# x축, y축 이름 및 범례 설정
plt.xlabel('CAF Band Prediction', size = 15)
plt.ylabel('Percentage (%)', size = 15)
plt.legend()
plt.show()
plt.savefig("./result.png")