import numpy as np
import matplotlib.pyplot as plt
from math import sqrt
from collections import Counter
#导入数据 可以观察到 数据有三列 最后一列表示其标签 也就是label 0表示良性肿瘤 1 表示恶性肿瘤
raw_data_X = np.array([[3,2,0],
                       [3,1,0],
                       [1,3,0],
                       [3,4,0],
                       [2,2,0],
                       [7,4,1],
                       [5,3,1],
                       [9,2,1],
                       [7,3,1],
                       [7,0,1]])

X_train = np.array(raw_data_X[:,:2]) #训练集
y_train = np.array(raw_data_X[:,-1]) #训练集

#画出散点图 
#X_train[y_train==0,0]为良性肿瘤横坐标 X_train[y_train==0,1]良性肿瘤纵坐标
#X_train[y_train==1,0]为恶性肿瘤横坐标 X_train[y_train==1,1]为恶性肿瘤纵坐标

plt.scatter(X_train[y_train==0,0], X_train[y_train==0,1],color="g")#y_train只有两种结果良性或者恶性
plt.scatter(X_train[y_train==1,0], X_train[y_train==1,1],color="r") #后面那个0/1是两列的数
plt.show()

#传入待检测样本
x = np.array([8.093607318,3.365731514])
plt.scatter(X_train[y_train==0,0], X_train[y_train==0,1],color="g")
plt.scatter(X_train[y_train==1,0], X_train[y_train==1,1],color="r") 
#画出待检测样本在图中的位置
plt.scatter(x[0],x[1],color="b")
plt.show()
#计算点间距
distance = [sqrt(np.sum((x_train - x)**2)) for x_train in X_train]  
print(distance)
#选取最近的k个点
k=6
#按最小排序返回索引
neareast = np.argsort(distance)
topK_y = [y_train[i] for i in neareast[:k]]
votes = Counter(topK_y)
predict_y = votes.most_common(1)[0][0]
if predict_y == 0:
	print("是良性肿瘤")
else:
    print("是恶性肿瘤")