#代码
import numpy as np
A = np.array([[1.0,1.5],[2.0,1.0]]) #a(1.0, 1.5) b(2.0, 1.0)
c = np.array([1.0,1.0])
#A-c 这里有个小知识点是关于矩阵广播的 大家可以网上查阅一下
d = A-c
f = [(d[i][0]**2+d[i][1]**2)**(1.0/2) for i in range(len(d))]    
if np.argmin(f):
    print("C为B类")
else:
    print("C为A类")	