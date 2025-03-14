import pandas as pd
import matplotlib.pyplot as plt
from fontTools.misc.timeTools import epoch_diff

data = pd.read_csv('score.csv')

print(data.head())

plt.scatter(data.Hours, data.Scores)
plt.show()

def loss_function(m,p,point):
    total_error = 0
    for i in range(len(point)):
        x= point.iloc[i].Hours
        y= point.iloc[i].Scores
        total_error += (y-(m*x+b))**2
    total_error / float(len(point))

def gradient_descent(m_now,b_now,point, alpha):
    m_grad =0
    b_grad =0

    n=len(point)
    for i in range(n):
        x= point.iloc[i].Hours
        y= point.iloc[i].Scores

        m_grad += -(2/n)*x*(y-(m_now*x +b_now))
        b_grad += -(2/n)*(y-(m_now*x +b_now))
    m= m_now -m_grad*alpha
    b= b_now -b_grad*alpha
    return m,b


m=0
b=0
l=0.0003
epochs = 1000

for i in range(epochs):
    m,b = gradient_descent(m,b,data,l)


print(m,b)

plt.scatter(data.Hours, data.Scores,color='red')
plt.plot(list(range(20,80)),[m*x+b for x in range(20,80)],color ='black')
plt.show()