import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


train_dataset = pd.read_csv('train.csv').values
test_dataset = pd.read_csv('test.csv').values
x_train = train_dataset[:,:1].reshape(-1)
y_train = train_dataset[:,1:2].reshape(-1)
x_test = test_dataset[:,:1].reshape(-1)
y_test = test_dataset[:,1:2].reshape(-1)


def gradient_descent(x,y):
    m_curr = b_curr = 0
    iterations = 100
    n = len(x)
    learning_rate = 0.0001
    
    plt.scatter(x_train,y_train , color = "green")
    for i in range(iterations):
        y_predicted = m_curr * x + b_curr
        plt.plot(x_train,y_predicted,color='red')
        cost = (1/n) * sum([val**2 for val in (y-y_predicted)])
        md = -(2/n)*sum(x*(y-y_predicted))
        bd = -(2/n)*sum(y-y_predicted)
        m_curr = m_curr - learning_rate * md
        b_curr = b_curr - learning_rate * bd
    return m_curr , b_curr




m , b = gradient_descent(x_train,y_train)
print(m,b)
residual_values = m*x_test+b - y_test
plt.title('Variation of regression line')
plt.xlabel('x')
plt.ylabel('y')
plt.show()


plt.scatter(x_test , residual_values , color = "Orange")
plt.title('Residual Plot')
plt.xlabel('x value')
plt.ylabel('Residual value')
plt.show()



