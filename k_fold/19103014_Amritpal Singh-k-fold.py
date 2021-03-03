import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from math import sqrt




dfx = pd.read_csv('/Users/amritpalsingh/Downloads/indian_liver_patient.csv')




dataset = dfx.values
temp = int(dataset.shape[0]* 0.7)
train_data = dataset[:temp]

test_data = dataset[temp:]

true_values = test_data[:,-1]




def get_distance(row1, row2):
    dis = 0.0
    for i in range(len(row1)-1):
        dis+=(row1[i] - row2[i])**2
    return sqrt(dis)




def get_neighbours(data , test_row , k):
    distances=[]
    for i in data:
        dis = get_distance(test_row, i)
        distances.append((i,dis))
    distances.sort(key = lambda x: x[1])
    neighbours = []
    for i in range(k):
        neighbours.append(distances[i][0])
    return neighbours
    




def knn(train_data , test_row , k):
    neighbours = get_neighbours(train_data , test_row,k)
    output_values = [row[-1] for row in neighbours]
    prediction = max(set(output_values), key=output_values.count)
    return prediction
    




def call_knn(train , test , j):
    count = 0
    for i in range (len(test)):
        res = knn(train , test[i] ,j)
        if test[i][-1]== res:
            count+=1
    return count/len(test)
    
    




Accuracy = {}
K_value = 0
for K in range(1,30):
    temp = []
    data = train_data
    kfold = KFold(n_splits=8)
    kfold.get_n_splits(data)
    for train, test in kfold.split(data):
        temp.append(call_knn(data[train] , data[test] ,K))
    Accuracy[K] = sum(temp)/len(temp)*100
K_value = max(Accuracy, key=Accuracy.get)
print("Value of K for which Accuracy is highest : ",K_value)
    

plt.plot(Accuracy.keys() , Accuracy.values())
plt.title('Accuracy of KNN algorithm for different values of K using k-fold cross validation')
plt.xlabel('K value')
plt.ylabel('Accuracy (%)(k-fold)')
plt.show()




count = 0
for i in test_data:
    res = knn(train_data , i , K_value)
    print('Result: %d, Actual: %d' %(res , i[-1]))
    if res == i[-1]:
        count+=1
    



print("Accuracy for training data(unseen) : ",count/len(test_data)*100)

