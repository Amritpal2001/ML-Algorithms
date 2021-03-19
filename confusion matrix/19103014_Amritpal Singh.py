import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from math import sqrt
from sklearn.model_selection import KFold
import seaborn as sn



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
    
    



def k_fold(train_data, test_data):    
    Accuracy = {}
    K_value = 0
    for K in range(1,30):
        temp = []
        data = train_data
        kfold = KFold(n_splits=10)
        kfold.get_n_splits(data)
        for train, test in kfold.split(data):
            temp.append(call_knn(data[train] , data[test] ,K))
        Accuracy[K] = sum(temp)/len(temp)*100
    K_value = max(Accuracy, key=Accuracy.get)
    
    count = 0
    true_values = []
    predicted_values = []
    for i in test_data:
        res = knn(train_data , i , K_value)
        true_values.append(i[-1])
        predicted_values.append(res)
        if res == i[-1]:
            count+=1
    return true_values , predicted_values



def confusion_matrix_fun(path):
    dfx = pd.read_csv(path)
    dataset = dfx.values
    np.random.shuffle(dataset)
    temp = int(dataset.shape[0]* 0.8)
    train_data = dataset[:temp]
    test_data = dataset[temp:]
    labels = list(np.unique(dataset[:,-1]))
    true_values , predicted_values  = k_fold(train_data,test_data)
    
    n = len(labels)
    matrix = np.zeros((n,n), dtype=int)
    for a,b in zip(true_values , predicted_values):
        x = labels.index(a)
        y = labels.index(b)
        matrix[x][y] = matrix[x][y]+1
    
    for i in range(len(labels)):
        print(labels[i])
        tp = matrix[i][i]
        print("True Positive : ",tp)
        fp = sum(matrix[i])-tp
        print("False Positive : ",fp)
        fn = 0
        for j in range(matrix.shape[0]):
            fn+=matrix[j][i]
        fn = fn - tp
        print("False Negative : ",fn)
        tn = np.sum(matrix) - tp - fp - fn
        print("True Negative : ",tn)
        print("Precision : ", tp/(tp+fp))
        print("Recall : ", tp/(tp+fn))
        print("Accuracy : ", (tp+tn)/len(test_data))
        print("False negative rate : ", fn/(fn+tp))
        print("False positive rate : ", fp/(fp+tn))
        print()
   
    
    
    df_cm = pd.DataFrame(matrix, (i for i in labels) , (i for i in labels))
    sn.heatmap(df_cm, annot=True)
    plt.show()
    
    



confusion_matrix_fun('iris.csv')
confusion_matrix_fun('indian_liver_patient.csv')
confusion_matrix_fun('Class_Ionosphere.csv')
