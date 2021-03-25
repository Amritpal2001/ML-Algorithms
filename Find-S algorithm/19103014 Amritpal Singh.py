import pandas as pd
import numpy as np



dfx = pd.read_csv("data.csv")
dataset = dfx.values
print(dataset)


res = []
n = dataset.shape[1]-1
for i in range(n):
    res.append('Φ')


print("The initial value of hypothesis:")
print(res)
print()
for i in dataset:
    if i[-1] == 'Yes':
        for j in range(len(i)-1):
            if(res[j] == '?'):
                pass
            elif (res[j] == i[j]) or (res[j]=='Φ'):
                res[j] = i[j];
            else:
                res[j] = '?'
        print(res)
    else:
        print(res)
        
print("\nFinal Hyposthesis : ",res)
