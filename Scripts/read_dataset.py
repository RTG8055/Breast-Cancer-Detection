import numpy as np
import pandas as pd
import sys
sys.path.append('/media/rahul/Stuff/github/rtg8055/Breast-Cancer-Detection/Scripts')
import decision_tree as dt
from sklearn.model_selection import train_test_split
import Evaluation as ev
import knn as knn
import SVM as svm
import matplotlib.pyplot as plt
'''
   1. Sample code number           
   2. Clump Thickness 
   3. Uniformity of Cell Size      
   4. Uniformity of Cell Shape     
   5. Marginal Adhesion            
   6. Single Epithelial Cell Size  
   7. Bare Nuclei 
   8. Bland Chromatin              
   9. Normal Nucleoli              
  10. "Mi,toses"  
  11. Class:                       
 '''

def check(x):
  if(x=='?'): return None
data = pd.read_csv("../Data/breast-cancer-wisconsin-data.csv",header=None)#,usecols=["Sample code number","Clump Thickness","Uniformity of Cell Size","Uniformity of Cell Shape","Marginal Adhesion","Single Epithelial Cell Size","Bare Nuclei","Bland Chromatin","Normal Nucleoli","Mitoses","Class"])
data.columns = ["Sample code number","Clump Thickness","Uniformity of Cell Size","Uniformity of Cell Shape","Marginal Adhesion","Single Epithelial Cell Size","Bare Nuclei","Bland Chromatin","Normal Nucleoli","Mitoses","Class"]
print(data.head())
print(data.shape)
data.drop("Sample code number",axis=1,inplace=True)
print(data.shape)
data.dropna(inplace=True)
print(data.shape)
for i in data.columns:
  data[i].apply(check)
  data[i].dropna(inplace=True)
print(data.shape)

y = data['Class']
X = data.drop('Class',axis=1)
X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=0)
print(X_train.shape,X_test.shape,y_train.shape,y_test.shape)


my_data = []
with open("../Data/breast-cancer-wisconsin-data.csv") as file:
  for line in file:
    if(line.find('?') != -1):
      continue
    line = line.rstrip().split(',')[1:]
    line = [int(l) for l in line]
    line[-1] = str(line[-1])

    my_data.append(line)

#splitting 30% and 70%
total = len(my_data)
train = int(.7 *total)
test =total - train
print(total,train,test)

train_data = my_data[0:train]
test_data = my_data[train:]


tree= dt.buildtree(train_data)
predicted_results = dt.predict(test_data,tree)
expected_results = [row[-1] for row in test_data]
# print(predicted_results,expected_results)


predicted_results2 = knn.main(train_data,test_data)
predicted_results3 = svm.predict(train_data,test_data)

print("----Confusion Matrix----")

'''
      predicted 
        2   4
actual 2 a    b
       4 c    d

'''

c1 = ev.confusion_matrix(predicted_results,expected_results)
c2 = ev.confusion_matrix(predicted_results2,expected_results)
c3 = ev.confusion_matrix(predicted_results3,expected_results)
print(c1)
print(c2)
print(c3)
fig ,axes = plt.subplots(2,2)

axes[0,0].plot([1,2,3],[c1[0][0],c2[0][0],c3[0][0]])
axes[0,0].set_title("Correctly Classified instances")
# axes[0,0].xlabel("Method used")
# axes[0,0].ylabel("count")
# axes[0,0].yticks(range(150,165,1))
plt.show()

print("----Kappa Score----")
print(ev.kappa_score(predicted_results,expected_results))
print(ev.kappa_score(predicted_results2,expected_results))
print("----Mean Absolute Error----")
print(ev.MAE(predicted_results,expected_results))
print(ev.MAE(predicted_results2,expected_results))
print("----Precision and Recall----")
print(ev.precision_recall(predicted_results,expected_results))
print(ev.precision_recall(predicted_results2,expected_results))


