import numpy as np
import pandas as pd


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
data = pd.read_csv("../Data/breast-cancer-wisconsin-data.csv",header=None)#,usecols=["Sample code number","Clump Thickness","Uniformity of Cell Size","Uniformity of Cell Shape","Marginal Adhesion","Single Epithelial Cell Size","Bare Nuclei","Bland Chromatin","Normal Nucleoli","Mitoses","Class"])
data.columns = ["Sample code number","Clump Thickness","Uniformity of Cell Size","Uniformity of Cell Shape","Marginal Adhesion","Single Epithelial Cell Size","Bare Nuclei","Bland Chromatin","Normal Nucleoli","Mitoses","Class"]
print(data.head())
print(data.shape)
data.drop("Sample code number",axis=1,inplace=True)
data.dropna(inplace=True)
print(data.shape)
x=data.drop("Class",1)
y=data['Class']


