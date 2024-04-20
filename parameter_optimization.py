import pandas as pd
from ucimlrepo import fetch_ucirepo 
import random as r
import svm
from sklearn.preprocessing import LabelEncoder  

from sklearn.metrics import accuracy_score
from sklearn.svm import NuSVC
# fetch dataset 
iris = fetch_ucirepo(id=53) 
  
# data (as pandas dataframes) 
X = iris.data.features 
y = iris.data.targets 


label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)


from sklearn.model_selection import StratifiedShuffleSplit

# Define the number of splits
num_splits = 10

# Initialize StratifiedShuffleSplit
sss = StratifiedShuffleSplit(n_splits=num_splits, test_size=0.3, random_state=42)

X_samples = []
y_samples = []

# Iterate over the splits
# Iterate over the splits
for train_index, test_index in sss.split(X, y_encoded):
    X_train_sample, X_test_sample = X.iloc[train_index], X.iloc[test_index]
    y_train_sample, y_test_sample = y_encoded[train_index], y_encoded[test_index]  # Flatten y_encoded using ravel()
    X_samples.append((X_train_sample, X_test_sample))
    y_samples.append((y_train_sample, y_test_sample))


# Display the shape of each sample
for i, (X_train_sample, X_test_sample) in enumerate(X_samples):
    print(f"Sample {i+1}:")
    print(f"X_train shape: {X_train_sample.shape}")
    print(f"X_test shape: {X_test_sample.shape}")
    print()

Accuracies=[]
Nus=[]

Kernels=[]


for i, ((X_train_sample, X_test_sample), (y_train_sample, y_test_sample)) in enumerate(zip(X_samples, y_samples)):
    bestAccuracy =0
    bestKernel =""
    bestNu = 0
    
    iteration =100
    kernelList = ['linear', 'poly', 'rbf', 'sigmoid']

    for j in range(1, 100):
        Nu = r.random()
        k = r.choice(kernelList)  # Choose a random kernel from kernelList
        model = NuSVC(kernel=k, nu=Nu)  # Specify the appropriate parameters
        model.fit(X_train_sample, y_train_sample)

        predicted = model.predict(X_test_sample)
        accuracy = accuracy_score(y_test_sample, predicted)
        if accuracy > bestAccuracy:
            bestAccuracy = accuracy
            bestKernel = k
            bestNu = Nu

    Accuracies.append(accuracy)
    Kernels.append(bestKernel)
    Nus.append(bestNu)
    

df = pd.DataFrame({
    "Sample":["S1","S2","S3","S4","S5","S6","S7","S8","S9","S10"],
    "Best Accuracy":Accuracies,
    "Best Nu":Nus,
    "Best Kernels":Kernels
})
print(df)

import matplotlib.pyplot as plt

iterations = [100,200,300,400,500,600,700,800,900,1000]

plt.plot(iterations, Accuracies, marker='o', linestyle='-')
plt.xlabel('Iterations')
plt.ylabel('Accuracy')
plt.title('Accuracy vs. Iterations')
plt.grid(True)
plt.show()

        




