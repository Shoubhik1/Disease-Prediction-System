import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler


diabetes_dataset=pd.read_csv('/content/drive/MyDrive/diabetes.csv')
diabetes_dataset.head()

X = diabetes_dataset.drop(columns = 'Outcome', axis=1)
Y = diabetes_dataset['Outcome']

scaler = StandardScaler()
scaler.fit(X)
X = scaler.transform(X)

X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.2, stratify=Y, random_state=2)
print(X.shape, X_train.shape, X_test.shape)

classifier = svm.SVC(kernel='linear')
classifier.fit(X_train,Y_train)

X_pred = classifier.predict(X_test)
accuracy = accuracy_score(Y_test, X_pred)
print("Accuracy score of the test data : {:.4f}".format(accuracy))

#making a predictive system

input_data = (5,166,72,19,175,25.8,0.587,51)

# Changing the input_data to numpy array
input_data_as_numpy_array = np.asarray(input_data)

# Reshape the array as we are predicting for one instance
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)
prediction = classifier.predict(input_data_reshaped)
print(prediction)

if (prediction[0] == 0):
  print('The person is not diabetic')
else:
  print('The person is diabetic')

import pickle
file = 'diabetes_model.pkl'
pickle.dump(classifier, open(file, 'wb'))
pickle.dump(scaler, open('scaler.pkl', 'wb'))
from google.colab import files

files.download('diabetes_model.pkl')
files.download('scaler.pkl')
