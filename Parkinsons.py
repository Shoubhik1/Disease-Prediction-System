import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

parkinsons_data=pd.read_csv('/content/drive/MyDrive/parkinsons.csv')
parkinsons_data.head()

parkinsons_data['status'].value_counts()

parkinsons_data.drop(columns='name').groupby('status').mean()

# Separating the data and labels
X = parkinsons_data.drop(columns=['name','status'], axis=1)
Y = parkinsons_data['status']
scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)
print(X.shape, X_train.shape, X_test.shape)

model = svm.SVC(kernel='linear')
# Training the SVM model with training data
model.fit(X_train, Y_train)

X_pred = model.predict(X_test)
accuracy = accuracy_score(Y_test, X_pred)
print(f'Accuracy score of test data :{accuracy:.4f}')


#predictive system
input_data = (197.07600,206.89600,192.05500,0.00289,0.00001,0.00166,0.00168,0.00498,0.01098,0.09700,0.00563,0.00680,0.00802,0.01689,0.00339,26.77500,0.422229,0.741367,-7.348300,0.177551,1.743867,0.085569)

# Changing input data to a numpy array
input_data_as_numpy_array = np.asarray(input_data)

# Reshape the numpy array
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

prediction = model.predict(input_data_reshaped)
print(prediction)


if (prediction[0] == 0):
  print("The Person does not have Parkinsons Disease")

else:
  print("The Person has Parkinsons")


import pickle
filename = 'parkinsons_model.pkl'
pickle.dump(model, open(filename, 'wb'))
pickle.dump(scaler, open('parkinsons_scaler.pkl', 'wb'))
from google.colab import files
files.download('parkinsons_model.pkl')
files.download('parkinsons_scaler.pkl')
