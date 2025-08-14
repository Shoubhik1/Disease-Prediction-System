import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

heart_data=pd.read_csv('/content/drive/MyDrive/heart_dataset.csv')
heart_data.head()

# Checking the distribution of Target Variable
heart_data['target'].value_counts()

# Separating the data and labels
X = heart_data.drop(columns='target', axis=1)
Y = heart_data['target']
scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)
print(X.shape, X_train.shape, X_test.shape)

model=LogisticRegression()
model.fit(X_train,Y_train)

X_pred = model.predict(X_test)
accuracy = accuracy_score(X_pred, Y_test)
print(f'Accuracy on Test data: {accuracy:.4f}')

#making a predictive system

input_data = [62,0,0,140,268,0,0,160,0,3.6,0,2,2]

# Change the input data to a numpy array
input_data_as_numpy_array= np.asarray(input_data)

# Reshape the numpy array as we are predicting for only on instance
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

prediction = model.predict(input_data_reshaped)
print(prediction)

if (prediction[0]== 0):
  print('The Person does not have a Heart Disease')
else:
  print('The Person has Heart Disease')

#saving the trained model
import pickle
file='heart_model.pkl'
pickle.dump(model,open(file,'wb'))
pickle.dump(scaler,open('heart_scaler.pkl','wb'))


from google.colab import files
files.download('heart_model.pkl')
files.download('heart_scaler.pkl')

