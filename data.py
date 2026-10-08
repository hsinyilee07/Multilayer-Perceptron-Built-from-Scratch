import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np

#Data Preparation Function
def data_preparation(data_file_path, seed_number=106, ratio = 0.8):
  data = pd.read_csv(data_file_path)
  # 0. PRECLEAN (remove samples that are missing information)
  # print(data.isnull().sum())  #count of many null values are in each column
  data = data.dropna()

  # 1. EXTRACT & PREPROCESS x & y values
  x_values = data.iloc[:,:-1].values   #adding .values converts the data from Pandas DataFrame to NumPy ndarray
  y_values = data.iloc[:,-1].values

  # 1) X: normalize x values
  scaler = StandardScaler()
  x_values = scaler.fit_transform(x_values)  #scales each input feature to have a mean of 0 and std of 1 (z score calculation)
  # 2) Y: other perprocessing
  y_values = y_values.reshape(-1,1) #add a column dimention to y

  #2. SEPERATE: seperate train & test dataset
  np.random.seed(seed_number)
  total_sample_size = x_values.shape[0]
  random_perm_of_indexes = np.random.permutation(total_sample_size)

  train_indexes = random_perm_of_indexes[:int(ratio * total_sample_size)]
  test_indexes = random_perm_of_indexes[int(ratio * total_sample_size):]

  x_train = x_values[train_indexes]
  y_train = y_values[train_indexes]
  x_test = x_values[test_indexes]
  y_test = y_values[test_indexes]
  
  return x_train, y_train, x_test, y_test