"""
Created on Tue Jan 19 20:24:26 2021

@author: ankit-shibu
"""
import pandas as pd
import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

from sklearn.linear_model import LinearRegression

def model():
    model = Sequential()
    model.add(Dense(8, input_dim=2, activation='linear'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(1, activation='linear'))
    model.compile(loss='mean_squared_error', optimizer = 'adam')
    return model

def generate_data(n=1000, seed=0, beta1=0.98, alpha1=0.4, alpha2=0.3):
    np.random.seed(seed)
    a = np.random.normal(65, 5, n)
    b = a / 18 + np.random.normal(size=n)
    c = beta1 * b + 2 * a + np.random.normal(size=n)
    
    return pd.DataFrame({'c': c, 'b': b, 'a': a})

def estimate(x, y, model = model(), treatment_idx=0):
    model.fit(x, y)
    Xt1 = pd.DataFrame.copy(x)
    Xt1[x.columns[treatment_idx]] = 1
    Xt0 = pd.DataFrame.copy(x)
    Xt0[x.columns[treatment_idx]] = 0
    return (model.predict(Xt1) - model.predict(Xt0)).mean()

if __name__ == '__main__':
    df = generate_data()

    ate = None
    ate1 = estimate(df[['b', 'a']], df['c'])
    ate2 = estimate(df[['b', 'a']], df['c'], model = LinearRegression())
    print('# Adjustment Formula Estimates #')
    print('ATE estimate:\t\t\t\t\t\t\t', ate1)
    print('ATE estimate:\t\t\t\t\t\t\t', ate2)





