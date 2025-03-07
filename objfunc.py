import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes
'''
def objective_function(X, fun_control=None):
    if not isinstance(X, np.ndarray):
        X = np.array(X)
    if X.shape[1] != 2:
        raise Exception
    x0 = X[:, 0]
    x1 = X[:, 1]
    y = x0**2 + 10*x1**2
    return y
'''
def objective_function(X, fun_control=None):
    if not isinstance(X, np.ndarray):
        X = np.array(X)
    if X.shape[1] != 2:
        raise Exception
    
    alpha = X[:, 0]
    n_iter = X[:, 1].astype(int)
    
    y_data = load_diabetes()

    X_data, y_true = y_data['data'], y_data['target']
    X_train, X_test, y_train, y_test = train_test_split(X_data, y_true, test_size=0.2, random_state=42)
    
    losses = []
    for a, n in zip(alpha, n_iter):
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        loss = mean_squared_error(y_test, y_pred)
        losses.append(loss)
    
    return np.array(losses)