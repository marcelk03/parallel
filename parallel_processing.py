import numpy as np
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import Pool
import joblib as jl

_model = None

def get_model():
    global _model
    if _model is None:
        _model = jl.load('random_forest_model.pkl')
    return _model

def objective_function(X, fun_control=None):
    if not isinstance(X, np.ndarray):
        X = np.ndarray(X)
    if X.ndim !=2 or X.shape[1] != 2:
        raise Exception
    
    X_train, y_train = make_regression()

    n_estimators=
    max_depth=

    model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, n_jobs=1, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_train)

    mse = mean_squared_error(y_train, y_pred)
    return mse

def evaluate_row(row):
    return objective_function(np.array([row]))

def parallel_objective_ppe(X, num_cores=2, fun_control=None):
    results = []

    with ProcessPoolExecutor(max_workers=num_cores) as executor:
        results = list(executor.map(evaluate_row, X))
    
    return np.array(results).flatten()

def parallel_objective_pool(X, num_cores=4, fun_control=None):
    results = []

    with Pool(processes=num_cores) as pool:
        results = pool.map(evaluate_row, X)
    
    return np.array(results).flatten()

def parallel_objective_joblib(X, num_cores=4, fun_control=None):
    results = []

    results = jl.Parallel(n_jobs=num_cores)(jl.delayed(evaluate_row)(x) for x in X)
    return np.array(results).flatten()

def sequential_objective(X, fun_control=None):
    results = []
    
    for row in X:
        result = evaluate_row(row)
        results.append(result)
    
    return np.array(results).flatten()
