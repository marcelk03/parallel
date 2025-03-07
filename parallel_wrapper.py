from multiprocessing import Pool
import numpy as np

def evaluate_row(row, objective_function):
        return objective_function(np.array([row]))

def parallel_objective_function(objective_function, X, num_cores, fun_control):
    
    with Pool(processes=num_cores) as pool:
        results = pool.starmap(evaluate_row, [(row, objective_function) for row in X])

    return np.array(results).flatten()
    
def parallel_wrap(obj_func, num_cores=4):
    global parallel_obj
    def parallel_obj(X, num_cores=num_cores, fun_control=None):
        return parallel_objective_function(obj_func, X, num_cores, fun_control)
    
    return parallel_obj