import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression
import joblib

def train_model():
    X_train, y_train = make_regression(n_samples=1000, n_features=2, noise=0.1, random_state=42)
    model = RandomForestRegressor(n_estimators=200, n_jobs=-1, random_state=42)
    model.fit(X_train, y_train)

if __name__ == "__main__":
    train_model()