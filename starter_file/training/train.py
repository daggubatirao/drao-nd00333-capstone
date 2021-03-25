from sklearn.linear_model import LogisticRegression
import argparse
import os
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.metrics import roc_auc_score, roc_curve
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
from azureml.core.run import Run
from azureml.core import Workspace, Dataset


run = Run.get_context()
ws = run.experiment.workspace


def main():
    # Add arguments to script
    parser = argparse.ArgumentParser()

    parser.add_argument('--C', type=float, default=1.0, help="Inverse of regularization strength. Smaller values cause stronger regularization")
    parser.add_argument('--max_iter', type=int, default=100, help="Maximum number of iterations to converge")

    args = parser.parse_args()
    
    # split data to train and test sets
    dataset = Dataset.get_by_name(ws, name="HeartFailure Data")
    dataset = dataset.to_pandas_dataframe()
    x = dataset.drop(columns=['DEATH_EVENT'])
    y = dataset['DEATH_EVENT']
    x_train, x_test, y_train, y_test = train_test_split(x,y)

    run.log("Regularization Strength:", np.float(args.C))
    run.log("Max iterations:", np.int(args.max_iter))

    model = LogisticRegression(C=args.C, max_iter=args.max_iter).fit(x_train, y_train)

    accuracy = model.score(x_test, y_test)
    run.log("Accuracy", np.float(accuracy))

    y_scores = model.predict_proba(x_test)
    auc = roc_auc_score(y_test,y_scores[:,1])    
    run.log('AUC', np.float(auc))

    os.makedirs('outputs', exist_ok=True)
    
    joblib.dump(model, 'outputs/model.joblib')
    
    run.log("Accuracy", np.float(accuracy))

if __name__ == '__main__':
    main()