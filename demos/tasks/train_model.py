from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.base import BaseEstimator
import pandas as pd


def train_classifier(
        cfg: dict,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series) -> BaseEstimator:

    if "n_estimators" in cfg:
        n_estimators = cfg['n_estimators']
    else:
        n_estimators = 10

    clf = RandomForestClassifier(n_estimators=n_estimators)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print("ACCURACY OF THE MODEL:", metrics.accuracy_score(y_test, y_pred))
    return clf
