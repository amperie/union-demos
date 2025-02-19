from demos.tasks.dataclass_defs import HpoResults
from demos.tasks.get_training_split import get_training_split


def get_predictions(obj: HpoResults):

    data = get_training_split(obj.data)
    X_train = data.get("X_train")
    X_test = data.get("X_test")
    y_train = data.get("y_train")
    y_test = data.get("y_test")
    clf = obj.model

    yhat_prob_train = clf.predict_proba(X_train)[:, 1]
    yhat_prob_test = clf.predict_proba(X_test)[:, 1]

    return y_train, yhat_prob_train, y_test, yhat_prob_test
