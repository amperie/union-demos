import pandas as pd
from demos.tasks.dataclass_defs import HpoResults


def get_predictions_old():
    """
    Get predictions from model

    Returns:
        tuple: y_train, yhat_prob_train, y_test, yhat_prob_test
    """

    X_train, X_test, y_train, y_test = load_data()
    clf = train_model(X_train, y_train)

    yhat_prob_train = clf.predict_proba(X_train)[:, 1]
    yhat_prob_test = clf.predict_proba(X_test)[:, 1]

    return y_train, yhat_prob_train, y_test, yhat_prob_test


def get_predictions():
    from union.remote import UnionRemote

    remote = UnionRemote.for_endpoint("demo.hosted.unionai.cloud")
    art = remote.get_artifact(
        "flyte://av0.2/demo/flytesnacks/development/"
        "pablo_classifier_model_results"
        "@adbglq57tlpsdndpb4b2/n4/0/o0")
    obj = art.get(HpoResults)
    model = obj.model

    yhat_prob_train = clf.predict_proba(X_train)[:, 1]
    yhat_prob_test = clf.predict_proba(X_test)[:, 1]

    return y_train, yhat_prob_train, y_test, yhat_prob_test
