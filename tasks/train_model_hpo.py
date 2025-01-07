from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from dataclass_defs import SearchSpace, Hyperparameters, HpoResults
from dataclass_defs import DataSplits
from itertools import product


def create_search_grid(searchspace: SearchSpace) -> list[Hyperparameters]:

    keys = vars(searchspace).keys()
    values = [getattr(searchspace, key) for key in keys]

    grid = [Hyperparameters(**dict(zip(keys, combination)))
            for combination in product(*values)]

    return grid


def train_classifier_hpo(
        cfg: dict,
        hp: Hyperparameters,
        splits: DataSplits) -> HpoResults:

    clf = RandomForestClassifier(
        max_depth=hp.max_depth,
        max_leaf_nodes=hp.max_leaf_nodes,
        n_estimators=hp.n_estimators)
    X_train = splits.X_train.dataframe
    X_test = splits.X_test.dataframe
    y_train = splits.y_train.dataframe
    y_test = splits.y_test.dataframe

    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    acc = metrics.accuracy_score(y_test, y_pred)
    print("ACCURACY OF THE MODEL:", acc)
    retVal = HpoResults(hp, acc)
    retVal.model = clf
    return retVal
