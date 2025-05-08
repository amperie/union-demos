import union
import pandas as pd

d = {"a": 1, "b": 2, "c": 3}
nd = {"nd": d}
n2d = {"n2d": nd}
pass


@union.task
def tsk(d_in: dict) -> None:
    print(d_in)


@union.workflow
def wf():
    tsk(n2d)


@union.task
def tsk2(d_in: dict) -> pd.Dataframe:
    pass
    return pd.DataFrame()


folder = "tmpStorage"
model.save(folder)
return FlyteDirectory(folder)

def use_model(model_flyte_dir: FlyteDirectory):
    model.load(model_flyte_dir)
union.Resources()