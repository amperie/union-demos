from fastapi import FastAPI
import pandas as pd
import os
from demos.tasks.dataclass_defs import HpoResults
from demos.tasks.featurize_data import featurize
from pydantic import BaseModel


class ModelFeatures(BaseModel):
    purpose: str
    int_rate: float
    installment: float
    log_annual_inc: float
    dti: float
    fico: int
    days_with_cr_line: float
    revol_bal: int
    revol_util: float
    inq_last_6mths: int
    delinq_2yrs: int
    pub_rec: int
    not_fully_paid: int


app = FastAPI()
obj = os.getenv("CLS_MODEL_RESULTS")
obj = HpoResults.from_flytedir(obj)


@app.get("/")
async def root(mf: ModelFeatures):
    return mf


@app.post("/")
async def root_post(mf: ModelFeatures):
    df = pd.DataFrame([mf.model_dump()])
    fdf = featurize(df)
    prediction = obj.model.predict_proba(fdf)
    return "{'prediction': " + str(prediction) + "}"
