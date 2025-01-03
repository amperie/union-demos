import pandas as pd
from flytekit import task


@task
def get_data(cfg: dict):
    data = {'Name': ['Tom', 'nick', 'krish', 'jack'],
            'Age': [20, 21, 19, 18]}

    # Create DataFrame
    df = pd.DataFrame(data)

    return df
