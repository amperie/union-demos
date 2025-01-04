from .get_data_hf import get_data_hf
from .featurize_data import featurize
from union import workflow


@workflow
def wf():
    df = get_data_hf({})
    fdf = featurize({}, df)
