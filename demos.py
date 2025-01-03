from  tasks.get_data import get_data
from flytekit import workflow


@workflow
def union_demo_wf():
    get_data({})
