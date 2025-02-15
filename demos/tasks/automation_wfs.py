import union
from union import Artifact
import union.app
from demos.tasks.dataclass_defs import HpoResults


image = union.ImageSpec(
    base_image="ghcr.io/unionai-oss/union:py3.10-latest",
    name="data",
    registry="pablounionai",
    packages=["scikit-learn", "datasets", "pandas", "union"]
)

ClsModelResults = Artifact(
    name="pablo_classifier_model_results"
)
cls = ClsModelResults.query()
print(cls)


@union.task(
    container_image=image,
)
def tsk_nothin(input: HpoResults):
    print(input)


@union.workflow
def model_automation_wf(input: HpoResults = cls):
    tsk_nothin(input)


if __name__ == "__main__":
    model_automation_wf()