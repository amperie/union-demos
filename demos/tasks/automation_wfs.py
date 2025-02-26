import union
from union import Artifact
from union import FlyteDirectory
from flytekit import Deck
from flytekit.deck import MarkdownRenderer
import union.app


image = union.ImageSpec(
    base_image="ghcr.io/unionai-oss/union:py3.10-latest",
    name="data",
    registry="pablounionai",
    packages=["scikit-learn", "datasets", "pandas", "union"]
)

ClsModelResults = Artifact(
    name="pablo_classifier_model_results_fd"
)
cls = ClsModelResults.query()
print(cls)


@union.task(
    container_image=image,
    enable_deck=True
)
def tsk_deploy_app(input: FlyteDirectory):
    md_text =\
        '<a href="https://bold-king-90739.apps.demo.hosted.unionai.cloud/">'\
        'App Deployment</a>'
    m = MarkdownRenderer()
    deck = Deck("App Deployment", m.to_html(md_text))
    deck.append(m.to_html(md_text))
    print(input)


@union.workflow
def model_automation_rwf(input: FlyteDirectory = cls):
    tsk_deploy_app(input)
