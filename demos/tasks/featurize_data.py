import pandas as pd
from union import current_context
from union import ImageSpec
from flytekitplugins.spark import Spark
from flytekitplugins.spark import DatabricksV2 as Databricks


def check_schema(df: pd.DataFrame) -> pd.DataFrame:
    if "purpose_debt_consolidation" not in df.columns:
        df["purpose_debt_consolidation"] = 0
    if "purpose_credit_card" not in df.columns:
        df["purpose_credit_card"] = 0
    if "purpose_small_business" not in df.columns:
        df["purpose_home_improvement"] = 0
    if "purpose_all_other" not in df.columns:
        df["purpose_all_other"] = 0
    if "purpose_educational" not in df.columns:
        df["purpose_educational"] = 0
    if "purpose_major_purchase" not in df.columns:
        df["purpose_major_purchase"] = 0
    if "purpose_small_business" not in df.columns:
        df["purpose_small_business"] = 0
    df = df.reindex(sorted(df.columns), axis=1)
    return df


def featurize(df: pd.DataFrame) -> pd.DataFrame:

    df_encoded = pd.get_dummies(df, columns=['purpose'])
    return check_schema(df_encoded)


spark_conf = {
    "spark.driver.memory": "1000M",
    "spark.executor.memory": "1000M",
    "spark.executor.cores": "1",
    "spark.executor.instances": "2",
    "spark.driver.cores": "1",
    "spark.jars":
        "https://storage.googleapis.com/"
        "hadoop-lib/gcs/gcs-connector-hadoop3-latest.jar",
}

spark_image_spec = ImageSpec(
        builder="envd",
        name="spark",
        registry="pablounionai",
        packages=["spark", "pandas", "datasets",
                  "flytekitplugins-spark", "union",
                  "flytekit", "scikit-learn"],
    )

spark_task_config = Spark(spark_conf=spark_conf)

databricks_image_spec = ImageSpec(
    builder="envd",
    name="spark",
    base_image="ghcr.io/unionai-oss/databricks:kmeans",
    registry="pablounionai",
    packages=[
        "pyspark", "numpy", "union", "flytekitplugins-spark",
        "flytekit", "datasets", "pandas", "scikit-learn"],
    source_root="."
)

databricks_task_config = Databricks(
    spark_conf=spark_conf,
    databricks_conf={
        "run_name": "featurization",
        "runtime_engine": "PHOTON",
        "new_cluster": {
            "spark_version": "14.3.x-scala2.12",
            "node_type_id": "r6id.xlarge",
            "num_workers": 1,
            "aws_attributes": {
                "availability": "SPOT_WITH_FALLBACK",
                "instance_profile_arn":
                    "arn:aws:iam::339713193121:instance-profile"
                    "/databricks-demo",
                "first_on_demand": 1,
                "zone_id": "auto",
            },
        },
        "timeout_seconds": 3600,
        "max_retries": 3,
    },
    databricks_instance="dbc-ca63b07f-c54a.cloud.databricks.com",
)


def featurize_spark(df: pd.DataFrame) -> pd.DataFrame:
    sess = current_context().spark_session
    # Spark bug handling
    df.iteritems = df.items

    df_sp = sess.createDataFrame(df)

    # Do stuff
    df_sp = df_sp.drop("purpose")

    return check_schema(df_sp.toPandas())
    # return df
