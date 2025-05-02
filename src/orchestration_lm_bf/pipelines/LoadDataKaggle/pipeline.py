"""
This is a boilerplate pipeline 'LoadDataKaggle'
generated using Kedro 0.19.12
"""

from kedro.pipeline import node, Pipeline, pipeline  # noqa
from .nodes import load_and_prepare_gtsrb_data

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=load_and_prepare_gtsrb_data,
            inputs=None,
            outputs="raw_images",
            name="load_and_prepare_gtsrb_data_node"
        )
    ])
