from clearml.automation.controller import PipelineDecorator
from clearml import TaskTypes


@PipelineDecorator.component(
    cache=True,
    #docker_bash_setup_script="apt update && apt install ffmpeg libsm6 libxext6 -y",
    packages="../requirements-load-dataset.txt",
    execution_queue="aws-gpu-g4dn-xl",
    return_values=["train_dataset_path"],
    task_type=TaskTypes.data_processing,
    docker="nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04",
    docker_args="--shm-size=512m",
    repo="https://pdmct@bitbucket.org/chedanalytics/asset-verification-mmclassification.git",
    repo_commit="bfff8c22a7339ccb339f48805bd190768ea345f1",
)
def data_download():
    print("loading dataset dataset")
    from clearml import Dataset

    dataset = Dataset.get(
        dataset_id="2ab02dad51ed4b07b3efbe676ff1ca40",
        dataset_version="train - 1.0.3",
        alias="train-drone-plus-synthetic-prc-cls-all-assest-classes",
        dataset_tags=None,
        only_completed=True,
        only_published=False,
    )
    train_dataset_path = dataset.get_local_copy()
    return train_dataset_path


@PipelineDecorator.pipeline(
    name="test-clearml-cache",
    project="test-clearml-cache",
    version="1.0.0",
    pipeline_execution_queue="aws-gpu-g4dn-xl",
    start_controller_locally=True,
)
def data_download_pipeline():
    """Start mmclassification pipeline."""
    train_dataset_path = data_download()
    print(train_dataset_path)


if __name__ == "__main__":
    data_download_pipeline()
