from enum import Enum
from data_utils.constants import DataSetNameClass


def get_dataset_pattern(dataset_name: DataSetNameClass) -> str:
    """
    Get the regex pattern for dataset names based on the dataset type.

    Args:
        dataset_name: The dataset name class enum value

    Returns:
        str: The regex pattern for matching dataset names
    """
    if dataset_name == DataSetNameClass.BASE:
        # Match g1–g12 only, with "_new"
        pattern = r'^g([1-9]|1[0-2])_point_cloud_\d+_new$'
    elif dataset_name == DataSetNameClass.DIFFERENT_PARTICIPANTS:
        # Match g1–g12 only, with double underscore
        pattern = r'^g([1-9]|1[0-2])_point_cloud__\d+$'
    elif dataset_name == DataSetNameClass.NEW_CLASSES:
        # Match only g13–g16 folders
        pattern = r'^g1[3-6]'
    else:
        # Match g1–g12 only for other dataset types
        pattern = rf'^g([1-9]|1[0-2])_point_cloud__{dataset_name}_\d+$'

    return pattern
