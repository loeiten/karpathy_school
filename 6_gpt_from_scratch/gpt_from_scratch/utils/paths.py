"""Module for paths."""

from pathlib import Path


def get_gpt_from_scratch_package_dir() -> Path:
    """
    Return the absolute path to the gpt_from_scratch package.

    Returns:
        Path: The path to the root directory
    """
    return Path(__file__).absolute().parents[1]


def get_data_path() -> Path:
    """
    Return the absolute path to the data directory.

    Returns:
        Path: The path to the root directory
    """
    return (
        get_gpt_from_scratch_package_dir()
        .parents[1]
        .joinpath("data", "tiny-shakespare.txt")
    )


def get_output_dir() -> Path:
    """
    Return the absolute path to the dir for outputs.

    Returns:
        Path: The path for outputs
    """
    output_dir = get_gpt_from_scratch_package_dir().parent.joinpath("output")
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir
