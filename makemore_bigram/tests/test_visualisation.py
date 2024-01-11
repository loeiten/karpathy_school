"""Contains tests for the visualisation module."""

from pathlib import Path

from _pytest._py.path import LocalPath
from makemore_bigram.visualisation import main


def test_visualisation_main(tmpdir: LocalPath) -> None:
    """Test the main function of the visualisation module.

    Args:
        tmpdir (LocalPath): Temporary directory
    """
    tmp_dir = Path(tmpdir)
    main(tmp_dir)
    assert len(list(tmp_dir.glob("heatmap.png"))) == 1
