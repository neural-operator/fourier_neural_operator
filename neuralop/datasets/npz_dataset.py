from pathlib import Path
from typing import Optional, Union

from numpy import load as np_load

DEFAULT_DATA_HOME = Path.home().joinpath("neuraloperator_data")


def _get_data_home(data_home: Optional[Union[str, Path]] = None) -> Path:
    if data_home is None:
        data_home = DEFAULT_DATA_HOME
    if isinstance(data_home, str):
        data_home = Path(data_home)

    if not data_home.exists():
        data_home.mkdir()

    return data_home


def fetch_npz(
    file_name: str,
    _zenodo_url: str,
    data_home: Optional[Union[str, Path]] = None,
):
    """Loads a target ``.npz`` file.

    This does not download the target file (though future versions of this
    function will automatically download the specified file
    if it's not on disk).

    Parameters
    ----------
    data_home : str, default=None
        Specify a download folder for the datasets. If None,
        all neuraloperator data is stored in "~/neuraloperator_data" subfolders.
    """
    data_home = _get_data_home(data_home)
    file_path = data_home / file_name
    try:
        return np_load(str(file_path))
    except FileNotFoundError:
        raise NotImplementedError(
            f"File {file_path} not found on disk. "
            f"Please make sure {file_name} is in the target directory "
            f"and try again"
        )
