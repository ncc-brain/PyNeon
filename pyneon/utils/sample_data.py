import zipfile
from pathlib import Path
from typing import Literal, Optional

import requests

data_dir = Path(__file__).parent.parent.parent / "data"

data_url_dict = {
    "simple": "https://osf.io/download/gv46n/",
    "markers": "https://osf.io/download/t56b2/",
    "PLR": "https://osf.io/download/5kmwp/",
}


def get_sample_data(
    data_name: str,
    replace: bool = False,
    format: Optional[Literal["cloud", "native"]] = None,
) -> Path:
    """
    Download and retrieve sample data for PyNeon.

    This function downloads sample data from predefined URLs if not already present,
    and returns the path to the data directory. Optionally, it can return paths to
    specific recording formats.

    Parameters
    ----------
    data_name : str
        Name of the sample dataset to retrieve. Must be one of:
        - "simple": Basic sample recording
        - "PLR": Pupil light reflex data
        - "markers": Recording with visual markers for surface mapping
    replace : bool, optional
        If True, re-download the data even if it already exists locally.
        Defaults to False.
    format : {'cloud', 'native'}, optional
        If specified, returns the path to a specific recording format:
        - "cloud": Returns path to "Timeseries Data + Scene Video" directory
        - "native": Returns path to "Native Recording Data" directory
        If None, returns the root data directory. Defaults to None.

    Returns
    -------
    Path
        Path to the requested data directory.

    Raises
    ------
    ValueError
        If `data_name` is not one of the available sample datasets.

    Examples
    --------
    >>> data_path = get_sample_data("simple")
    >>> print(data_path)
    .../data/simple

    >>> cloud_path = get_sample_data("PLR", format="cloud")
    >>> print(cloud_path)
    .../data/PLR/Timeseries Data + Scene Video
    """
    if data_name not in data_url_dict:
        raise ValueError(
            f"Unknown data_name: {data_name}, can only be one of {list(data_url_dict.keys())}"
        )
    if (not (data_dir / data_name).exists()) or replace:
        data_dir.mkdir(parents=True, exist_ok=True)
        data_url = data_url_dict[data_name]
        zip_path = data_dir / f"{data_name}.zip"
        with open(zip_path, "wb") as f:
            response = requests.get(data_url)
            f.write(response.content)
        # Unzip the data
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(data_dir)
        # Remove the zip
        zip_path.unlink()
    if format is not None:
        if format == "cloud":
            return data_dir / data_name / "Timeseries Data + Scene Video"
        elif format == "native":
            return data_dir / data_name / "Native Recording Data"
    else:
        return data_dir / data_name


if __name__ == "__main__":
    data_dir = get_sample_data("simple")
    assert data_dir.exists()
    print(f"Sample data is ready at {data_dir}")
