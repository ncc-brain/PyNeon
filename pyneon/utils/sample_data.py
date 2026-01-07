import zipfile
from pathlib import Path

import requests

data_dir = Path(__file__).parent.parent.parent / "data"

data_url_dict = {
    "boardView": "https://osf.io/download/t52b4/",
    "screenFlash": "https://osf.io/download/6836d6e49bf72b3880d6a388/",
    "PLR": "https://figshare.com/ndownloader/articles/30921452/versions/1?folder_path=PLR",
    "Artworks": "https://osf.io/download/685036cf0675b635986df74f/",
}


def get_sample_data(data_name: str, replace=False) -> Path:
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
    return data_dir / data_name


if __name__ == "__main__":
    data_dir = get_sample_data("PLR")
    assert data_dir.exists()
