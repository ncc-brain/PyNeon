from pathlib import Path
import requests
import zipfile

data_dir = Path(__file__).parent.parent.parent / "data"

data_url_dict = {
    "boardView": "https://osf.io/download/t52b4/",
    "ArtLab": "https://osf.io/download/nb53p/",
    "CircleFlash": "https://osf.io/download/6793b3540717d27236b52d7f/",
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
    get_sample_data("CircleFlash")
