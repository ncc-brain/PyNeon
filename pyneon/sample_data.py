from pathlib import Path
import requests
import zipfile

data_dir = Path(__file__).parent.parent / "data"

data_url_dict = {
    "OfficeWalk": "https://osf.io/download/3gvyp/",
}


def get_sample_data(data_name: str) -> Path:
    if data_name not in data_url_dict:
        raise ValueError(
            f"Unknown data_name: {data_name}, can only be one of {list(data_url_dict.keys())}"
        )
    print(data_dir)
    # Download the data to the data directory
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
    get_sample_data("OfficeWalk")