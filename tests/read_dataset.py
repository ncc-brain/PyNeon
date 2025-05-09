from pathlib import Path
from pyneon import Dataset

dataset_dir = Path(__file__).parent / "data" / "Timeseries Data + Scene Video"
dataset = Dataset(dataset_dir)

print(dataset)
