from pathlib import Path
from pyneon import NeonDataset

dataset_dir = Path(__file__).parent / "data" / "Timeseries Data + Scene Video"
dataset = NeonDataset(dataset_dir)

print(dataset)
