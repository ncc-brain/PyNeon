from pathlib import Path
from typing import Union

import pandas as pd

from .recording import NeonRecording


class NeonDataset:
    """Holder for a dataset of multiple recordings."""

    def __init__(self, dataset_dir: Union[str, Path]):
        dataset_dir = Path(dataset_dir)
        if not dataset_dir.is_dir():
            raise FileNotFoundError(f"Directory not found: {dataset_dir}")

        self.dataset_dir = dataset_dir
        sections_path = dataset_dir.joinpath("sections.csv")
        if not sections_path.is_file():
            raise FileNotFoundError(f"sections.csv not found in {dataset_dir}")

        self.recordings = list()
        self.sections = pd.read_csv(sections_path)
        recording_ids = self.sections["recording id"]

        for rec_id in recording_ids:
            rec_id_start = rec_id.split("-")[0]
            rec_dir = [d for d in dataset_dir.glob(f"*-{rec_id_start}") if d.is_dir()]
            if len(rec_dir) == 1:
                rec_dir = rec_dir[0]
                try:
                    self.recordings.append(NeonRecording(rec_dir))
                except Exception as e:
                    raise RuntimeWarning(
                        f"Skipping reading recording {rec_id} " f"due to error:\n{e}"
                    )
            elif len(rec_dir) == 0:
                raise FileNotFoundError(
                    "Recording directory not found for recording id " f"{rec_id_start}"
                )
            else:
                raise FileNotFoundError(
                    f"Multiple recording directories found for recording id "
                    f"{rec_id_start}"
                )

    def __repr__(self):
        return f"NeonDataset | {len(self.recordings)} recordings"

    def __len__(self):
        return len(self.recordings)

    def __getitem__(self, index):
        return self.recordings[index]
