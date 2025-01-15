from pathlib import Path

import pandas as pd

from .recording import NeonRecording


class NeonDataset:
    """
    Holder for multiple recordings. It reads from a directory containing a **project**
    downloaded from Pupil Cloud with the **Timeseries CSV** or **Timeseries CSV and
    Scene Video** option. For example, a dataset with 2 recordings would have the
    following folder structure:

    .. code-block:: text

        dataset_dir/
        ├── recording_dir_1/
        │   ├── info.json
        │   ├── gaze.csv
        |   └── ...
        ├── recording_dir_2/
        │   ├── info.json
        │   ├── gaze.csv
        |   └── ...
        ├── ...
        ├── enrichment_info.txt
        └── sections.csv

    Individual recordings will be read into NeonRecording objects based on
    ``sections.csv``. They are accessible through the ``recordings`` attribute.

    Parameters
    ----------
    dataset_dir : str or pathlib.Path
        Path to the directory containing the dataset.

    Attributes
    ----------
    dataset_dir : pathlib.Path
        Path to the directory containing the dataset.
    recordings : list of NeonRecording
        List of NeonRecording objects for each recording in the dataset.
    sections : pandas.DataFrame
        DataFrame containing the sections of the dataset.
    """

    def __init__(self, dataset_dir: str | Path):
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
                        f"Skipping reading recording {rec_id} due to error:\n{e}"
                    )
            elif len(rec_dir) == 0:
                raise FileNotFoundError(
                    f"Recording directory not found for recording id {rec_id_start}"
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
        """Get a NeonRecording by index."""
        return self.recordings[index]

    def load_enrichment(self, enrichment_dir: str | Path):
        """
        Load enrichment information from an enrichment directory. The directory must
        contain an enrichment_info.txt file. Enrichment data will be parsed for each
        recording ID and added to NeonRecording object in the dataset.

        The method is currently being developed and is not yet implemented.

        Parameters
        ----------
        enrichment_dir : str or pathlib.Path
            Path to the directory containing the enrichment information.
        """
        pass
