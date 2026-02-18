from pathlib import Path
from warnings import warn

import pandas as pd

from .recording import Recording


class Dataset:
    """
    Container for multiple recordings. Reads from a directory containing multiple
    recordings.

    For example, a dataset with 2 recordings downloaded from Pupil Cloud
    would have the following folder structure:

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

    Or a dataset with multiple native recordings:

    .. code-block:: text

        dataset_dir/
        ├── recording_dir_1/
        │   ├── info.json
        │   ├── blinks ps1.raw
        |   ├── blinks ps1.time
        |   ├── blinks.dtype
        |   └── ...
        └── recording_dir_2/
            ├── info.json
            ├── blinks ps1.raw
            ├── blinks ps1.time
            ├── blinks.dtype
            └── ...

    Individual recordings will be read into :class:`pyneon.Recording` instances
    (based on ``sections.csv``, if available) and accessible through the
    ``recordings`` attribute.

    Parameters
    ----------
    dataset_dir : str or pathlib.Path
        Path to the directory containing the dataset.

    Attributes
    ----------
    dataset_dir : pathlib.Path
        Path to the directory containing the dataset.
    recordings : list of Recording
        List of :class:`pyneon.Recording` instances for each recording in the dataset.
    sections : pandas.DataFrame
        DataFrame containing the sections of the dataset.

    Examples
    --------
    >>> from pyneon import Dataset
    >>> dataset = Dataset("path/to/dataset")
    >>> print(dataset)

    Dataset | 2 recordings

    >>> rec = dataset.recordings[0]
    >>> print(rec)

    Data format: cloud
    Recording ID: 56fcec49-d660-4d67-b5ed-ba8a083a448a
    Wearer ID: 028e4c69-f333-4751-af8c-84a09af079f5
    Wearer name: Pilot
    Recording start time: 2025-12-18 17:13:49.460000
    Recording duration: 8235000000 ns (8.235 s)
    """

    def __init__(self, dataset_dir: str | Path):
        dataset_dir = Path(dataset_dir)
        if not dataset_dir.is_dir():
            raise FileNotFoundError(f"Directory not found: {dataset_dir}")

        self.dataset_dir: Path = dataset_dir
        self.recordings: list[Recording] = list()

        sections_path = dataset_dir / "sections.csv"

        if sections_path.is_file():
            self.sections = pd.read_csv(sections_path)
            recording_ids = self.sections["recording id"]

            # Assert if recording IDs are correct
            for rec_id in recording_ids:
                rec_id_start = rec_id.split("-")[0]
                rec_dir = [
                    d for d in dataset_dir.glob(f"*-{rec_id_start}") if d.is_dir()
                ]
                if len(rec_dir) == 1:
                    rec_dir = rec_dir[0]
                    try:
                        self.recordings.append(Recording(rec_dir))
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
        else:  # Do not expect sections.csv and construct `sections` from recordings
            rec_dirs = [d for d in dataset_dir.iterdir() if d.is_dir()]
            for rec_dir in rec_dirs:
                try:
                    self.recordings.append(Recording(rec_dir))
                except Exception as e:
                    warn(
                        f"Skipping directory {rec_dir} due to error:\n{e}\n"
                        "Ensure it contains a valid recording structure.",
                        RuntimeWarning,
                    )

            # Rebuild a `sections` DataFrame from the Recording instances
            sections = []
            for rec in self.recordings:
                sections.append(
                    {
                        "section id": None,
                        "recording id": rec.recording_id,
                        "recording name": None,
                        "wearer id": rec.info.get("wearer_id", None),
                        "wearer name": rec.info.get("wearer_name", None),
                        "section start time [ns]": rec.start_time,
                        "section end time [ns]": rec.start_time
                        + rec.info.get("duration", 0),
                    }
                )

            self.sections = pd.DataFrame(sections)

    def __repr__(self):
        return f"Dataset | {len(self.recordings)} recordings"

    def __len__(self):
        return len(self.recordings)

    def __getitem__(self, index: int) -> Recording:
        """Get a Recording by index."""
        return self.recordings[index]
