from pathlib import Path

import pandas as pd

from .recording import Recording


class Dataset:
    """
    Holder for multiple recordings. It reads from a directory containing a multiple
    recordings downloaded from Pupil Cloud with the **Timeseries CSV** or
    **Timeseries CSV and Scene Video** option. For example, a dataset with 2 recordings
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

    Individual recordings will be read into :class:`pyneon.Recording` objects based on
    ``sections.csv``. They are accessible through the ``recordings`` attribute.

    Parameters
    ----------
    dataset_dir : str or pathlib.Path
        Path to the directory containing the dataset.
    custom : bool, optional
        Whether to expect a custom dataset structure. If ``False``, the dataset
        is expected to follow the standard Pupil Cloud dataset structure with a
        ``sections.csv`` file. If True, every directory in ``dataset_dir`` is
        considered a recording directory, and the ``sections`` attribute is
        constructed from the ``info`` of recordings found.
        Defaults to ``False``.

    Attributes
    ----------
    dataset_dir : pathlib.Path
        Path to the directory containing the dataset.
    recordings : list of Recording
        List of :class:`pyneon.Recording` objects for each recording in the dataset.
    sections : pandas.DataFrame
        DataFrame containing the sections of the dataset.

    """

    def __init__(self, dataset_dir: str | Path, custom: bool = False):
        dataset_dir = Path(dataset_dir)
        if not dataset_dir.is_dir():
            raise FileNotFoundError(f"Directory not found: {dataset_dir}")

        self.dataset_dir = dataset_dir
        self.recordings = list()

        if not custom:
            sections_path = dataset_dir.joinpath("sections.csv")
            if not sections_path.is_file():
                raise FileNotFoundError(f"sections.csv not found in {dataset_dir}")
            self.sections = pd.read_csv(sections_path)

            recording_ids = self.sections["recording id"]

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
                    raise RuntimeWarning(
                        f"Skipping reading recording {rec_dir.name} due to error:\n{e}"
                    )

            # Rebuild a `sections` DataFrame from the Recording objects
            sections = []
            for i, rec in enumerate(self.recordings):
                sections.append(
                    {
                        "section id": i,
                        "recording id": rec.recording_id,
                        "recording name": rec.recording_id,
                        "wearer id": rec.info["wearer_id"],
                        "wearer name": rec.info["wearer_name"],
                        "section start time [ns]": rec.start_time,
                        "section end time [ns]": rec.start_time + rec.info["duration"],
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

    def load_enrichment(self, enrichment_dir: str | Path):
        """
        Load enrichment information from an enrichment directory. The directory must
        contain an enrichment_info.txt file. Enrichment data will be parsed for each
        recording ID and added to Recording object in the dataset.

        The method is currently being developed and is not yet implemented.

        Parameters
        ----------
        enrichment_dir : str or pathlib.Path
            Path to the directory containing the enrichment information.
        """
        raise NotImplementedError("Enrichment loading is not yet implemented.")
