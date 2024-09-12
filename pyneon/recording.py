from pathlib import Path
from typing import Union
import pandas as pd
import json
from datetime import datetime
import warnings
import numpy as np
import cv2

from .data import (
    NeonGaze,
    NeonIMU,
    NeonEyeStates,
    NeonBlinks,
    NeonFixations,
    NeonSaccades,
    NeonEvents,
    NeonVideo,
)
from .preprocess import concat_streams, concat_events
from .io import export_motion_bids, exports_eye_bids


def _check_file(dir_path: Path, stem: str):
    csv = dir_path / f"{stem}.csv"
    if csv.is_file():
        return True, csv.name, csv
    elif len(files := sorted(dir_path.glob(f"{stem}*"))) > 0:
        files_name = ", ".join([f.name for f in files])
        return True, files_name, files
    else:
        return False, None, None


class NeonRecording:
    """
    Data from a single recording. The recording directory could be downloaded from
    either a single recording or a project on Pupil Cloud. In either case, the directory
    must contain an ``info.json`` file. For example, a recording directory could have the
    following structure:

    .. code-block:: text

        recording_dir/
        ├── info.json (REQUIRED)
        ├── gaze.csv
        ├── 3d_eye_states.csv
        ├── imu.csv
        ├── blinks.csv
        ├── fixations.csv
        ├── saccades.csv
        ├── events.csv
        ├── labels.csv
        ├── world_timestamps.csv
        ├── scene_camera.json
        └── <scene_video>.mp4 (if present)

    Streams, events, (and scene video) will be located but not loaded until
    accessed as properties such as ``gaze``, ``imu``, and ``eye_states``.

    Parameters
    ----------
    recording_dir : str or :class:`pathlib.Path`
        Path to the directory containing the recording.

    Attributes
    ----------
    recording_id : str
        Recording ID.
    recording_dir : :class:`pathlib.Path`
        Path to the recording directory.
    info : dict
        Information about the recording. Read from ``info.json``. For details, see
        https://docs.pupil-labs.com/neon/data-collection/data-format/#info-json.
    start_datetime : :class:`datetime.datetime`
        Start time of the recording as in ``info.json``.
        May not match the start time of each data stream.
    contents : :class:`pandas.DataFrame`
        Contents of the recording directory. Each index is a stream or event name
        (e.g. ``gaze`` or ``imu``) and columns are ``exist`` (bool),
        ``filename`` (str), and ``path`` (Path).
    """

    def __init__(self, recording_dir: Union[str, Path]):
        recording_dir = Path(recording_dir)
        if not recording_dir.is_dir():
            raise FileNotFoundError(f"Directory not found: {recording_dir}")
        if not (info_path := recording_dir / "info.json").is_file():
            raise FileNotFoundError(f"info.json not found in {recording_dir}")

        with open(info_path) as f:
            self.info = json.load(f)
        self.start_datetime = datetime.fromtimestamp(self.info["start_time"] / 1e9)

        self.recording_id = self.info["recording_id"]
        self.recording_dir = recording_dir

        self._gaze = None
        self._eye_states = None
        self._imu = None
        self._blinks = None
        self._fixations = None
        self._saccades = None
        self._events = None
        self._video = None

        self._get_contents()

    def __repr__(self) -> str:
        return f"""
Recording ID: {self.recording_id}
Wearer ID: {self.info['wearer_id']}
Wearer name: {self.info['wearer_name']}
Recording start time: {self.start_datetime}
Recording duration: {self.info["duration"] / 1e9} s
{self.contents.to_string()}
"""

    def _get_contents(self):
        contents = pd.DataFrame(
            index=[
                "3d_eye_states",
                "blinks",
                "events",
                "fixations",
                "gaze",
                "imu",
                "labels",
                "saccades",
                "world_timestamps",
                "scene_video",
            ],
            columns=["exist", "filename", "path"],
        )
        # Check for CSV files
        for stem in contents.index:
            contents.loc[stem, :] = _check_file(self.recording_dir, stem)

        # Check for scene video
        if len(video_path := list(self.recording_dir.glob("*.mp4"))) == 1:
            contents.loc["scene_video", :] = (True, video_path[0].name, video_path[0])
            if (camera_info := self.recording_dir / "scene_camera.json").is_file():
                with open(camera_info) as f:
                    self.camera_info = json.load(f)
            else:
                raise FileNotFoundError(
                    "Scene video has no accompanying scene_camera.json in "
                    f"{self.recording_dir}"
                )
        elif len(video_path) > 1:
            raise FileNotFoundError(
                f"Multiple scene video files found in {self.recording_dir}"
            )
        self.contents = contents

    @property
    def gaze(self) -> Union[NeonGaze, None]:
        """
        Returns a NeonGaze object or None if no gaze data is found.
        """
        if self._gaze is None:
            if self.contents.loc["gaze", "exist"]:
                gaze_file = self.contents.loc["gaze", "path"]
                self._gaze = NeonGaze(gaze_file)
            else:
                warnings.warn("Gaze data not loaded because no file was found.")
        return self._gaze

    @property
    def imu(self) -> Union[NeonIMU, None]:
        """
        Returns a NeonIMU object or None if no IMU data is found.
        """
        if self._imu is None:
            if self.contents.loc["imu", "exist"]:
                imu_file = self.contents.loc["imu", "path"]
                self._imu = NeonIMU(imu_file)
            else:
                warnings.warn("IMU data not loaded because no file was found.")
        return self._imu

    @property
    def eye_states(self) -> Union[NeonEyeStates, None]:
        """
        Returns a NeonEyeStates object or None if no eye states data is found.
        """
        if self._eye_states is None:
            if self.contents.loc["3d_eye_states", "exist"]:
                eye_states_file = self.contents.loc["3d_eye_states", "path"]
                self._eye_states = NeonEyeStates(eye_states_file)
            else:
                warnings.warn(
                    "3D eye states data not loaded because no file was found."
                )
        return self._eye_states

    @property
    def blinks(self) -> Union[NeonBlinks, None]:
        """
        Returns a NeonBlinks object or None if no blinks data is found.
        """
        if self._blinks is None:
            if self.contents.loc["blinks", "exist"]:
                blinks_file = self.contents.loc["blinks", "path"]
                self._blinks = NeonBlinks(blinks_file)
            else:
                warnings.warn("Blinks data not loaded because no file was found.")
        return self._blinks

    @property
    def fixations(self) -> Union[NeonFixations, None]:
        """
        Returns a NeonFixations object or None if no fixations data is found.
        """
        if self._fixations is None:
            if self.contents.loc["fixations", "exist"]:
                fixations_file = self.contents.loc["fixations", "path"]
                self._fixations = NeonFixations(fixations_file)
            else:
                warnings.warn("Fixations data not loaded because no file was found.")
        return self._fixations

    @property
    def saccades(self) -> Union[NeonSaccades, None]:
        """
        Returns a NeonSaccades object or None if no saccades data is found.
        """
        if self._saccades is None:
            if self.contents.loc["saccades", "exist"]:
                saccades_file = self.contents.loc["saccades", "path"]
                self._saccades = NeonSaccades(saccades_file)
            else:
                warnings.warn("Saccades data not loaded because no file was found.")
        return self._saccades

    @property
    def events(self) -> Union[NeonEvents, None]:
        """
        Returns a NeonEvents object or None if no events data is found.
        """
        if self._events is None:
            if self.contents.loc["events", "exist"]:
                events_file = self.contents.loc["events", "path"]
                self._events = NeonEvents(events_file)
            else:
                warnings.warn("Events data not loaded because no file was found.")
        return self._events

    @property
    def video(self) -> Union[NeonVideo, None]:
        """
        Returns a NeonVideo object or None if no scene video is found.
        """
        if self._video is None:
            if (
                self.contents.loc["scene_video", "exist"]
                and self.contents.loc["world_timestamps", "exist"]
            ):
                video_file = self.contents.loc["scene_video", "path"]
                timestamp_file = self.contents.loc["world_timestamps", "path"]
                self._video = NeonVideo(video_file, timestamp_file)
            else:
                warnings.warn(
                    "Scene video not loaded because no video or video timestamps file was found."
                )
        return self._video

    def concat_streams(
        self,
        stream_names: Union[str, list[str]],
        sampling_freq: Union[float, int, str] = "min",
        resamp_float_kind: str = "linear",
        resamp_other_kind: str = "nearest",
        inplace: bool = False,
    ) -> pd.DataFrame:
        """
        Concatenate data from different streams under common timestamps.
        Since the streams may have different timestamps and sampling frequencies,
        resampling of all streams to a set of common timestamps is performed.
        The latest start timestamp and earliest last timestamp of the selected sreams
        are used to define the common timestamps.

        Parameters
        ----------
        stream_names : str or list of str
            Stream names to concatenate. If "all", then all streams will be used.
            If a list, items must be in
            ``{"gaze", "imu", "eye_states", "3d_eye_states"}``.
        sampling_freq : float or int or str, optional
            Sampling frequency to resample the streams to.
            If numeric, the streams will be resampled to this frequency.
            If ``"min"``, the lowest nominal sampling frequency
            of the selected streams will be used.
            If ``"max"``, the highest nominal sampling frequency will be used.
            Defaults to ``"min"``.
        resamp_float_kind : str, optional
            Kind of interpolation applied on columns of float type,
            Defaults to ``"linear"``. For details see :class:`scipy.interpolate.interp1d`.
        resamp_other_kind : str, optional
            Kind of interpolation applied on columns of other types.
            Defaults to ``"nearest"``.
        inplace : bool, optional
            Replace selected stream data with resampled data during concatenation
            if``True``. Defaults to ``False``.

        Returns
        -------
        concat_data : :class:`pandas.DataFrame`
            Concatenated data.
        """
        return concat_streams(
            self,
            stream_names,
            sampling_freq,
            resamp_float_kind,
            resamp_other_kind,
            inplace,
        )

    def concat_events(self, event_names: list[str]) -> pd.DataFrame:
        """
        Concatenate types of events and return a DataFrame with all events.

        Parameters
        ----------
        event_names : list of str
            List of event names to concatenate. Event names must be in
            ``{"blinks", "fixations", "saccades", "events"}``.

        Returns
        -------
        concat_events : :class:`pandas.DataFrame`
            Concatenated events.
        """
        return concat_events(self, event_names)

    def map_gaze_to_video(
        self, output_pkl: Union[str, Path] = "data/gaze_mapped_to_video.pkl"
    ):
        """
        Tracks fixations across video frames using gaze data and world timestamps.

        Parameters
        ----------
        output_pkl : str or :class:pathlib.Path, optional
            Path to save the pickle file with fixations per frame, by default 'fixations_per_frame.pkl'.
        """

        # Read the first frame to get the initial grayscale image
        ret, first_frame = cap.read()
        if not ret:
            raise ValueError("Unable to read the first frame of the video.")
        prev_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)

        # Initialize current fixation tracking variables
        current_id = 0

        # Process each frame according to world timestamps
        for frame_idx, frame_time in enumerate(world_data):
            last_id = current_id

            # Find the closest gaze data point to the current frame time
            gaze_idx = np.argmin(np.abs(gaze_data["time"] - frame_time))
            active_gaze = gaze_data.iloc[gaze_idx]
            gaze_x = active_gaze["gaze x [px]"]
            gaze_y = active_gaze["gaze y [px]"]
            active_fixation = active_gaze["fixation id"]

            # Determine fixation status based on the fixation id
            if not pd.isna(active_fixation):
                current_id = int(active_fixation)
            else:
                current_id = 0

            # Determine the status of the fixation
            if current_id > last_id:
                flag = "onset"
                fixation_id = current_id
            elif current_id < last_id:
                flag = "offset"
                fixation_id = last_id
            elif current_id == last_id:
                if current_id == 0:
                    flag = "lost"
                    fixation_id = None
                else:
                    flag = "active"
                    fixation_id = current_id

            fixation_list = [fixation_id, gaze_x, gaze_y, flag]
            fixation_df = pd.DataFrame(
                [fixation_list], columns=["fixation_id", "x", "y", "status"]
            )

            if not fixation_df.empty:
                mapped_gaze = mapped_gaze._append(
                    {"frame": frame_idx, "fixations": fixation_df}, ignore_index=True
                )

        # Save the fixations DataFrame to a CSV and pickle file
        mapped_gaze.to_pickle(output_pkl)

        self.mapped_gaze = mapped_gaze
        cap.release()
        cv2.destroyAllWindows()

    def track_fixations_with_optical_flow(
        self, updated_pkl: Union[str, Path] = "data/past_fixations_mapped_to_video.pkl"
    ):
        """
        Applies optical flow to track fixations across frames in the video and update their positions.
        """
        # Load the fixations DataFrame from the previously saved pkl file
        if self.mapped_gaze is None:
            # run the map_gaze_to_video method to generate the fixations DataFrame
            self.map_gaze_to_video()

        mapped_gaze = self.mapped_gaze.copy()

        if not self.contents.loc["scene_video", "exist"]:
            raise FileNotFoundError(
                "Scene video file not found in the recording directory."
            )

        video_path = self.contents.loc["scene_video", "path"]
        cap = cv2.VideoCapture(str(video_path))

        # Initialize parameters for Lucas-Kanade Optical Flow
        lk_params = dict(
            winSize=(15, 15),
            maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
        )

        prev_frame = None

        # Iterate over each frame and corresponding fixations data
        for idx in range(len(mapped_gaze)):
            # Read the current frame from the video
            ret, frame = cap.read()
            if not ret:
                break

            curr_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            if idx >= 1:
                # Access previous frame fixations and filter by status
                prev_fixations = mapped_gaze.at[idx - 1, "fixations"]
                prev_fixations = prev_fixations[
                    (prev_fixations["status"] == "offset")
                    | (prev_fixations["status"] == "tracked")
                ]

                if not prev_fixations.empty:
                    # Prepare points for tracking
                    prev_pts = np.array(
                        prev_fixations[["x", "y"]].dropna().values, dtype=np.float32
                    ).reshape(-1, 1, 2)
                    prev_ids = prev_fixations["fixation_id"].values

                    # Calculate optical flow to find new positions of the points
                    curr_pts, status, err = cv2.calcOpticalFlowPyrLK(
                        prev_frame, curr_frame, prev_pts, None, **lk_params
                    )

                    # Update fixations for the current frame
                    curr_fixations = mapped_gaze.at[idx, "fixations"].copy()

                    # Append new or updated fixation points
                    for i, (pt, s, e) in enumerate(zip(curr_pts, status, err)):
                        if s[0]:  # Check if the point was successfully tracked
                            x, y = pt.ravel()
                            add_fixation = pd.DataFrame(
                                [
                                    {
                                        "fixation_id": prev_ids[i],
                                        "x": x,
                                        "y": y,
                                        "status": "tracked",
                                    }
                                ]
                            )
                            if not curr_fixations.empty and not add_fixation.empty:
                                curr_fixations = pd.concat(
                                    [curr_fixations, add_fixation], ignore_index=True
                                )
                        else:
                            # Handle cases where the point could not be tracked
                            add_fixation = pd.DataFrame(
                                [
                                    {
                                        "fixation_id": prev_ids[i],
                                        "x": None,
                                        "y": None,
                                        "status": "lost",
                                    }
                                ]
                            )
                            if not curr_fixations.empty and not add_fixation.empty:
                                curr_fixations = pd.concat(
                                    [curr_fixations, add_fixation], ignore_index=True
                                )

                    # Update the DataFrame with the modified fixations
                    mapped_gaze.at[idx, "fixations"] = curr_fixations

            # Update the previous frame for the next iteration
            prev_frame = curr_frame

        self.tracked_past_fixations = mapped_gaze

        # Release resources
        cap.release()
        cv2.destroyAllWindows()

        # Save the updated DataFrame to pickle
        mapped_gaze.to_pickle(updated_pkl)

    def overlay_fixations_on_video(
        self, output_path: Union[str, Path] = "data/fixations_overlayed_video.mp4"
    ):
        """
        Overlays fixation data on the video and saves the result to a new video file.

        Parameters
        ----------
        input_pkl : str or :class:pathlib.Path
            Path to the pickle file containing the updated fixation data.
        output_path : str or :class:pathlib.Path
            Path to save the video file with overlaid fixations.
        """
        # check if tracked_past_fixations is available
        if self.tracked_past_fixations is None:
            # run the track_fixations_with_optical_flow method to generate the tracked_past_fixations DataFrame
            self.track_fixations_with_optical_flow()

        df = self.tracked_past_fixations

        # Initialize video capture
        video_path = self.contents.loc["scene_video", "path"]
        cap = cv2.VideoCapture(str(video_path))

        # Set up video writer to save the overlayed video
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(
            str(output_path), fourcc, fps, (frame_width, frame_height)
        )

        # Iterate over each frame and corresponding fixations data
        for idx in df.index:
            # Read the current frame from the video
            ret, frame = cap.read()
            if not ret:
                break

            # Get the fixations for the current frame
            fixations = df.at[idx, "fixations"]

            # Draw fixations on the frame
            for _, fixation in fixations.iterrows():
                x, y = fixation["x"], fixation["y"]
                status = fixation["status"]
                if status == "tracked":
                    color = (0, 255, 0)
                elif status == "lost":
                    color = (0, 0, 255)
                else:
                    color = (255, 0, 0)

                # Only draw if coordinates are valid
                if pd.notna(x) and pd.notna(y):
                    # Draw a circle at the fixation point
                    cv2.circle(
                        frame, (int(x), int(y)), radius=10, color=color, thickness=-1
                    )

                    # Optionally, add text showing fixation status and ID
                    cv2.putText(
                        frame,
                        f"ID: {fixation['fixation_id']} Status: {status}",
                        (int(x) + 10, int(y)),
                        cv2.FONT_HERSHEY_PLAIN,
                        1,
                        color,
                        1,
                    )

            # Display the frame with overlays (Optional)
            cv2.imshow("Fixations Overlay", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

            # Write the frame to the output video
            out.write(frame)

        # Release resources
        cap.release()
        out.release()
        cv2.destroyAllWindows()

    def to_motion_bids(
        self,
        output_dir: Union[str, Path],
        prefix: str = "sub-XX_task-YY_tracksys-NeonIMU",
    ):
        """
        Export IMU data to Motion-BIDS format. Continuous samples are saved to a .tsv
        file and metadata (with template fields) are saved to a .json file.
        Users should later edit the metadata file according to the experiment to make
        it BIDS-compliant.

        Parameters
        ----------
        output_dir : str or :class:`pathlib.Path`
            Output directory to save the Motion-BIDS formatted data.
        prefix : str, optional
            Prefix for the BIDS filenames, by default "sub-XX_task-YY_tracksys-NeonIMU".
            The format should be `sub-<label>[_ses-<label>]_task-<label>_tracksys-<label>[_acq-<label>][_run-<index>]`
            (Fields in [] are optional). Files will be saved as
            ``{prefix}_motion.<tsv|json>``.

        Notes
        -----
        Motion-BIDS is an extension to the Brain Imaging Data Structure (BIDS) to
        standardize the organization of motion data for reproducible research [1]_.
        For more information, see
        https://bids-specification.readthedocs.io/en/stable/modality-specific-files/motion.html.

        References
        ----------
        .. [1] Jeung, S., Cockx, H., Appelhoff, S., Berg, T., Gramann, K., Grothkopp, S., ... & Welzel, J. (2024). Motion-BIDS: an extension to the brain imaging data structure to organize motion data for reproducible research. *Scientific Data*, 11(1), 716.
        """
        export_motion_bids(self, output_dir)

    def to_eye_bids(
        self,
        output_dir: Union[str, Path],
        prefix: str = "sub-XX_task-YY_tracksys-NeonGaze",
    ):
        """
        Export eye-tracking data to Eye-tracking-BIDS format. Continuous samples
        and events are saved to .tsv.gz files with accompanying .json metadata files.
        Users should later edit the metadata files according to the experiment.

        Parameters
        ----------

        output_dir : str or :class:`pathlib.Path`
            Output directory to save the Eye-tracking-BIDS formatted data.
        prefix : str, optional
            Prefix for the BIDS filenames, by default "sub-XX_recording-eye".
            The format should be `<matches>[_recording-<label>]_<physio|physioevents>.<tsv.gz|json>`
            (Fields in [] are optional). Files will be saved as
            ``{prefix}_physio.<tsv.gz|json>`` and ``{prefix}_physioevents.<tsv.gz|json>``.

        Notes
        -----
        Eye-tracking-BIDS is an extension to the Brain Imaging Data Structure (BIDS) to
        standardize the organization of eye-tracking data for reproducible research.
        The extension is still being finialized. This method follows the latest standards
        outlined in https://github.com/bids-standard/bids-specification/pull/1128.
        """
        exports_eye_bids(self, output_dir)
