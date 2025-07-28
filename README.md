![GitHub License](https://img.shields.io/github/license/ncc-brain/PyNeon?style=plastic)
![Website](https://img.shields.io/website?url=https%3A%2F%2Fncc-brain.github.io%2FPyNeon%2F&up_message=online&style=plastic&label=Documentation)

# PyNeon

PyNeon is a lightweight Python package designed to streamline the processing
and analysis of multimodal eye-tracking data from the
[Neon eye-tracking system](https://pupil-labs.com/products/neon)
(Pupil Labs GmbH). This community-driven effort provides a versatile set of
tools to work with Neon's rich data, including gaze, eye states, IMU, video,
events, and more.

Currently, PyNeon supports the `Timeseries Data` or `Timeseries Data + Scene Video` formats of data, downloaded from [Pupil Cloud](https://cloud.pupil-labs.com/). For reading data in the `native` format, please refer to the
[`pl-neon-recording`](https://github.com/pupil-labs/pl-neon-recording/) project, which inspired the design of PyNeon.

Documentation for PyNeon is available at <https://ncc-brain.github.io/PyNeon/> which includes detailed references for classes and functions, as well as step-by-step tutorials presented as Jupyter notebooks.

## Key Features

- [(Tutorial)](https://ncc-brain.github.io/PyNeon/tutorials/read_recording.html) Easy API for reading in datasets and recordings. Quick access to various modalities of data.
- [(Tutorial)](https://ncc-brain.github.io/PyNeon/tutorials/interpolate_and_concat.html) Various preprocessing functions, including data cropping, interpolation,
  concatenation, etc.
- [(Tutorial)](https://ncc-brain.github.io/PyNeon/tutorials/pupil_size_and_epoching.html) Flexible epoching of data for trial-based analysis.
- [(Tutorial)](https://ncc-brain.github.io/PyNeon/tutorials/video.html) Methods for working with scene video, including scanpath estimation and AprilTags-based mapping.
- [(Tutorial)](https://ncc-brain.github.io/PyNeon/tutorials/export_to_bids.html) Exportation to [Motion-BIDS](https://www.nature.com/articles/s41597-024-03559-8) (and forthcoming Eye-Tracking-BIDS) format for interoperability across the cognitive neuroscience community.

## Installation

To install PyNeon, clone the PyNeon repository from
<https://github.com/ncc-brain/PyNeon> and run:

```bash
pip install .
```

PyPI and conda releases are planned for the future.

## Citing PyNeon

If you use PyNeon in your research, please cite the
[accompanying paper](https://osf.io/preprints/psyarxiv/y5jmg)
as follows:

```bibtex
@misc{pyneon,
    title={PyNeon: a Python package for the analysis of Neon multimodal mobile eye-tracking data},
    url={osf.io/preprints/psyarxiv/y5jmg_v1},
    DOI={10.31234/osf.io/y5jmg_v1},
    publisher={PsyArXiv},
    author={Chu, Qian and Hartel, Jan-Gabriel and Lepauvre, Alex and Melloni, Lucia},
    year={2025},
    month={Jun}
}
```