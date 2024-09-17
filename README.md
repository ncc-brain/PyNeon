# PyNeon
PyNeon is a light-weight library to work with Neon (Pupil Labs) multi-modal eye-tracking data.

## Documentation
[PyNeon Documentation](https://ncc-brain.github.io/PyNeon/) contains public API and tutorials as Jupyter notebooks.

## Diagram of classes

```plaintext
Timeseries Data                 >> NeonDataset('Timeseries Data')
├── Recording1-e116e606         >> NeonRecording('Timeseries Data/Recording1-e116e606')
│   ├── info.json                  ├── .info (dict)
|   ├── gaze.csv                   ├── .gaze (NeonGaze)             
|   ├── 3d_eye_states.csv          ├── .eye_states (NeonEyeStates)
|   ├── imu.csv                    ├── .imu (NeonIMU)
|   ├── blinks.csv                 ├── .blinks (NeonBlinks)
|   ├── fixations.csv              ├── .fixations (NeonFixations)
|   ├── saccades.csv               ├── .saccades (NeonSaccades)
|   ├── events.csv                 ├── .events (NeonEvents)
|   ├── scene_camera.json
|   ├── world_timestamps.csv
|   ├── <scene_video>.mp4
|   └── labels.csv
├── Recording2-93b8c234
│   ├── info.json
│   ├── gaze.csv
│   └── ....
├── enrichment_info.txt
└── sections.csv
```
