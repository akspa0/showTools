# Product Context

## Problem
Automated audio annotation often lacks flexibility. Users need control over the separation and annotation processes to tailor results to specific needs and audio types. Managing recurring annotation tasks (like searching for specific sounds) can be repetitive. Processing large numbers of files individually is inefficient.

## User Experience (Phase 1)
- **Simple Workflow**: Upload a single audio file, select models/prompts, adjust key parameters (like confidence), and click Analyze.
- **Configurability**: Offer clear options for choosing audio separation models and CLAP prompts.
- **Presets**: Allow users to easily save, load, and manage custom sets of CLAP prompts via a dropdown and save mechanism, improving efficiency for repeated tasks.
- **Control**: Provide sliders/inputs for essential parameters like CLAP confidence threshold.
- **Clear Output**: Present results in a readable JSON format within automatically generated, uniquely named output folders. Allow easy downloading of the results.

## Future Enhancements (Phase 2)
- Extend the UI to accept folders as input for batch processing.
- Implement a job queue view allowing users to monitor and manage multiple analysis tasks (pause/cancel). 