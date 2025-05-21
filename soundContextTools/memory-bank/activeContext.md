# activeContext.md

**Purpose:**
Tracks current work focus, recent changes, next steps, and active decisions/considerations.

## Current Focus

Implementing a robust pipeline orchestrator to:
- Centralize control of all pipeline stages for each run
- Track global and per-file progress (tqdm-ready)
- Manage jobs (input files/tuples) with state and metadata
- Handle errors, retries, and logging in a unified way
- Prepare for future parallelism, job queuing, and UI/CLI progress bars

## Recent Changes

- Decided to introduce a PipelineOrchestrator class/module
- Orchestrator will manage jobs, stages, progress, and logging
- File input stage will be integrated as the first orchestrated step

## Next Steps

- Design and document the Job and PipelineOrchestrator abstractions
- Refactor file input logic to produce jobs for the orchestrator
- Stub out additional pipeline stages as orchestrator methods
- Implement global and per-file progress tracking (tqdm)
- Ensure all logging and error handling is routed through the orchestrator

## Active Decisions & Considerations

- All pipeline stages will be orchestrated for safety, extensibility, and debuggability
- Orchestrator will make it easy to add parallelism, job queuing, and UI features in the future
- Progress bars and job state tracking are required for large batch processing and debugging 