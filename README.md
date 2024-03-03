# showTools
post-processing scripts for phone call audio

preMix21.py:  Processes stereo audio files, splitting them into two mono streams, normalizing and compressing the audio, and then merging the streams into a single output file. It supports both stereo input files and pre-split left and right channels. Additionally, it provides an option to append an audio segment to the end of each processed file. The script utilizes the PyDub library for audio manipulation and supports various audio formats such as WAV.


mix-stereo.sh: exports recorded call audio into multiple legs and a stereo file, which is utilized by preMix21.py for futher processing.


speakerTranscribe29I.py: Takes audio files as an input (directory or single file [which might not work]) and processes them with voice detection to find segments of audible speech, passes the segments into Google Speech to Text API, and fetches a transcription for the segment, then outputs the segment into FLAC and mp3 format audio files with the transcription as the filename. Duplicate segments are filtered out. Processes phrases a little bit with some normalization to ensure audible vocal samples.


transcribeBites0F.py: A new in-development version of speakerTranscribe29I.py, with a few minor changes but planned to have many other changes to track progress better. 
