# showTools
post-processing scripts for phone call audio

preMix21.py:  Processes stereo audio files, splitting them into two mono streams, normalizing and compressing the audio, and then merging the streams into a single output file. It supports both stereo input files and pre-split left and right channels. Additionally, it provides an option to append an audio segment to the end of each processed file. The script utilizes the PyDub library for audio manipulation and supports various audio formats such as WAV.


mix-stereo.sh: exports recorded call audio into multiple legs and a stereo file, which is utilized by preMix21.py for futher processing.
