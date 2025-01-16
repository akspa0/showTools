# showTools
pre and post-processing tools for audio. 

``whisperBite``
mixing/whisperBite/whisperBite.py - Using OpenAI's whisper Turbo model, transcribe audio into soundbites, quickly and easily. Supports --url option with yt-dlp as a backend, along with Demucs to split vocal audio.

mixing/miscAudioTools/merge_audio.py - Quick script to merge Vocal and Instrument tracks into a single stereo audio track.

mixing/miscAudioTools/convertToMono.py - Converts a file or folders with sub-directories to mono. 

mixing/miscAudioTools/5.1_tool.py - Convert from or to 5.1 surround FLAC files.

mixing/miscAudioTools/splitOrConcat.py - Splits or Concatenates audio files into 15 minute (default) segments.

---

``Asterisk/FreePBX stuff`` - 
freepbx/mix-stereo.sh: For Asterisk/FreePBX installs, to be setup as a Post call-recording script. Exports recorded call audio into Incoming and Outgoing audio, mixes to a single stereo file and a single mono output file. The stereo outputs OR each separate leg of the call (trans_out and recv_out files) are acceptable inputs into the preMix21.py script for futher processing.

``Setup your asterisk/freePBX to run a “Post call recording processing script”, clone the repo and copy the bash script (.sh) into ‘/var/asterisk/’, set the file as executable (chmod +x mix-stereo.sh), and point freePBX/Asterisk at the script. Now, when calls are made on your PBX, you will get four files per call – “out_”, “trans_out”, “recv_out”, and “stereo_out_”. This makes it easier to post-process and edit the calls to fix bad audio.``

freepbx/mix_stereo_archived.py - A tool designed to generate 'stereo_out-' files from 'trans_out-' and 'recv_out-' prefixed files.

