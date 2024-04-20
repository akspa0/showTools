# showTools
post-processing scripts for phone call audio

preMix.py - A premixing script for stereo recordings. Uses below mix-stereo script for Asterisk/FreePBX for input data. In a constant state of breaking changes that are minimally tested - https://github.com/akspa0/showTools/blob/main/README_preMix.md

transcribeBites12.py - highly experimental version of the older transcribeBites0F.py script below.

transcribeBites0F.py - successor to speakerTranscripe29I. Development version that is more stable than 29I.
(https://github.com/akspa0/showTools/blob/main/README_transcribeBites.md)

convertToMono.py - Converts a file or folders with sub-directories to mono. 
(https://github.com/akspa0/showTools/blob/main/README_convertToMono.md)

``DEPRECATED``
preMix21.py - A tool that takes audio captured from Asterisk/FreePBX recordings, and pre-mixes them with stereo separation between outgoing and incoming audio 
(https://github.com/akspa0/showTools/blob/main/README_preMix21.md)

``DEPRECATED``
speakerTranscribe29I.py - A proof of concept that segments audio and transcribes the segments, however, it has several issues with detecting speech properly and has been deprecated in favor of 'TranscribeBites'.

---

``Asterisk/FreePBX stuff`` - 
mix-stereo.sh: For Asterisk/FreePBX installs, to be setup as a Post call-recording script. Exports recorded call audio into Incoming and Outgoing audio, mixes to a single stereo file and a single mono output file. The stereo outputs OR each separate leg of the call (trans_out and recv_out files) are acceptable inputs into the preMix21.py script for futher processing.

``Setup your asterisk/freePBX to run a “Post call recording processing script”, clone the repo and copy the bash script (.sh) into ‘/var/asterisk/’, set the file as executable (chmod +x mix-stereo.sh), and point freePBX/Asterisk at the script. Now, when calls are made on your PBX, you will get four files per call – “out_”, “trans_out”, “recv_out”, and “stereo_out_”. This makes it easier to post-process and edit the calls to fix bad audio.``



