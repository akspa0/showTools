# showTools
post-processing scripts for phone call audio

preMix21.py - (akspa0/showTools/blob/main/README_preMix21.md)

mix-stereo.sh: For Asterisk/FreePBX installs, to be setup as a Post call-recording script. Exports recorded call audio into Incoming and Outgoing audio, mixes to a single stereo file and a single mono output file. The stereo outputs OR each separate leg of the call (trans_out and recv_out files) are acceptable inputs into the preMix21.py script for futher processing. See https://immoralhole.com/show/asterisk-ramblings/ for more info.


speakerTranscribe29I.py - (https://github.com/akspa0/showTools/blob/main/README_transcribeBites.md)


transcribeBites0F.py - successor to speakerTranscripe29I. Development version that is more stable than 29I.
(akspa0/showTools/blob/main/README_transcribeBites.md)

convertToMono.py - Converts a file or folders with sub-directories to mono. (akspa0/showTools/blob/main/README_convertToMono.md)
