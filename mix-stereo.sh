#!/bin/bash

SOX="/usr/bin/sox -M"
RM="/bin/rm"

IN="${1}recv_$2.wav"
OUT="${1}trans_$2.wav"
DESTINATION="${1}stereo_$2.wav"

$SOX -M $IN $OUT $DESTINATION
