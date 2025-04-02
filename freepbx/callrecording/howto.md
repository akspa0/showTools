## Set Globals

Set these variables in globals_custom.conf:

SS=$
MIXMON_DIR=/var/spool/asterisk/monitor/
MONITOR_REC_OPTION = br(${SS}{MIXMON_DIR}${SS}{YEAR}/${SS}{MONTH}/${SS}{DAY}/recv_${SS}{CALLFILENAME}.${SS}{MON_FMT})t(${SS}{MIXMON_DIR}${SS}{YEAR}/${SS}{MONTH}/${SS}{DAY}/trans_${SS}{CALLFILENAME}.${SS}{MON_FMT})
MIXMON_POST = /var/spool/asterisk/mix-stereo.sh ${SS}{MIXMON_DIR}${SS}{YEAR}/${SS}{MONTH}/${SS}{DAY}/ ${SS}{CALLFILENAME}

create the shell script /var/spool/asterisk/mix-stereo.sh with the following:

#!/bin/bash

SOX="/usr/bin/sox -M"
RM="/bin/rm"

IN="${1}recv_$2.wav"
OUT="${1}trans_$2.wav"
DESTINATION="${1}stereo_$2.wav"

$SOX -M $IN $OUT $DESTINATION && $RM $IN $OUT

# CallRecording Patch for Stereo Recordings

Then patch /var/www/html/admin/modules/callrecording/functions.inc.php to place the ${EVAL( )} around ${MONITOR_REC_OPTION} and ${MONMIX_POST} so that the embedded variable in the global variables are evaluated when used. Adding the EVAL function has no negative impact when not being used so hopefully Sangoma will include this mod is future releases.

orig:      $ext->add($context, $exten, 'monitorcmd', new ext_mixmonitor('${MIXMON_DIR}${YEAR}/${MONTH}/${DAY}/${CALLFILENAME}.${MON_FMT}','a${MONITOR_REC_OPTION}i(${LOCAL_MIXMON_ID})${MIXMON_BEEP}','${MIXMON_POST}'));

patched:     $ext->add($context, $exten, 'monitorcmd', new ext_mixmonitor('${MIXMON_DIR}${YEAR}/${MONTH}/${DAY}/${CALLFILENAME}.${MON_FMT}','a${EVAL(${MONITOR_REC_OPTION})}i(LOCAL_MIXMON_ID)${MIXMON_BEEP}','${EVAL(${MIXMON_POST})}'));
