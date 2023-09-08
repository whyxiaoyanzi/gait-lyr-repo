#!/usr/bin/bash
# Usage    : For web app to delete 1 assessment file folder
# Parameter: 1) File name (e.g ARGait_P_H019_Free walk_09-12-2022_15-24-52)

WEB_ROOT=/data/pdlogger
FOLDER="${WEB_ROOT}/storage/app/public/${1}"

# check if there is input file, exit if none
if [ ! -d "$FOLDER" ]; then
 echo ERROR: $FOLDER folder not found
 exit 1
fi

echo rm -rf "$FOLDER"
rm -rf "$FOLDER"
exit 0
