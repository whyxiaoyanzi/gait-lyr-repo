# run $1 shell script in Singularity container
singularity exec --bind=/data /data/openpose/opp20.sif /bin/bash $1