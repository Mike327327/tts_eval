#!/bin/bash
#
#$ -S /bin/bash
#$ -N run_utmos
#$ -l gpu=1,gpu_ram=16G,ram_free=100G
#$ -pe smp 2
#$ -q long.q@@speech-gpu
#

ulimit -f unlimited
ulimit -t unlimited

N_GPUS=1
# Set the CUDA_VISIBLE_DEVICES variable
# If N_GPUS is set, export devices
if [ -n "$N_GPUS" ]; then
  export $(/mnt/matylda4/kesiraju/bin/gpus $N_GPUS) || exit 1
  echo "Visible devices: ${CUDA_VISIBLE_DEVICES}"
else
  export CUDA_VISIBLE_DEVICES=""
fi

echo "Activating Conda env"
source /mnt/matylda4/xluner01/miniconda3/bin/activate /mnt/matylda4/xluner01/miniconda3/envs/UTMOSv2

cd /mnt/matylda4/xluner01/tts_eval || exit 1
echo "$PWD"

chmod 755 ./*.py

echo "Running the utmos_default script"
python utmos_default.py
