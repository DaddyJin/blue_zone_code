# All relative paths are relative to this extracted directory.
CONVERTER=./robocoin_converter_v4
LOADER=./robocoin_loader_tests_v4
# Set these to your installed Python interpreters (relative paths or command names).
# 转换与加载共用同一个 conda 环境：data-preprocess（Python 3.13 + LeRobot 0.5.0）
# 注意：/mnt/sfs_turbo/liujin/software/miniconda3 的 conda.sh 实际指向 /data5T/liujin/software/miniconda3
CONVERT_PY=/data5T/liujin/software/miniconda3/envs/data-preprocess/bin/python
LEROBOT_PY=/data5T/liujin/software/miniconda3/envs/data-preprocess/bin/python
INPUT=/mnt/sfs_turbo/zhangqinchuan/robocoint_process/samples/
OUTPUT=/mnt/sfs_turbo/liujin/dataset/robocoin-preprocess/data
WORK=/mnt/sfs_turbo/liujin/dataset/robocoin-preprocess/work
WORKERS=4
VIDEO_WORKERS=4
DECODE_THREADS=8
CPU_BUDGET=128
VIDEO_MB=128
