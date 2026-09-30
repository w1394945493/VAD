


PYTHONPATH="$(pwd):${PYTHONPATH:-}" \
python /vepfs-mlp2/c20250502/haoce/wangyushen/VAD/tools/test.py \
    /vepfs-mlp2/c20250502/haoce/wangyushen/VAD/projects/configs/VAD/VAD_tiny_stage_2_custom.py \
    /c20250502/wangyushen/Weights/vad/VAD_tiny.pth \
    --eval bbox


python tools/analysis_tools/visualization.py \
    --result-path /path/to/inference/results \
    --save-path /path/to/save/visualization/results

python /home/wys/wsl/forks/VAD/scripts/create_scene_videos.py \
    --input-dir 