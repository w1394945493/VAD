#!/usr/bin/env python3
"""将按 scene token 保存的 VAD 可视化帧批量合成为 MP4 视频。"""

import argparse
import shutil
import subprocess
from pathlib import Path


DEFAULT_INPUT_DIR = Path(
    "/home/wys/wsl/forks/Outputs/outputs/vad_base_vis")


def parse_args():
    parser = argparse.ArgumentParser(
        description="将 <scene_token>/frame_XXX.png 合成为逐场景 MP4。")
    parser.add_argument(
        "--input-dir", type=Path, default=DEFAULT_INPUT_DIR,
        help="包含 scene token 子目录的可视化根目录。")
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help="视频输出目录；默认是 <input-dir>/videos。")
    parser.add_argument(
        "--fps", type=float, default=2.0,
        help="输出视频帧率；nuScenes 关键帧默认按 2 FPS 播放。")
    parser.add_argument(
        "--overwrite", action="store_true",
        help="覆盖已经存在的同名 MP4。")
    return parser.parse_args()


def frame_number(path):
    """从 frame_XXX.png 中提取整数帧号，用于检查帧序列。"""
    try:
        return int(path.stem.rsplit("_", 1)[1])
    except (IndexError, ValueError) as error:
        raise ValueError(f"无法解析帧号：{path.name}") from error


def create_video(scene_dir, output_path, fps, overwrite):
    frames = sorted(scene_dir.glob("frame_*.png"), key=frame_number)
    if not frames:
        return False

    frame_numbers = [frame_number(path) for path in frames]
    expected = list(range(frame_numbers[0], frame_numbers[-1] + 1))
    if frame_numbers != expected:
        missing = sorted(set(expected) - set(frame_numbers))
        print(f"[跳过] {scene_dir.name}: 缺少帧 {missing}")
        return False

    if output_path.exists() and not overwrite:
        print(f"[跳过] 已存在：{output_path}")
        return False

    command = [
        "ffmpeg", "-hide_banner", "-loglevel", "error",
        "-y" if overwrite else "-n",
        "-framerate", str(fps),
        "-start_number", str(frame_numbers[0]),
        "-i", str(scene_dir / "frame_%03d.png"),
        "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
        "-c:v", "libx264", "-preset", "medium", "-crf", "20",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart",
        str(output_path),
    ]
    subprocess.run(command, check=True)
    print(f"[完成] {scene_dir.name}: {len(frames)} 帧 -> {output_path}")
    return True


def main():
    args = parse_args()
    input_dir = args.input_dir.resolve()
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None else input_dir / "videos")

    if not input_dir.is_dir():
        raise SystemExit(f"输入目录不存在：{input_dir}")
    if shutil.which("ffmpeg") is None:
        raise SystemExit("找不到 ffmpeg，请先安装并加入 PATH。")
    if args.fps <= 0:
        raise SystemExit("--fps 必须大于 0。")

    output_dir.mkdir(parents=True, exist_ok=True)
    scene_dirs = sorted(
        path for path in input_dir.iterdir()
        if path.is_dir() and path != output_dir
    )

    completed = 0
    for scene_dir in scene_dirs:
        output_path = output_dir / f"{scene_dir.name}.mp4"
        completed += create_video(
            scene_dir, output_path, args.fps, args.overwrite)

    print(f"处理结束：发现 {len(scene_dirs)} 个场景，生成 {completed} 个视频。")


if __name__ == "__main__":
    main()
