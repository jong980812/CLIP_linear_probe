import os
import subprocess
import glob
import shutil
from tqdm import tqdm

def extract_last_frame_and_cleanup(video_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    fname = os.path.basename(video_path)
    frame_dir = os.path.join(output_dir, fname.replace(".mp4", "_frames"))
    os.makedirs(frame_dir, exist_ok=True)

    # 1) 모든 프레임 추출
    cmd = [
        "ffmpeg",
        "-i", video_path,
        "-q:v", "1",
        os.path.join(frame_dir, "frame_%06d.jpg"),
        "-loglevel", "error",
        "-y"
    ]
    subprocess.run(cmd, check=True)

    # 2) 마지막 프레임 찾기
    frames = sorted(glob.glob(os.path.join(frame_dir, "frame_*.jpg")))
    if len(frames) == 0:
        print(f"[SKIP] No frames: {video_path}")
        return

    last_frame = frames[-1]

    # 3) 최종 저장 경로
    out_path = os.path.join(output_dir, fname.replace(".mp4", ".jpg"))
    shutil.move(last_frame, out_path)

    # 4) 나머지 프레임 삭제
    for f in frames[:-1]:
        os.remove(f)

    # 5) 빈 폴더 삭제
    os.rmdir(frame_dir)


def process_all_mp4(input_dir, output_dir):
    mp4_files = sorted(glob.glob(os.path.join(input_dir, "*.mp4")))
    print(f"Found {len(mp4_files)} videos")

    for video in tqdm(mp4_files):
        try:
            extract_last_frame_and_cleanup(video, output_dir)
        except Exception as e:
            print(f"[ERROR] {video}: {e}")


import argparse
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--jpg_q", type=int, default=1,
                        help="ffmpeg jpg quality: 1(best) ~ 31(worst). default=1")
    args = parser.parse_args()

    process_all_mp4(args.input_dir, args.output_dir, jpg_q=args.jpg_q)


if __name__ == "__main__":
    main()