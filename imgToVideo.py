import cv2
import os
from tqdm import tqdm


def create_video(input_path, output_video_path, fps):

    frame_paths = sorted([os.path.join(input_path, f) for f in os.listdir(input_path) if f.endswith('.jpg')])

    img = cv2.imread(frame_paths[0])
    height, width, _ = img.shape

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    # 使用tqdm显示进度条
    with tqdm(total=len(frame_paths)) as pbar:
        for frame_path in frame_paths:
            # 读取每帧图片
            img = cv2.imread(frame_path)
            out.write(img)

            pbar.update(1)  # 更新进度条

    out.release()


# ---------------------------------------------------------
fps = 30  # 你拍摄视频的fps
# ---------------------------------------------------------

# 定义帧图片路径、输出视频路径和帧率
input_path = "./imgToVideo"
output_video_path = "./outputVideo.mp4"  # 视频保存路径

# 调用函数将帧图片重新组合为视频
create_video(input_path, output_video_path, fps)
# ---------------------------------------------------------
