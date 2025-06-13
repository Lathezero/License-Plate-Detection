import os
import cv2
from tqdm import tqdm


def extract_frames(video_path, output_dir, desired_fps):
    # 判断输出目录是否存在，如果不存在则创建
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
        print(f"Created directory: {output_dir}")

    vidcap = cv2.VideoCapture(video_path)
    success, image = vidcap.read()
    count = 0
    fps = vidcap.get(cv2.CAP_PROP_FPS)  # 获取视频的原始帧率
    interval = int(round(fps / desired_fps))  # 计算每隔多少帧提取一次
    if interval < 1:
        interval = 1

    total_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))

    # 使用tqdm创建一个进度条
    progress_bar = tqdm(total=total_frames, desc='Extracting frames', unit='frames')

    while success:
        if count % interval == 0:
            frame_path = os.path.join(output_dir, f"frame{count}.jpg")
            cv2.imwrite(frame_path, image)  # 保存帧为图片

            progress_bar.update(1)  # 更新进度条

        success, image = vidcap.read()
        count += 1

    progress_bar.close()  # 关闭进度条

    print("\n完成！")


# ---------------------------------------------------------
fps = 30  # 你拍摄视频的fps ps：如果这个值大于了你视频实际的帧率，则会以1帧为单位提取
# ---------------------------------------------------------
# 定义输入视频路径和帧输出路径
video_path = "./sss.mp4"
output_path = "./videoToImg"

# 调用函数进行视频帧提取
extract_frames(video_path, output_path, fps)
