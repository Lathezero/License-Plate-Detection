import numpy as np
import os
import gradio as gr
import uvicorn
from fastapi import FastAPI
from tqdm import tqdm
import cv2
import numpy
from shutil import copy
from kenshutsu import Kenshutsu # 导入检测类
from read_plate import ReadPlate
from datetime import datetime
from PIL import Image, ImageDraw, ImageFont
from db_operations import create_database, create_table, insert_plate, get_all_plates

# from logger_module import Logger


# logger = Logger('/www/wwwpython/logClass/lpd_log.txt')


def DrawChinese(img, text, positive, fontSize=20, fontColor=(
        255, 0, 0)):  # args-(img:numpy.ndarray, text:中文文本, positive:位置, fontSize:字体大小默认20, fontColor:字体颜色默认绿色)
    cv2img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # cv2和PIL中颜色的hex码的储存顺序不同
    pilimg = Image.fromarray(cv2img)
    # PIL图片上打印汉字
    draw = ImageDraw.Draw(pilimg)  # 图片上打印
    font = ImageFont.truetype("MSJHL.TTC", fontSize, encoding="utf-8")  # 参数1：字体文件路径，参数2：字体大小
    draw.text(positive, text, fontColor, font=font)  # 参数1：打印坐标，参数2：文本，参数3：字体颜色，参数4：字体格式
    # print(text)
    cv2charimg = cv2.cvtColor(numpy.array(pilimg), cv2.COLOR_RGB2BGR)  # PIL图片转cv2 图片
    return cv2charimg, text

def process_image(input_image):
    gr.Info("调用模型开始推理……")
    detecter = Kenshutsu(False)
    read_plate = ReadPlate()
    boxes = detecter(input_image)
    plates = []
    detected_plates = []  # Store detected plates for later use
    for box in boxes:
        x1, y1, x2, y2, the, c = box
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        if int(c) >= 2 and int(c) <= 8:
            image_ = input_image[y1:y2, x1:x2]
            result = read_plate(image_)
            class_name = detecter.names[int(c)]
            if result:
                plate, (x11, y11, x22, y22) = result[0]
                plates.append((x1, y1, x2, y2, plate, x11 + x1, y11 + y1, x22 + x1, y22 + y1))
                detected_plates.append(plate)  # Store the detected plate

    result_image = input_image.copy()
    car_numbers = []
    new_car_number = ""
    result_str = ""
    index = 1
    for plate in plates:
        x1, y1, x2, y2, plate_name, x11, y11, x22, y22 = plate
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        x11, y11, x22, y22 = int(x11), int(y11), int(x22), int(y22)
        result_image = cv2.rectangle(result_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        result_image = cv2.rectangle(result_image, (x11 - 5, y11 - 5), (x22 + 5, y22 + 5), (0, 0, 255), 2)
        result_image, new_car_number = DrawChinese(result_image, plate_name, (x11, y22), 30)
        car_numbers.append(new_car_number)
        result_str += f"车辆{index}--> 车型：{class_name}  车牌号 : {new_car_number}\n"
        index += 1

    gr.Info("Success! 推理完成……")
    # 返回检测到的车牌号列表
    return result_image, result_str.rstrip("\n"), gr.Dropdown(choices=car_numbers)

def add_owner_name(plate_number, owner_name):
    if plate_number and owner_name:
        insert_plate(plate_number, owner_name)
        return get_all_plates()
    return get_all_plates()

def clear_inputs():
    return None, None, "", None, None

def clear_images_in_directory(directory): # 清空指定目录下的所有图片文件。
    for root, dirs, files in os.walk(directory):
        for file in files:
            file_extension = os.path.splitext(file)[1].lower()
            if file_extension in ['.jpg', '.jpeg', '.png', '.bmp', '.gif']:
                file_path = os.path.join(root, file)
                os.remove(file_path)



# 图片合成为视频
def create_video(input_path, output_video_path, fps, progress=gr.Progress()):

    frame_paths = sorted([os.path.join(input_path, f) for f in os.listdir(input_path) if f.endswith('.jpg')])
    img = cv2.imread(frame_paths[0])
    height, width, _ = img.shape

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    # 使用tqdm显示进度条
    cnt = 0
    with tqdm(total=len(frame_paths)) as pbar:
        for frame_path in frame_paths:
            # 读取每帧图片
            img = cv2.imread(frame_path)
            out.write(img)
            progress(cnt / len(frame_paths), desc="step3:视频合成ing……")
            pbar.update(1)  # 更新进度条
            cnt = cnt + 1

    out.release()

# 视频拆分为图片
def extract_frames(video_path, output_dir, desired_fps, progress=gr.Progress()):
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
    cnt = 0;
    progress_bar = tqdm(total=total_frames, desc='拆分视频帧', unit='frames')

    while success:
        if count % interval == 0:
            frame_path = os.path.join(output_dir, f"frame{count}.jpg")
            cv2.imwrite(frame_path, image)  # 保存帧为图片

            progress(cnt / total_frames, desc="step1:预处理视频帧ing……")
            progress_bar.update(1)  # 更新进度条
            cnt = cnt + 1

        success, image = vidcap.read()
        count += 1

    progress_bar.close()  # 关闭进度条

    # logger.write_log(f'视频帧拆分完成:')


def process_video(input_video, progress=gr.Progress()):
    # logger.write_log(f'\n视频检测开始:')
    gr.Info("视频模型推理开始……")
    # 获取当前日期和时间，优化时间格式获取写法，更简洁
    time_str = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    # 指定保存视频的目录
    save_dir = "user_upload_videos"
    os.makedirs(save_dir, exist_ok=True)  # 确保目录存在，简化创建目录逻辑

    try:
        with open(input_video, "rb") as video_file:
            video_content = video_file.read()
    except FileNotFoundError:
        # logger.write_log(f"输入视频文件 {input_video} 不存在，无法继续处理。")
        return None

    video_path = os.path.join(save_dir, f'{time_str}.mp4')  # 用户上传的视频保存路径，使用os.path.join更规范

    try:
        with open(video_path, "wb") as output_file:
            output_file.write(video_content)
    except IOError as e:
        # logger.write_log(f"保存视频文件 {video_path} 时出错: {str(e)}")
        return None

    # logger.write_log(f'开始将上传的视频切割为图片.')
    gr.Info("视频预处理……")
    clear_images_in_directory('./videoToImg')
    extract_frames(video_path, "./videoToImg", 60)  # 将视频转换为每秒30帧的图片
    gr.Info("预处理完成……")
    # logger.write_log(f'图片切割完成.')

    root = 'videoToImg'  # 检测视屏帧用这个
    detecter = Kenshutsu(is_cuda=True)
    read_plate = ReadPlate()

    output_dir = 'imgToVideo'  # 新建保存图像的目录
    os.makedirs(output_dir, exist_ok=True)  # 确保目录存在
    clear_images_in_directory('./imgToVideo')  # 清空

    img_size = len(os.listdir(root))
    cnt = 0
    car_number = ""
    gr.Info("开始视频推理，视频推理时间较长，请耐心等待……")

    for image_name in os.listdir(root):
        image_path = os.path.join(root, image_name)
        image = cv2.imread(image_path)
        boxes = detecter(image)
        plates = []
        for box in boxes:
            x1, y1, x2, y2, the, c = box
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            if c == 2 or c == 5:
                image_ = image[y1:y2, x1:x2]
                result = read_plate(image_)
                if result:
                    plate, (x11, y11, x22, y22) = result[0]
                    plates.append((x1, y1, x2, y2, plate, x11 + x1, y11 + y1, x22 + x1, y22 + y1))
        for plate in plates:
            x1, y1, x2, y2, plate_name, x11, y11, x22, y22 = plate
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            x11, y11, x22, y22 = int(x11), int(y11), int(x22), int(y22)
            image = cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            image = cv2.rectangle(image, (x11 - 5, y11 - 5), (x22 + 5, y22 + 5), (0, 0, 255), 2)
            image, car_number = DrawChinese(image, plate_name, (x11, y22), 30)

        output_path = os.path.join(output_dir, image_name)
        cv2.imwrite(output_path, image)

        progress(cnt / img_size, desc="step2:模型推理阶段ing……")
        cnt = cnt + 1

    gr.Info("推理完成，开始合成……")
    # logger.write_log(f'开始合成视频.')
    try:
        create_video('./imgToVideo', os.path.join('./result_videos', f'{time_str}.mp4'), 60)  # 将图片合成为每秒60帧的视频，规范路径拼接
    except Exception as e:
        # logger.write_log(f"合成视频时出错: {str(e)}")
        raise gr.Warning(f"合成视频时出错: {str(e)}")
        return None
    # logger.write_log(f'合成完毕.')
    gr.Info("Success! 合成完毕……")
    output_video = gr.Video(value=f'./result_videos/{time_str}.mp4')
    return output_video


def clear_video_inputs():
    return None, None

# 自定义CSS样式
custom_css = """
.start-button {
    color: blue;
    margin: 4px 2px;
}

.clear-button {
    color: red;
    margin: 4px 2px;
}
"""

title_text = """
<center><h1>基于深度学习的高效车型车牌检测与识别算法</h1></center>
<center><h5>PS: 页面分类两个模块，分别用来检测图片和视频，请根据需要点击下方导航栏选择</h5></center>
"""
image_tips = """
<p><b>PS: 下方Examples中提供了若干可供测试的图片，点击具体图片即可直接导入上方图片输入框</b></p>
<p><b>PS: 检测结束后右侧下方会显示检测车牌号的结果</b></p>
"""

video_tips = """
<p><b>PS2: 点击下方"开始检测"按钮后右侧视频返回框会有进度条显示当前进度，进度条共有三个阶段，请耐心等待</b></p>
"""

with gr.Blocks(css=custom_css, title='基于深度学习Yolo与生成式网络的车牌检测算法设计与实现') as demo:
    gr.Markdown(title_text)
    with gr.Tabs():  # 使用gr.Tabs来创建选项卡容器
        with gr.TabItem("图像中检测车型及车牌"):  # 每个具体的选项卡页面使用gr.TabItem定义
            gr.Markdown(image_tips)
            with gr.Row():
                with gr.Column():
                    input_image = gr.Image(label="待检测图像")
                    with gr.Row():
                        clear_button = gr.Button("清空图片输入", elem_classes="clear-button")
                        submit_button = gr.Button("开始检测", elem_classes="start-button")
                with gr.Column():
                    result = gr.Image(label="检测结果")
                    output_logs = gr.Textbox(label="车型及车牌号", lines=3)

            # 添加样例输入
            examples = [
                 ["./test_image/1.jpg"],
                 ["./test_image/2.jpg"],
                 ["./test_image/3.jpg"],
                 ["./test_image/4.png"],
                 ["./test_image/5.jpg"],
                 ["./test_image/6.jpg"],
                 ["./test_image/7.png"],
                 ["./test_image/8.jpg"],
                 ["./test_image/9.jpg"],
                 ["./test_image/10.jpg"],
                 ["./test_image/11.jpg"],
                 ["./test_image/12.png"],
                 ["./test_image/13.jpeg"],
                 ["./test_image/14.jpg"]
            ]

            gr.Examples(
                examples=examples,
                inputs=[input_image],
            )

            # Add post-detection name input section
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### 添加车主信息")
                    plate_dropdown = gr.Dropdown(label="选择车牌号", choices=[], interactive=True)
                    owner_name = gr.Textbox(label="车主姓名", placeholder="请输入车主姓名")
                    add_name_button = gr.Button("添加车主信息", elem_classes="start-button")

            # Add database records display
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### 已记录的车牌信息")
                    db_records = gr.Dataframe(
                        headers=["车牌号", "车主姓名"],
                        datatype=["str", "str"],
                        col_count=(2, "fixed"),
                        row_count=(10, "dynamic"),
                        value=get_all_plates()
                    )

            # Add refresh button for database records
            refresh_button = gr.Button("刷新数据库记录")
            refresh_button.click(
                fn=lambda: get_all_plates(),
                inputs=[],
                outputs=[db_records]
            )

            submit_button.click(
                process_image,
                inputs=[input_image],
                outputs=[result, output_logs, plate_dropdown]
            )

            add_name_button.click(
                add_owner_name,
                inputs=[plate_dropdown, owner_name],
                outputs=[db_records]
            )

            clear_button.click(
                clear_inputs,
                inputs=[],
                outputs=[input_image, result, output_logs, plate_dropdown, owner_name]
            )


        with gr.TabItem("视频中检测车牌"):
            gr.Markdown(video_tips)
            with gr.Row():
                with gr.Column():
                    input_video = gr.Video(label="输入视频")
                    with gr.Row():
                        clear_video_button = gr.Button("清空视频输入", elem_classes="clear-button")
                        submit_video_button = gr.Button("开始检测", elem_classes="start-button")

                with gr.Column():
                    output_video = gr.Video(label="输出视频")


                submit_video_button.click(
                    process_video,
                    inputs=[input_video],
                    outputs=[output_video]
                )
                clear_video_button.click(
                    clear_video_inputs,
                    inputs=[],
                    outputs=[input_video, output_video]
                )
    # gr.Markdown(copyright_text)


app = FastAPI()
app = gr.mount_gradio_app(app, demo, path="/")

if __name__ == '__main__':
    uvicorn.run(host='127.0.0.1',app='main:app', port=9096, reload=False)