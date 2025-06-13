### 使用方法
- 创建和激活环境（使用conda）
    ```
    conda create -y -n  kenshutsu python=3.8
    conda activate kenshutsu
    pip install -r requirements.txt -i  https://pypi.tuna.tsinghua.edu.cn/simple
    ```
- 将待检测的视频放在根目录下，然后修改videoToImage.py文件夹下的output_path路径为你视频的路径
- 在控制台执行python videoToImg.py 将视频逐帧裁剪为图片
- 在控制台执行python kenshutsu.py 将裁剪出来的图片检测后存入指定文件夹
- 在控制台执行python imgToVideo.pt 将检测完的图片重新整合为视频

- 注意，上述描述的是通过kenshutsu.py文件对视频进行检测的过程，如果你需要检测的是图片，那么需要将kenshutsu.py文件中第109行root = 'videoToImg'代码打开，将下面一行去掉，另外将145、146行也都解开，检测结束的图片保存在imgToVideo文件夹下

### 更推荐使用方法2，直接启动web端
- 激活conda环境后，直接在终端执行python main.py打开弹出的地址即可