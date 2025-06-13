# LPD (License Plate Detection)

LPD 是一个基于深度学习的智能车牌检测与识别系统。它能够自动识别图片和视频中的车牌信息，支持实时处理和批量处理，并提供直观的网页界面。系统采用先进的计算机视觉技术，具有高准确率和快速响应的特点。系统集成了数据库功能，可以存储和管理车牌信息，支持车牌号与车主信息的关联查询。

## 功能特点

- 车牌检测与识别
- 支持图片和视频处理
- 提供便捷的网页交互界面
- 支持命令行批量处理
- 数据库集成存储结果
- 实时处理能力
- 车牌信息管理
  - 车牌号与车主信息关联
  - 历史记录查询
  - 数据统计分析

## 环境要求

- Python 3.8
- Conda（用于环境管理）
- CUDA 兼容的 GPU（推荐使用，可提升性能）
- MySQL 数据库

## 安装步骤

1. 克隆仓库：
```bash
git clone https://github.com/Lathezero/License-Plate-Detection.git
cd License-Plate-Detection
```

2. 创建并激活 conda 环境：
```bash
conda create -y -n kenshutsu python=3.8
conda activate kenshutsu
```

3. 安装依赖：
```bash
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

4. 配置数据库：
```bash
# 创建数据库
mysql -u root -p < db.sql
```

​	修改 `db_operations.py` 中的 `user` 和 `password`

## 使用方法

### 网页界面（推荐）

1. 激活 conda 环境：
```bash
conda activate kenshutsu
```

2. 启动网页服务器：
```bash
python main.py
```

3. 在浏览器中打开显示的地址即可访问界面。

### 命令行界面

#### 视频处理：

1. 将视频文件放在根目录下
2. 修改 `videoToImg.py` 中的 `output_path` 为你的视频文件路径
3. 依次运行以下命令：
```bash
python videoToImg.py    # 将视频转换为帧
python kenshutsu.py     # 处理视频帧
python imgToVideo.py    # 将处理后的帧合成为视频
```

#### 图片处理：

1. 将图片放在相应目录下
2. 修改 `kenshutsu.py`：
   - 取消注释第109行：`root = 'videoToImg'`
   - 注释掉下面一行
   - 取消注释第145和146行
3. 运行检测：
```bash
python kenshutsu.py
```

处理后的图片将保存在 `imgToVideo` 目录中。

## 项目结构

- `main.py` - 网页界面实现
- `kenshutsu.py` - 主要检测脚本
- `videoToImg.py` - 视频转图片
- `imgToVideo.py` - 图片转视频
- `db_operations.py` - 数据库操作
- `db.sql` - 数据库结构定义
- `models/` - 模型定义
- `weights/` - 预训练模型权重
- `utils/` - 工具函数
- `imgToVideo/` - 处理后图片输出目录
- `videoToImg/` - 视频帧存储目录
- `result_videos/` - 处理后视频存储目录
- `user_upload_videos/` - 用户上传视频存储目录

## 数据库功能

系统使用 MySQL 数据库存储以下信息：
- 车牌信息
  - 车牌号
  - 检测时间
- 车主信息
  - 车主姓名
- 历史记录
  - 检测记录
  - 统计信息

## 依赖说明

项目使用的主要依赖：
- PyTorch：深度学习框架
- OpenCV：图像处理
- Flask 和 FastAPI：网页界面
- MySQL：数据库操作
- mysql-connector-python：MySQL 数据库连接
- 其他工具库（详见 requirements.txt）

