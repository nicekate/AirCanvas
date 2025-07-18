# 🎨 AirCanvas - 空中画布

> Draw in the air using hand gestures | 用手势在空中绘画

[English](#english) | [中文](#中文)

---

## English

AirCanvas is a real-time air drawing application that allows users to draw in the air using hand gestures without any physical contact. Powered by MediaPipe for high-precision hand detection, it supports real-time gesture recognition and various drawing features.

### ✨ Features

- 🎨 **Air Drawing**: Draw by moving your index finger in the air with real-time trajectory display
- 🌈 **Gesture Color Switching**: Switch between 8 different colors using hand gestures
- 🧹 **Gesture Canvas Clearing**: Clear the entire canvas with a multi-finger gesture
- 📹 **Real-time Preview**: Live display of drawing effects, hand tracking, and gesture status
- 🎯 **Precise Recognition**: High-precision hand detection and gesture recognition using MediaPipe
- 💡 **Smart Anti-mistouch**: Built-in cooldown mechanism to prevent accidental operations
- 🖥️ **Intuitive Interface**: Real-time finger status display, current color, and operation instructions

### 🚀 Quick Start

#### Requirements
- Python 3.8+
- Camera device
- Good lighting conditions

#### Installation

**Method 1: Using uv (Recommended)**
```bash
# 1. Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Clone or download the project
# 3. Navigate to project directory

# 4. Install dependencies and run
uv sync                        # Create virtual environment and install dependencies
uv run python air_drawing.py   # Run the application
```

**Method 2: Using pip**
```bash
# 1. Clone or download the project
# 2. Navigate to project directory

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the application
python air_drawing.py
```

### 🎮 Controls

| Gesture | Function | Description |
|---------|----------|-------------|
| 🖕 **Index finger only** | Drawing mode | Index fingertip trajectory will be drawn on canvas |
| ✌️ **Index + Middle finger** | Switch color | Cycle through 8 preset colors |
| 🖐 **4 or 5 fingers** | Clear canvas | Clear the entire canvas |

#### Available Colors
8 preset colors in cycling order:
1. 🔴 Red 2. 🟢 Green 3. 🔵 Blue 4. 🟡 Yellow 5. 🟣 Purple 6. 🔵 Cyan 7. ⚪ White 8. ⚫ Black

#### Keyboard Shortcuts
- **'q'**: Quit application
- **'c'**: Clear canvas
- **'n'**: Switch color manually

---

## 中文

这是一个基于计算机视觉和手势识别的空中绘画应用，让用户可以通过手指在空中绘画，无需任何物理接触。使用MediaPipe进行高精度手部检测，支持实时手势识别和多种绘画功能。

## ✨ 功能特点

- 🎨 **空中绘画**: 通过食指在空中移动来绘画，轨迹实时显示
- 🌈 **手势换色**: 通过手势切换8种不同颜色（红、绿、蓝、黄、紫、青、白、黑）
- 🧹 **手势清除**: 通过手势一键清除整个画布
- 📹 **实时预览**: 实时显示绘画效果、手部追踪和手势状态
- 🎯 **精确识别**: 使用MediaPipe进行高精度手部检测和手势识别
- 💡 **智能防误触**: 内置冷却机制，防止误操作
- 🖥️ **直观界面**: 实时显示手指状态、当前颜色和操作说明

## 🚀 快速开始

### 环境要求
- Python 3.8+
- 摄像头设备
- 良好的光照条件

### 安装方式

#### 方法一：使用 uv (推荐)

```bash
# 1. 安装 uv (如果还没安装)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. 克隆或下载项目到本地
# 3. 进入项目目录

# 4. 安装依赖并运行
uv sync                        # 创建虚拟环境并安装依赖
uv run python air_drawing.py   # 运行应用
```

#### 方法二：使用 pip

```bash
# 1. 克隆或下载项目到本地
# 2. 进入项目目录

# 3. 安装依赖
pip install -r requirements.txt

# 4. 运行应用
python air_drawing.py
```

## 🎮 操作指南

### 手势控制

| 手势 | 功能 | 说明 |
|------|------|------|
| 🖕 **只伸出食指** | 绘画模式 | 食指尖端轨迹将被实时绘制到画布上 |
| ✌️ **食指+中指** | 切换颜色 | 循环切换8种预设颜色 |
| 🖐 **4个或5个手指** | 清除画布 | 一键清空整个画布 |

### 可用颜色
应用提供8种预设颜色，按顺序循环切换：
1. 🔴 红色 (Red)
2. 🟢 绿色 (Green)
3. 🔵 蓝色 (Blue)
4. 🟡 黄色 (Yellow)
5. 🟣 紫色 (Purple)
6. 🔵 青色 (Cyan)
7. ⚪ 白色 (White)
8. ⚫ 黑色 (Black)

### 键盘快捷键
- **'q'**: 退出应用
- **'c'**: 清除画布
- **'n'**: 手动切换颜色

## 🖥️ 界面说明

应用界面分为几个区域，提供丰富的实时信息：

- **左上角**: 当前选中的颜色名称
- **左侧中部**: 实时手势检测状态
  - 显示检测到的手指数量 (Fingers: X/5)
  - 显示每个手指的状态 (UP/DOWN)
- **左下角**: 操作说明和快捷键
- **右上角**: 颜色选择器，显示所有可用颜色
- **中央**: 摄像头画面 + 绘画画布叠加显示

## 💡 使用技巧

### 获得最佳体验
1. **环境设置**
   - 确保光线充足，避免背光
   - 选择简洁的背景，避免复杂图案
   - 保持摄像头稳定，避免晃动

2. **手势技巧**
   - 保持手部完全在摄像头视野内
   - 做手势时要清晰明确，避免模糊动作
   - 绘画时保持手部相对稳定，移动要平滑
   - 切换手势时稍作停顿，让系统识别

3. **绘画技巧**
   - 从简单的线条和形状开始练习
   - 利用不同颜色创作更丰富的作品
   - 善用清除功能重新开始

### 故障排除
- **手势识别不准确**: 检查光照条件，调整手部位置
- **绘画断断续续**: 确保只伸出食指，其他手指收起
- **颜色切换频繁**: 系统有冷却机制，稍等片刻再尝试

## 🔧 常见问题解决

### 摄像头相关问题

**问题**: 摄像头无法打开
- **解决**: 检查摄像头是否被其他应用占用（如Zoom、Skype等）
- **macOS用户**: 检查系统偏好设置 → 安全性与隐私 → 隐私 → 摄像头，确保Python/终端有权限

**问题**: 画面显示异常
- **解决**: 重启应用，检查摄像头连接

### 手势识别问题

**问题**: 手部检测不准确
- **解决**:
  - 调整光照条件，确保光线充足
  - 调整摄像头角度和距离
  - 确保手部完全在视野内
  - 避免复杂背景

**问题**: 绘画线条不连续
- **解决**:
  - 确保只伸出食指，其他手指完全收起
  - 保持手部稳定，避免快速移动
  - 检查实时手势状态显示

**问题**: 手势切换不灵敏
- **解决**:
  - 手势要清晰明确，保持1-2秒
  - 系统有防误触冷却机制，稍等片刻再尝试
  - 观察界面上的手指状态显示

## ⚙️ 自定义设置

您可以在 `air_drawing.py` 中调整以下参数来优化体验：

### 绘画设置
```python
self.brush_size = 5  # 画笔大小 (1-20)
```

### 检测精度设置
```python
min_detection_confidence=0.7  # 手部检测置信度 (0.1-1.0)
min_tracking_confidence=0.5   # 手部追踪置信度 (0.1-1.0)
```

### 颜色设置
```python
# 添加新颜色 (BGR格式)
self.colors['orange'] = (0, 165, 255)
self.colors['pink'] = (203, 192, 255)
```

### 冷却时间设置
```python
self.color_change_cooldown = 60  # 颜色切换冷却时间
self.clear_gesture_cooldown = 90 # 清除手势冷却时间
```

## 🛠️ 技术实现

### 核心技术栈
- **OpenCV**: 图像处理、摄像头控制和图形绘制
- **MediaPipe**: Google开发的手部检测和关键点识别框架
- **NumPy**: 高效的数值计算和图像数组操作

### 架构设计
- **实时处理**: 基于帧的实时图像处理管道
- **手势识别**: 21个手部关键点的精确检测和分析
- **状态管理**: 智能的绘画状态和手势状态管理
- **防误触机制**: 基于时间的冷却系统

### 性能特点
- **低延迟**: 实时手势识别和绘画响应
- **高精度**: MediaPipe提供亚像素级别的手部追踪
- **稳定性**: 内置错误处理和状态恢复机制

## 📁 项目结构

```
air-drawing/
├── air_drawing.py      # 主应用程序
├── pyproject.toml      # uv项目配置
├── requirements.txt    # pip依赖列表
└── README.md          # 项目说明文档
```

## 🚀 进阶使用

### 扩展功能建议
1. **添加更多手势**: 实现撤销、重做功能
2. **保存作品**: 添加画作保存和加载功能
3. **画笔设置**: 实现可调节的画笔大小和透明度
4. **多人协作**: 支持多手同时绘画

### 开发环境
推荐使用以下工具进行开发：
- **IDE**: VS Code 或 PyCharm
- **包管理**: uv (现代化的Python包管理器)
- **版本控制**: Git

## 📄 许可证

本项目采用 MIT 许可证，详情请参阅 LICENSE 文件。

## 🤝 贡献

欢迎提交 Issue 和 Pull Request 来改进这个项目！

---

**享受您的空中绘画体验！** 🎨✨
