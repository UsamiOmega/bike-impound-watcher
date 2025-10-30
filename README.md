# 🚲 SITP - BikeWatch : 共享单车监控系统

> 基于YOLO的实时单车区域监控与事件检测系统

## 🎯 功能特点

- **实时检测** 🔍 - 使用YOLO模型实时检测视频流中的单车
- **区域监控** 🗺️ - 定义停车区(🟢)和通行区(🔴)多边形ROI
- **事件触发** ⚡ - 检测**阻塞事件**和**违规停放事件**
- **智能记录** 📝 - 自动保存事件截图和日志数据
- **性能优化** ⚡ - 支持CUDA加速，动态帧率控制

## 🛠️ 工作原理

### 检测流程:

视频输入 → YOLO检测 → ROI区域判断 → 事件逻辑处理 → 日志记录

### 核心逻辑
- **停车区检测** 🅿️ - 车辆是否在指定停车区域内
- **通行区检测** 🚦 - 车辆是否阻塞通行区域  
- **时间阈值** ⏱️ - 持续超过设定时间才触发事件
- **IoU计算** 📐 - 使用多边形交集判断车辆位置

## 📁 文件结构
- ./run(main).py  主程序
- ./config.json # 配置文件
- ./weights/* # 模型权重存储
- ./config_files/ # Ultralytics参数配置
- ./snaps/ # 事件截图
- ./logs.csv # 运行日志

## 🚀 运行方式

- 主程序：
```bash
python run(main).py
```
- ROI选择：
```bash
python roi_picker.py --capture                # 捕获frame.jpg
python roi_picker.py --mark <picture-path>    # 鼠标绘制多边形形式ROI
```
