# 摩尔斯电码训练器 (Morse Trainer)

这是一个使用Kivy框架开发的Android摩尔斯电码解码和训练应用。本应用可以实时解码通过麦克风输入的摩尔斯电码信号，并使用AI进行预测和辅助学习。

## 主要功能

- **实时音频处理**：捕获和分析麦克风输入，提取摩尔斯电码信号
- **自适应解码**：智能调整解码参数，适应不同发送速度和信号质量
- **AI辅助识别**：使用神经网络模型提高识别准确率和错误纠正
- **训练模式**：提供练习环境，帮助用户学习摩尔斯电码
- **性能监控**：显示实时波形和频谱分析
- **历史记录**：保存解码结果和AI预测，支持数据导出分析
- **悬浮窗模式**：Android设备上支持悬浮窗操作

## 项目结构

- `src/`: 源代码目录
  - `main.py`: 主程序入口和UI控制
  - `morse_decoder.py`: 摩尔斯解码器算法
  - `audio_processor.py`: 音频处理和信号分析
  - `morse_ai.py`: AI预测模型实现
  - `morse.kv`: Kivy UI界面定义
  - `service/`: Android前台服务实现
- `data/images/`: 应用图标和启动画面
- `.github/workflows/`: GitHub Actions自动构建配置

## 技术特点

- **高效信号处理**：使用FFT和带通滤波提取摩尔斯信号
- **轻量级AI模型**：针对移动设备优化的神经网络
- **前台服务**：确保Android后台运行不被系统杀死
- **低功耗设计**：优化算法减少电池消耗
- **Material Design界面**：美观且符合现代UI设计规范

## 构建和安装

本项目使用GitHub Actions自动构建Android APK，详细步骤请参考 [github_build_guide.md](github_build_guide.md)。

简要步骤：
1. 将代码上传到GitHub仓库
2. GitHub Actions自动构建APK
3. 从Actions页面下载构建好的APK
4. 在Android设备上安装APK

## 手动构建（可选）

如果需要手动构建，请确保已安装以下工具：
- Python 3.9+
- Kivy 2.2.1+
- Buildozer

构建命令：
```bash
buildozer android debug
```

## 使用说明

1. 启动应用并授予麦克风权限
2. 点击录音按钮开始捕获音频
3. 向麦克风输入摩尔斯电码（声音、灯光或按键）
4. 应用将实时解码并显示结果
5. 可以调整目标频率、带宽和灵敏度参数优化识别效果
6. 使用AI训练按钮可以根据历史数据改进模型

## 许可证

MIT License 