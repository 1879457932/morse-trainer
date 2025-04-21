# 摩尔斯电码训练器 (Morse Trainer)

摩尔斯电码训练器是一个用于学习和练习摩尔斯电码的Android应用，支持实时音频解码和AI辅助预测功能。

![Morse Trainer Logo](data/images/icon.png)

## 主要功能

- 实时音频解码：将声音转换为摩尔斯电码并显示
- 智能AI预测：基于上下文预测可能的字符
- 自适应频率调整：自动适应不同的信号频率
- 训练模式：帮助用户学习和记忆摩尔斯电码
- 悬浮窗模式：可在使用其他应用时继续接收和解码

## 技术栈

- Python 3.9+
- Kivy/KivyMD框架用于跨平台UI
- 音频处理：NumPy和SciPy
- Buildozer用于Android打包

## 项目结构

```
morse-trainer/
├── .github/workflows/    # GitHub Actions配置
├── src/                  # 源代码
│   ├── main.py           # 应用程序入口点
│   ├── morse_decoder.py  # 摩尔斯解码器
│   ├── morse_ai.py       # AI预测模块
│   ├── audio_processor.py # 音频处理模块
│   ├── morse.kv          # Kivy UI布局文件
│   └── service/          # Android服务相关代码
├── data/                 # 静态资源
│   └── images/           # 图标和启动画面
├── buildozer.spec        # Buildozer配置文件
├── create_icons.py       # 生成应用图标的脚本
└── requirements.txt      # Python依赖项
```

## 开发环境设置

### Windows开发环境

1. 安装Python 3.9
```
winget install Python.Python.3.9
```

2. 安装依赖
```
pip install -r requirements.txt
```

3. 安装额外的开发工具
```
pip install black pylint pytest
```

### 本地运行

直接运行主程序：
```
python src/main.py
```

### 构建Android APK

1. 安装Buildozer
```
pip install buildozer
```

2. 生成图标
```
python create_icons.py
```

3. 构建调试APK
```
buildozer android debug
```

## 使用GitHub Actions构建

1. Fork此仓库或创建自己的仓库
2. 上传代码
3. GitHub Actions会自动构建APK
4. 在Actions选项卡中下载构建好的APK

## 构建问题排查

如果构建过程中遇到问题：

1. 检查日志中的错误消息
2. 确保所有依赖项在requirements.txt和buildozer.spec中保持一致
3. 系统依赖问题参考buildozer文档

## 贡献指南

欢迎通过以下方式贡献：

1. 提交Issue报告bug或建议新功能
2. 通过Pull Request提交代码改进
3. 改进文档或提供使用示例

## 许可证

本项目采用MIT许可证 