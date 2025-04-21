# 使用GitHub Actions构建Android APK指南

本指南将帮助您使用GitHub Actions自动构建摩尔斯电码训练器应用的Android APK文件。

## 项目结构

摩尔斯电码训练器包含以下关键文件和目录：

- `src/` - 主要源代码目录
  - `main.py` - 应用程序入口点
  - `morse_decoder.py` - 摩尔斯解码器
  - `morse_ai.py` - AI预测模块
  - `audio_processor.py` - 音频处理模块
  - `morse.kv` - Kivy UI布局文件
  - `service/` - Android服务相关代码
- `data/images/` - 图标和启动画面
- `buildozer.spec` - Buildozer配置文件
- `create_icons.py` - 生成应用图标的脚本
- `.github/workflows/build-android.yml` - GitHub Actions配置

## 使用GitHub上传代码并构建APK

### 步骤1：创建GitHub仓库

1. 登录您的GitHub账户
2. 点击右上角的"+"图标，选择"New repository"
3. 输入仓库名称（如"morse-trainer"）
4. 保持仓库设置为"Public"（除非您需要私有仓库）
5. 点击"Create repository"

### 步骤2：使用GitHub Desktop上传代码

1. 下载并安装[GitHub Desktop](https://desktop.github.com/)
2. 打开GitHub Desktop并登录您的GitHub账户
3. 点击"File" -> "Add local repository"
4. 选择项目文件夹（包含src目录和buildozer.spec）
5. 如果提示"不是Git仓库"，点击"创建仓库"
6. 输入提交信息（如"初始提交"）
7. 点击"Commit to master"
8. 点击"Publish repository"
9. 确认仓库名称后点击"Publish repository"

### 步骤3：等待GitHub Actions自动构建APK

1. 代码上传后，GitHub会自动检测`.github/workflows/build-android.yml`文件
2. GitHub Actions将开始构建Android APK
3. 这个过程完全自动化，无需您的干预
4. 首次构建通常需要15-30分钟

### 步骤4：下载构建好的APK

1. 在GitHub仓库页面，点击"Actions"选项卡
2. 找到最新的完成的工作流
3. 点击进入工作流详情页面
4. 滚动到页面底部，在"Artifacts"部分找到"app-debug"
5. 点击下载链接获取APK文件（zip格式）
6. 解压下载的zip文件得到APK安装文件

### 步骤5：安装APK到Android设备

1. 将APK文件传输到Android设备
2. 在Android设备上打开文件管理器
3. 找到并点击APK文件开始安装
4. 如果提示"未知来源"安装警告，请允许安装
5. 等待安装完成，然后启动应用

## 构建失败时的故障排除

如果构建失败，请检查：

1. 查看Actions日志以了解具体错误
2. 确保buildozer.spec文件正确配置
3. 检查源代码中没有语法错误
4. 确保所有依赖项都正确列在buildozer.spec文件中

## 更新应用版本

要更新应用：

1. 修改本地代码
2. 在GitHub Desktop中提交更改
3. 推送到GitHub
4. GitHub Actions会自动构建新版本的APK

## 自定义构建过程

如果您需要自定义构建过程，可以修改 `.github/workflows/build-android.yml` 文件：

- 修改触发条件
- 添加更多的依赖
- 更改构建参数
- 添加测试步骤
- 配置自动发布到Google Play等

## 常见问题解决

1. **构建失败**：检查Actions日志以了解具体错误
2. **依赖问题**：确保buildozer.spec文件中的依赖正确
3. **权限问题**：某些API可能需要在构建中配置特殊权限

## 相关资源

- [GitHub Actions文档](https://docs.github.com/cn/actions)
- [Buildozer文档](https://buildozer.readthedocs.io/)
- [Kivy文档](https://kivy.org/doc/stable/) 