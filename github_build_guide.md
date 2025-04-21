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
- `optimize_repo.py` - 仓库优化脚本（解决上传和构建问题）

## 常见构建问题及解决方案

### 1. 工作流运行失败

可能的原因及解决方案：

- **依赖项冲突**: 确保buildozer.spec和requirements.txt中的依赖保持一致
  ```
  # 执行优化脚本清理冲突
  python optimize_repo.py --clean-all
  ```

- **构建超时**: 移除过大的依赖项（如torch）或使用预编译包
  ```
  # 编辑buildozer.spec文件，移除以下依赖：
  # torch, tensorboard, numba等大型库
  ```

- **Windows路径问题**: 在GitHub Actions上，应避免使用Windows特定路径
  ```
  # 移除buildozer.spec中的Windows路径：
  # android.windows_sdk_path = C:\\Android\\Sdk
  ```

### 2. 文件上传问题

可能的原因及解决方案：

- **文件过大**: GitHub有文件大小限制（通常为100MB）
  ```
  # 查找大文件
  python optimize_repo.py --find-large
  
  # 优化图像文件
  python optimize_repo.py --optimize-images
  ```

- **缓存和构建产物**: 应当排除不必要的缓存和构建文件
  ```
  # 清理构建产物和缓存
  python optimize_repo.py --clean-build --clean-cache
  ```

- **二进制文件**: 避免上传大型二进制文件，使用GitHub Releases管理APK
  ```
  # 更新.gitignore文件，排除bin目录和.apk文件
  ```

## 使用GitHub Actions上传代码并构建APK

### 步骤1：创建GitHub仓库

1. 登录您的GitHub账户
2. 点击右上角的"+"图标，选择"New repository"
3. 输入仓库名称（如"morse-trainer"）
4. 保持仓库设置为"Public"（除非您需要私有仓库）
5. 点击"Create repository"

### 步骤2：优化代码库

在上传之前，强烈建议使用优化脚本处理代码库：

```bash
# 安装脚本依赖
pip install Pillow

# 运行仓库优化
python optimize_repo.py --clean-all --optimize-images

# 创建优化后的归档（可选）
python optimize_repo.py --create-archive --output morse-trainer-optimized.zip
```

### 步骤3：使用GitHub Desktop上传代码

1. 下载并安装[GitHub Desktop](https://desktop.github.com/)
2. 打开GitHub Desktop并登录您的GitHub账户
3. 点击"File" -> "Add local repository"
4. 选择项目文件夹（包含src目录和buildozer.spec）
5. 如果提示"不是Git仓库"，点击"创建仓库"
6. 输入提交信息（如"初始提交"）
7. 点击"Commit to master"
8. 点击"Publish repository"
9. 确认仓库名称后点击"Publish repository"

### 步骤4：等待GitHub Actions自动构建APK

1. 代码上传后，GitHub会自动检测`.github/workflows/build-android.yml`文件
2. GitHub Actions将开始构建Android APK
3. 这个过程完全自动化，无需您的干预
4. 首次构建通常需要15-30分钟

### 步骤5：排查构建失败问题

如果构建失败，请检查以下内容：

1. 打开GitHub仓库页面，点击"Actions"选项卡
2. 找到失败的工作流运行，点击查看
3. 查看日志中的具体错误信息
4. 常见错误包括：
   - 依赖项冲突或安装失败
   - SDK或NDK版本问题
   - 构建超时（通常是因为依赖项太多或太大）
   - 权限问题

5. 根据错误信息修改配置文件，主要检查：
   - `buildozer.spec`
   - `requirements.txt`
   - `.github/workflows/build-android.yml`

6. 修改后提交更改，GitHub Actions会自动重新构建

### 步骤6：下载构建好的APK

1. 在GitHub仓库页面，点击"Actions"选项卡
2. 找到最新的完成的工作流
3. 点击进入工作流详情页面
4. 滚动到页面底部，在"Artifacts"部分找到"app-debug"
5. 点击下载链接获取APK文件（zip格式）
6. 解压下载的zip文件得到APK安装文件

### 步骤7：安装APK到Android设备

1. 将APK文件传输到Android设备
2. 在Android设备上打开文件管理器
3. 找到并点击APK文件开始安装
4. 如果提示"未知来源"安装警告，请允许安装
5. 等待安装完成，然后启动应用

## 高级定制

### 自定义构建过程

如果您需要自定义构建过程，可以修改 `.github/workflows/build-android.yml` 文件：

```yaml
# 添加缓存以加速构建
- name: Cache Buildozer global directory
  uses: actions/cache@v3
  with:
    path: ~/.buildozer
    key: ${{ runner.os }}-buildozer-global-${{ hashFiles('buildozer.spec') }}

# 自定义构建命令
- name: Build with Buildozer
  run: |
    export BUILDOZER_BUILD_MODE=debug
    export BUILDOZER_VERBOSE=1
    buildozer -v android debug
```

### 定制构建参数

修改`buildozer.spec`文件以自定义APK构建选项：

```ini
# 应用版本和包名
version = 1.0.0
package.name = morsetrainer
package.domain = org.morse

# 构建参数
android.api = 31
android.minapi = 21
android.archs = arm64-v8a, armeabi-v7a

# 启用调试选项
android.debuggable = True
```

## 持续集成最佳实践

- 使用语义化版本号（1.0.0, 1.0.1等）
- 每次提交前运行优化脚本
- 在大型更新前先分支测试
- 保留成功构建的APK
- 定期清理旧构建和缓存

## 相关资源

- [GitHub Actions文档](https://docs.github.com/cn/actions)
- [Buildozer文档](https://buildozer.readthedocs.io/)
- [Kivy文档](https://kivy.org/doc/stable/) 
- [Python-for-Android文档](https://python-for-android.readthedocs.io/) 