[app]
title = Morse Trainer
package.name = morsetrainer
package.domain = org.morse
source.dir = .
source.include_exts = py,png,jpg,kv,atlas,json
source.include_patterns = src/*,data/*
source.exclude_dirs = tests,docs,bin,.vscode,.git
source.exclude_patterns = .gitignore,__pycache__/**,*.pyc,*.git*
source.main = src/main.py
version = 1.0.0
# 使用标准配置
requirements = python3==3.9,kivy==2.1.0
p4a.bootstrap = sdl2
orientation = portrait
fullscreen = 0

# 基本Android设置
android.api = 31
android.minapi = 24
android.ndk = 25b
android.arch = armeabi-v7a
android.permissions = RECORD_AUDIO

# 钩子
p4a.hook = .

# 调试设置
android.debug = True

[buildozer]
log_level = 2
warn_on_root = 1
