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
requirements = python3,kivy==2.2.1,kivymd==1.1.1,numpy==1.21.6,scipy==1.7.3,matplotlib==3.5.3,pyjnius==1.5.0,android==1.0.0
p4a.bootstrap = sdl2
p4a.branch = develop
p4a.local_recipes = .
orientation = portrait
osx.python_version = 3
osx.kivy_version = 2.2.1
fullscreen = 0

# Android settings
android.api = 31
android.minapi = 24
android.ndk = 25b
android.gradle_dependencies = androidx.core:core:1.7.0
android.permissions = RECORD_AUDIO,MODIFY_AUDIO_SETTINGS,FOREGROUND_SERVICE,WAKE_LOCK,REQUEST_IGNORE_BATTERY_OPTIMIZATIONS
android.wakelock = True
android.allow_backup = True
android.backup_rules = backup_rules.xml
android.arch = armeabi-v7a
android.enable_androidx = True
android.add_compile_options = -Xlint:deprecation
android.add_gradle_repositories = maven { url 'https://jitpack.io' }
android.accept_sdk_license = True
android.archs = arm64-v8a, armeabi-v7a
android.sdk_download_url = https://mirrors.tuna.tsinghua.edu.cn/android/repository/
android.ndk_download_url = https://mirrors.tuna.tsinghua.edu.cn/android/repository/

# Application settings
android.appid = org.morse.morsetrainer
android.appname = Morse Trainer
android.description = 摩尔斯电码训练器，支持实时解码和AI预测
author = Morse Trainer Team
author.email = support@morsetrainer.org
android.category = EDUCATION
android.theme = Theme.MaterialComponents.DayNight.NoActionBar

# Feature flags
android.private_storage = True
android.use_sdcard = True
android.foreground_service = True
android.record_audio = True
android.use_network = False
android.use_location = False
android.use_camera = False
android.use_bluetooth = False
android.use_nfc = False
android.use_sensors = False
android.use_vibrator = True
android.use_notification = True
android.use_storage = True
android.use_contacts = False
android.use_calendar = False
android.use_sms = False
android.use_telephony = False
android.use_wifi = False

# Build settings
requirements.source.pip = https://mirrors.aliyun.com/pypi/simple/
p4a.hook = 

# 添加debuggable选项以便调试
android.release_artifact = apk
android.debuggable = True

[buildozer]
log_level = 2
warn_on_root = 1
