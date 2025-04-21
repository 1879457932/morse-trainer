#!/usr/bin/env python3
"""
Android NDK下载辅助脚本

用于在GitHub Actions和其他CI环境中替代buildozer内部的下载逻辑，
解决tempfile相关问题和确保更高的下载稳定性
"""

import os
import sys
import shutil
import argparse
import subprocess
import platform
from pathlib import Path

def get_ndk_url(version='25b', system=None):
    """获取特定版本NDK的下载URL"""
    if system is None:
        system = platform.system().lower()
        
    if system == 'linux':
        return f"https://dl.google.com/android/repository/android-ndk-r{version}-linux.zip"
    elif system == 'darwin' or system == 'macos':
        return f"https://dl.google.com/android/repository/android-ndk-r{version}-darwin.zip"
    elif system == 'windows':
        return f"https://dl.google.com/android/repository/android-ndk-r{version}-windows.zip"
    else:
        raise ValueError(f"不支持的操作系统: {system}")

def get_buildozer_dir():
    """获取buildozer目录"""
    home = Path.home()
    return home / '.buildozer'

def download_file(url, target_path):
    """下载文件到指定路径"""
    print(f"下载 {url} 到 {target_path}...")
    
    # 确保目标目录存在
    os.makedirs(os.path.dirname(target_path), exist_ok=True)
    
    # 尝试不同的下载工具
    if shutil.which('wget'):
        # 使用wget
        result = subprocess.run(['wget', '-q', '--show-progress', '-O', target_path, url])
        if result.returncode != 0:
            raise Exception(f"wget下载失败，返回代码: {result.returncode}")
    elif shutil.which('curl'):
        # 使用curl
        result = subprocess.run(['curl', '-L', '-o', target_path, url])
        if result.returncode != 0:
            raise Exception(f"curl下载失败，返回代码: {result.returncode}")
    else:
        # 使用纯Python下载，避免使用urllib（可能会有tempfile问题）
        try:
            import requests
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            block_size = 1024  # 1 KB
            downloaded = 0
            
            with open(target_path, 'wb') as f:
                for data in response.iter_content(block_size):
                    f.write(data)
                    downloaded += len(data)
                    # 显示进度
                    if total_size > 0:
                        percent = downloaded * 100 // total_size
                        sys.stdout.write(f"\r下载进度: {percent}% [{downloaded} / {total_size}]")
                        sys.stdout.flush()
            print("\n下载完成！")
            
        except ImportError:
            try:
                # 由于tempfile问题，这种方法可能在某些环境下不稳定
                import urllib.request
                urllib.request.urlretrieve(url, target_path)
            except Exception as e:
                raise Exception(f"下载失败: {e}. 请安装wget, curl或requests库")

def unzip_file(zip_path, extract_path):
    """解压文件"""
    print(f"解压 {zip_path} 到 {extract_path}...")
    
    # 确保目标目录存在
    os.makedirs(extract_path, exist_ok=True)
    
    # 使用不同的解压工具
    if shutil.which('unzip'):
        # 使用unzip工具
        result = subprocess.run(['unzip', '-q', '-o', zip_path, '-d', extract_path])
        if result.returncode != 0:
            raise Exception(f"unzip解压失败，返回代码: {result.returncode}")
    else:
        # 使用Python的zipfile
        import zipfile
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_path)

def setup_ndk(version='25b', system=None, force=False):
    """设置Android NDK"""
    buildozer_dir = get_buildozer_dir()
    android_dir = buildozer_dir / 'android'
    platform_dir = android_dir / 'platform'
    ndk_dir = platform_dir / f'android-ndk-r{version}'
    
    # 确保目录存在
    os.makedirs(platform_dir, exist_ok=True)
    
    # 如果已存在且不强制重新下载，则跳过
    if ndk_dir.exists() and not force:
        print(f"NDK已存在于 {ndk_dir}，跳过下载")
        return ndk_dir
        
    # 获取下载URL
    url = get_ndk_url(version, system)
    
    # 临时zip文件路径
    zip_file = platform_dir / f'android-ndk-r{version}.zip'
    
    try:
        # 下载
        download_file(url, zip_file)
        
        # 解压
        unzip_file(zip_file, platform_dir)
        
        # 清理zip文件
        if zip_file.exists():
            os.remove(zip_file)
            print(f"已删除临时文件 {zip_file}")
            
        print(f"NDK已成功安装到 {ndk_dir}")
        return ndk_dir
        
    except Exception as e:
        print(f"安装NDK失败: {e}")
        if zip_file.exists():
            try:
                os.remove(zip_file)
            except:
                pass
        return None

def main():
    parser = argparse.ArgumentParser(description='下载和安装Android NDK')
    parser.add_argument('--version', default='25b', help='NDK版本 (例如 "25b")')
    parser.add_argument('--system', help='目标系统 (linux, darwin, windows)')
    parser.add_argument('--force', action='store_true', help='强制重新下载，即使已存在')
    
    args = parser.parse_args()
    
    try:
        ndk_dir = setup_ndk(args.version, args.system, args.force)
        if ndk_dir:
            print(f"NDK安装成功: {ndk_dir}")
            return 0
        else:
            print("NDK安装失败")
            return 1
    except Exception as e:
        print(f"出错: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 