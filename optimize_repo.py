#!/usr/bin/env python3
"""
优化仓库脚本 - 用于清理和压缩项目文件
此脚本帮助解决GitHub上传限制和工作流运行问题
"""

import os
import sys
import shutil
import glob
from PIL import Image
import zipfile
import argparse

def optimize_images(directory="data/images", max_size=1024, quality=85, verbose=True):
    """压缩图像文件以减小体积"""
    if not os.path.exists(directory):
        print(f"目录不存在: {directory}")
        return

    total_saved = 0
    for ext in ['*.png', '*.jpg', '*.jpeg']:
        for img_path in glob.glob(os.path.join(directory, ext)):
            original_size = os.path.getsize(img_path)
            
            try:
                img = Image.open(img_path)
                
                # 如果图像尺寸超过最大值，调整大小
                if max(img.size) > max_size:
                    ratio = max_size / max(img.size)
                    new_size = (int(img.size[0] * ratio), int(img.size[1] * ratio))
                    img = img.resize(new_size, Image.Resampling.LANCZOS)
                
                # 保存优化后的图像
                img.save(img_path, optimize=True, quality=quality)
                
                # 计算节省的空间
                new_size = os.path.getsize(img_path)
                saved = original_size - new_size
                total_saved += saved
                
                if verbose and saved > 0:
                    print(f"优化 {img_path}: {original_size/1024:.1f}KB -> {new_size/1024:.1f}KB (节省 {saved/1024:.1f}KB)")
            
            except Exception as e:
                print(f"处理图像 {img_path} 时出错: {e}")
    
    print(f"总计节省: {total_saved/1024:.1f}KB")

def clean_python_cache(directory="."):
    """清理Python缓存文件"""
    # 删除__pycache__目录
    pycache_dirs = []
    for root, dirs, files in os.walk(directory):
        for d in dirs:
            if d == "__pycache__":
                pycache_dirs.append(os.path.join(root, d))
    
    # 删除.pyc文件
    pyc_files = []
    for root, dirs, files in os.walk(directory):
        for f in files:
            if f.endswith(".pyc") or f.endswith(".pyo"):
                pyc_files.append(os.path.join(root, f))
    
    # 执行删除
    deleted_dirs = 0
    deleted_files = 0
    
    for d in pycache_dirs:
        try:
            shutil.rmtree(d)
            deleted_dirs += 1
            print(f"删除目录: {d}")
        except Exception as e:
            print(f"删除目录 {d} 时出错: {e}")
    
    for f in pyc_files:
        try:
            os.remove(f)
            deleted_files += 1
            print(f"删除文件: {f}")
        except Exception as e:
            print(f"删除文件 {f} 时出错: {e}")
    
    print(f"清理完成: 删除了 {deleted_dirs} 个目录和 {deleted_files} 个文件")

def clean_build_artifacts(directory="."):
    """清理构建产物"""
    # 要删除的目录
    dirs_to_remove = [
        ".buildozer",
        "bin",
        "build",
        "dist",
        ".pytest_cache",
        ".tox",
        "htmlcov",
        "__pycache__"
    ]
    
    # 要删除的文件模式
    file_patterns = [
        "*.apk",
        "*.aab",
        "*.so",
        "*.pyc",
        "*.pyo",
        "*.pyd",
        "*.log",
        "*.bak",
        "*.tmp",
        "*.temp",
        ".coverage"
    ]
    
    # 删除目录
    for d in dirs_to_remove:
        for path in glob.glob(os.path.join(directory, "**", d), recursive=True):
            if os.path.isdir(path):
                try:
                    shutil.rmtree(path)
                    print(f"删除目录: {path}")
                except Exception as e:
                    print(f"删除目录 {path} 时出错: {e}")
    
    # 删除文件
    deleted_files = 0
    for pattern in file_patterns:
        for path in glob.glob(os.path.join(directory, "**", pattern), recursive=True):
            if os.path.isfile(path):
                try:
                    os.remove(path)
                    deleted_files += 1
                except Exception as e:
                    print(f"删除文件 {path} 时出错: {e}")
    
    print(f"删除了 {deleted_files} 个文件")

def find_large_files(directory=".", min_size_kb=1000):
    """查找大文件"""
    large_files = []
    
    for root, dirs, files in os.walk(directory):
        # 排除一些目录
        if ".git" in root or ".buildozer" in root or "bin" in root:
            continue
            
        for file in files:
            file_path = os.path.join(root, file)
            try:
                size = os.path.getsize(file_path)
                if size > min_size_kb * 1024:
                    large_files.append((file_path, size))
            except Exception:
                pass
    
    # 按大小排序
    large_files.sort(key=lambda x: x[1], reverse=True)
    
    if large_files:
        print(f"找到 {len(large_files)} 个大于 {min_size_kb}KB 的文件:")
        for path, size in large_files:
            print(f"{path}: {size/1024/1024:.2f}MB")
    else:
        print(f"没有找到大于 {min_size_kb}KB 的文件")
    
    return large_files

def create_archive(output_file="optimized_project.zip", exclude_dirs=None, exclude_patterns=None):
    """创建优化后的项目归档"""
    if exclude_dirs is None:
        exclude_dirs = [".git", ".buildozer", "bin", "build", "dist", "__pycache__"]
    
    if exclude_patterns is None:
        exclude_patterns = ["*.pyc", "*.pyo", "*.log", "*.apk", "*.tmp", "*.bak"]
    
    # 确保输出目录存在
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    with zipfile.ZipFile(output_file, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk("."):
            # 过滤排除的目录
            dirs[:] = [d for d in dirs if d not in exclude_dirs and not any(d.endswith(ed) for ed in exclude_dirs)]
            
            for file in files:
                file_path = os.path.join(root, file)
                
                # 跳过归档文件本身
                if os.path.abspath(file_path) == os.path.abspath(output_file):
                    continue
                
                # 检查排除模式
                skip = False
                for pattern in exclude_patterns:
                    if pattern.startswith("*"):
                        if file.endswith(pattern[1:]):
                            skip = True
                            break
                
                if not skip:
                    arcname = file_path
                    if arcname.startswith("./"):
                        arcname = arcname[2:]
                    zipf.write(file_path, arcname)
    
    print(f"创建归档: {output_file}")
    print(f"归档大小: {os.path.getsize(output_file)/1024/1024:.2f}MB")

def main():
    parser = argparse.ArgumentParser(description="优化项目仓库")
    parser.add_argument("--clean-all", action="store_true", help="执行所有清理操作")
    parser.add_argument("--clean-cache", action="store_true", help="清理Python缓存文件")
    parser.add_argument("--clean-build", action="store_true", help="清理构建产物")
    parser.add_argument("--optimize-images", action="store_true", help="优化图像文件")
    parser.add_argument("--find-large", action="store_true", help="查找大文件")
    parser.add_argument("--min-size", type=int, default=1000, help="最小文件大小(KB)，用于查找大文件")
    parser.add_argument("--create-archive", action="store_true", help="创建优化后的项目归档")
    parser.add_argument("--output", type=str, default="optimized_project.zip", help="归档文件名")
    
    args = parser.parse_args()
    
    # 如果没有指定任何操作，显示帮助信息
    if not (args.clean_all or args.clean_cache or args.clean_build or 
            args.optimize_images or args.find_large or args.create_archive):
        parser.print_help()
        return
    
    if args.clean_all or args.clean_cache:
        clean_python_cache()
    
    if args.clean_all or args.clean_build:
        clean_build_artifacts()
    
    if args.clean_all or args.optimize_images:
        optimize_images()
    
    if args.clean_all or args.find_large:
        find_large_files(min_size_kb=args.min_size)
    
    if args.clean_all or args.create_archive:
        create_archive(output_file=args.output)

if __name__ == "__main__":
    main() 