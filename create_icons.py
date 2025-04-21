from PIL import Image, ImageDraw
import os
import sys

def create_icons():
    """创建应用图标和启动画面"""
    try:
        # 确保目录存在
        os.makedirs("data/images", exist_ok=True)
        
        # 首先检查是否已存在图标，如存在则跳过生成
        if os.path.exists("data/images/icon.png") and os.path.exists("data/images/presplash.png"):
            print("图标和启动画面已存在，跳过生成。")
            return True
        
        # 在CI环境中识别并避免潜在问题
        is_ci = os.environ.get('CI', 'false').lower() == 'true'
        if is_ci:
            print("检测到CI环境，使用最小配置创建图标...")
            
        # 创建应用图标（不同尺寸）
        success = False  # 用于跟踪是否成功生成了至少一个图标
        
        sizes = [32, 48, 64, 96, 128, 192, 256, 512]
        for size in sizes:
            try:
                img = Image.new('RGB', (size, size), (46, 125, 50))  # 使用RGB而非RGBA以提高兼容性
                draw = ImageDraw.Draw(img)
                
                # 绘制背景 - 简化为实心圆形
                draw.ellipse((0, 0, size, size), fill=(46, 125, 50))
                
                # 绘制摩尔斯电码符号（点和划）- 使用更简单的绘制方式
                center_y = size // 2
                
                # 点
                dot_radius = max(size // 10, 1)
                dot_x = size // 3
                draw.ellipse((dot_x - dot_radius, center_y - dot_radius, 
                             dot_x + dot_radius, center_y + dot_radius), 
                             fill=(255, 255, 255))
                
                # 划
                dash_width = max(size // 4, 2)
                dash_height = max(size // 10, 1)
                dash_x = 2 * size // 3 - dash_width // 2
                draw.rectangle((dash_x, center_y - dash_height//2, 
                               dash_x + dash_width, center_y + dash_height//2), 
                               fill=(255, 255, 255))
                
                # 保存图标前确保目录存在
                img.save(f"data/images/icon_{size}.png")
                print(f"生成图标: icon_{size}.png")
                
                # 标记至少一个图标创建成功
                success = True
                
                # 512x512的图标复制为主图标
                if size == 512:
                    img.save("data/images/icon.png")
                    print("生成主图标: icon.png")
            except Exception as e:
                print(f"创建 {size}x{size} 图标时出错: {e}")
                # 继续尝试创建其他尺寸的图标

        # 如果没有成功创建任何图标，创建一个简单的默认图标
        if not success or not os.path.exists("data/images/icon.png"):
            try:
                # 创建一个非常简单的图标
                simple_img = Image.new('RGB', (128, 128), (46, 125, 50))
                simple_img.save("data/images/icon.png")
                print("创建了简单的默认图标")
                success = True
            except Exception as e:
                print(f"创建简单默认图标时出错: {e}")

        # 创建启动画面
        try:
            # 使用非常简单的启动画面设计
            splash_img = Image.new('RGB', (512, 512), (0, 100, 0))
            splash_img.save("data/images/presplash.png")
            print("生成启动画面: presplash.png")
            success = True
        except Exception as e:
            print(f"创建启动画面时出错: {e}")
            
            # 如果创建失败，尝试创建最简单的启动画面
            try:
                simple_splash = Image.new('RGB', (256, 256), (0, 0, 0))
                simple_splash.save("data/images/presplash.png")
                print("创建了简单的默认启动画面")
                success = True
            except Exception as e2:
                print(f"创建简单启动画面也失败: {e2}")

        # 返回是否至少有一个图标或启动画面创建成功
        return success
    except Exception as e:
        print(f"创建图标时发生错误: {e}")
        
        # 尝试使用最简单方法创建文件
        try:
            with open("data/images/icon.png", "wb") as f:
                f.write(b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x10\x00\x00\x00\x10\x08\x02\x00\x00\x00\x90\x91h6\x00\x00\x00\x01sRGB\x00\xae\xce\x1c\xe9\x00\x00\x00\x15IDAT(\x91c\xfc\xff\xff?\x03\x05\x80\x91\x81\x02\x00\xaa\x00\x00\x00\xff\xff\x03\x00\x06\xd8\x027r\xa3\xec\xcf\x00\x00\x00\x00IEND\xaeB`\x82')
            with open("data/images/presplash.png", "wb") as f:
                f.write(b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x10\x00\x00\x00\x10\x08\x02\x00\x00\x00\x90\x91h6\x00\x00\x00\x01sRGB\x00\xae\xce\x1c\xe9\x00\x00\x00\x15IDAT(\x91c\xfc\xff\xff?\x03\x05\x80\x91\x81\x02\x00\xaa\x00\x00\x00\xff\xff\x03\x00\x06\xd8\x027r\xa3\xec\xcf\x00\x00\x00\x00IEND\xaeB`\x82')
            print("使用预设的最小PNG文件创建了图标")
            return True
        except Exception as e2:
            print(f"所有创建图标的尝试都失败了: {e2}")
            return False

if __name__ == "__main__":
    success = create_icons()
    if not success:
        sys.exit(1)  # 如果失败，返回非零退出码
    sys.exit(0)  # 如果成功，返回零退出码 