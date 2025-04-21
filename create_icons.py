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
            return True  # 返回True表示成功
            
        # 创建应用图标（不同尺寸）
        sizes = [32, 48, 64, 96, 128, 192, 256, 512]
        for size in sizes:
            img = Image.new('RGBA', (size, size), (0, 0, 0, 0))
            draw = ImageDraw.Draw(img)
            
            # 绘制背景
            draw.ellipse((0, 0, size, size), fill=(46, 125, 50, 255))
            
            # 绘制摩尔斯电码符号（点和划）
            padding = size // 4
            center_y = size // 2
            
            # 点
            dot_radius = max(size // 10, 1)  # 确保至少有1像素
            dot_x = padding + dot_radius
            draw.ellipse((dot_x - dot_radius, center_y - dot_radius, 
                         dot_x + dot_radius, center_y + dot_radius), 
                         fill=(255, 255, 255, 255))
            
            # 划
            dash_width = max(size // 3, 2)  # 确保至少有2像素
            dash_height = max(size // 10, 1)  # 确保至少有1像素
            dash_x = size - padding - dash_width
            draw.rectangle((dash_x, center_y - dash_height//2, 
                           dash_x + dash_width, center_y + dash_height//2), 
                           fill=(255, 255, 255, 255))
            
            # 保存图标前确保目录存在
            try:
                img.save(f"data/images/icon_{size}.png")
                print(f"生成图标: icon_{size}.png")
            except Exception as e:
                print(f"保存图标 icon_{size}.png 时出错: {e}")
            
            # 512x512的图标复制为主图标
            if size == 512:
                try:
                    img.save("data/images/icon.png")
                    print("生成主图标: icon.png")
                except Exception as e:
                    print(f"保存主图标时出错: {e}")

        # 创建启动画面，使用较小的尺寸以减少内存消耗
        presplash_size = (512, 512)  # 降低分辨率以节省内存
        img = Image.new('RGB', presplash_size, (0, 0, 0))  # 使用RGB而非RGBA以节省内存
        draw = ImageDraw.Draw(img)

        # 渐变背景，使用步长以减少操作次数
        step = 4  # 每4像素绘制一次
        for y in range(0, presplash_size[1], step):
            # 从顶部的深绿色渐变到底部的黑色
            g = int(46 * (1 - y/presplash_size[1]))
            color = (0, g, 0)
            for i in range(min(step, presplash_size[1] - y)):
                draw.line([(0, y+i), (presplash_size[0], y+i)], fill=color)

        # 添加文字（用图形代替，因为PIL简单绘图不支持字体）
        center_x = presplash_size[0] // 2
        center_y = presplash_size[1] // 2

        # 绘制摩尔斯电码字符
        morse_patterns = [
            # M (--): 两个划
            [(center_x - 100, center_y - 50), (center_x - 40, center_y - 50), 10],
            [(center_x - 30, center_y - 50), (center_x + 30, center_y - 50), 10],
            
            # T (-): 一个划
            [(center_x - 65, center_y), (center_x + 65, center_y), 10],
            
            # 下方绘制点和划的符号作为装饰
            # 点
            [(center_x - 50, center_y + 50), 10],
            # 划
            [(center_x, center_y + 50), (center_x + 50, center_y + 50), 10],
        ]

        # 绘制摩尔斯图案
        for pattern in morse_patterns:
            if len(pattern) == 3:  # 这是一个划
                start, end, width = pattern
                draw.line([start, end], fill=(255, 255, 255), width=width)
            else:  # 这是一个点
                center, radius = pattern
                draw.ellipse((center[0] - radius, center[1] - radius, 
                             center[0] + radius, center[1] + radius), 
                             fill=(255, 255, 255))

        # 保存启动画面
        try:
            img.save("data/images/presplash.png")
            print("生成启动画面: presplash.png")
        except Exception as e:
            print(f"保存启动画面时出错: {e}")

        print("图标和启动画面创建完成！")
        return True
    except Exception as e:
        print(f"创建图标时发生错误: {e}")
        return False

if __name__ == "__main__":
    success = create_icons()
    if not success:
        sys.exit(1)  # 如果失败，返回非零退出码
    sys.exit(0)  # 如果成功，返回零退出码 