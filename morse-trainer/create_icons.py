from PIL import Image, ImageDraw
import os

# 确保目录存在
os.makedirs("data/images", exist_ok=True)

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
    dot_radius = size // 10
    dot_x = padding + dot_radius
    draw.ellipse((dot_x - dot_radius, center_y - dot_radius, 
                 dot_x + dot_radius, center_y + dot_radius), 
                 fill=(255, 255, 255, 255))
    
    # 划
    dash_width = size // 3
    dash_height = size // 10
    dash_x = size - padding - dash_width
    draw.rectangle((dash_x, center_y - dash_height//2, 
                   dash_x + dash_width, center_y + dash_height//2), 
                   fill=(255, 255, 255, 255))
    
    # 保存图标
    img.save(f"data/images/icon_{size}.png")
    
    # 512x512的图标复制为主图标
    if size == 512:
        img.save("data/images/icon.png")

# 创建启动画面
presplash_size = (1024, 1024)
img = Image.new('RGBA', presplash_size, (0, 0, 0, 255))
draw = ImageDraw.Draw(img)

# 渐变背景
for y in range(presplash_size[1]):
    # 从顶部的深绿色渐变到底部的黑色
    g = int(46 * (1 - y/presplash_size[1]))
    draw.line([(0, y), (presplash_size[0], y)], fill=(0, g, 0, 255))

# 添加文字（用图形代替，因为PIL简单绘图不支持字体）
center_x = presplash_size[0] // 2
center_y = presplash_size[1] // 2

# 绘制摩尔斯电码字符
morse_patterns = [
    # M (--): 两个划
    [(center_x - 200, center_y - 100), (center_x - 80, center_y - 100), 20],
    [(center_x - 60, center_y - 100), (center_x + 60, center_y - 100), 20],
    
    # T (-): 一个划
    [(center_x - 130, center_y), (center_x + 130, center_y), 20],
    
    # 下方绘制点和划的符号作为装饰
    # 点
    [(center_x - 100, center_y + 100), 20],
    # 划
    [(center_x, center_y + 100), (center_x + 100, center_y + 100), 20],
]

# 绘制摩尔斯图案
for pattern in morse_patterns:
    if len(pattern) == 3:  # 这是一个划
        start, end, width = pattern
        draw.line([start, end], fill=(255, 255, 255, 255), width=width)
    else:  # 这是一个点
        center, radius = pattern
        draw.ellipse((center[0] - radius, center[1] - radius, 
                     center[0] + radius, center[1] + radius), 
                     fill=(255, 255, 255, 255))

# 保存启动画面
img.save("data/images/presplash.png")

print("图标和启动画面创建完成！") 