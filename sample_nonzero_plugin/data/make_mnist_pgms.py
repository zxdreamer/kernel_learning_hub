import os

# 目标尺寸（与 engine 的输入一致；MNIST 通常 28x28）
H, W = 28, 28
OUT_DIR = "./data/mnist"

# 7 段数码管的 7 个段：a,b,c,d,e,f,g
# 映射到每个数字开启的段
SEGMENTS_BY_DIGIT = {
    0: ("a", "b", "c", "d", "e", "f"),
    1: ("b", "c"),
    2: ("a", "b", "g", "e", "d"),
    3: ("a", "b", "g", "c", "d"),
    4: ("f", "g", "b", "c"),
    5: ("a", "f", "g", "c", "d"),
    6: ("a", "f", "g", "e", "c", "d"),
    7: ("a", "b", "c"),
    8: ("a", "b", "c", "d", "e", "f", "g"),
    9: ("a", "b", "c", "d", "f", "g"),
}

def make_canvas(h, w, bg=255):
    """创建 h×w 的白底画布（255）"""
    return [[bg for _ in range(w)] for _ in range(h)]

def draw_hline(img, y, x1, x2, thickness=2, val=0):
    """画水平粗线（闭区间），越界自动裁剪"""
    h, w = len(img), len(img[0])
    if y < 0 or y >= h: 
        return
    x1, x2 = max(0, x1), min(w - 1, x2)
    for yy in range(max(0, y - thickness), min(h - 1, y + thickness) + 1):
        for xx in range(x1, x2 + 1):
            img[yy][xx] = val

def draw_vline(img, x, y1, y2, thickness=2, val=0):
    """画垂直粗线（闭区间），越界自动裁剪"""
    h, w = len(img), len(img[0])
    if x < 0 or x >= w:
        return
    y1, y2 = max(0, y1), min(h - 1, y2)
    for xx in range(max(0, x - thickness), min(w - 1, x + thickness) + 1):
        for yy in range(y1, y2 + 1):
            img[yy][xx] = val

def draw_7seg_digit(img, digit, box, thickness=2, val=0):
    """
    在 img 上用 7 段数码管风格绘制一个数字。
    box = (top, left, bottom, right) 指绘制区域（闭区间）
    """
    t, l, b, r = box
    # 计算三条水平线 y 坐标：顶部 a、中间 g、底部 d
    y_top    = t + (b - t) // 10 * 2
    y_mid    = t + (b - t) // 2
    y_bottom = b - (b - t) // 10 * 2
    # 计算两条竖线 x 坐标：左 f/e、右 b/c
    x_left   = l + (r - l) // 10 * 2
    x_right  = r - (r - l) // 10 * 2
    # 竖线上下的分界（上半区、下半区）
    y_upper_end = t + (b - t) // 2 - (b - t) // 10
    y_lower_st  = t + (b - t) // 2 + (b - t) // 10

    segs = set(SEGMENTS_BY_DIGIT[digit])

    # a: 顶部水平线
    if "a" in segs:
        draw_hline(img, y_top,   x_left, x_right, thickness=thickness, val=val)
    # b: 右上竖线
    if "b" in segs:
        draw_vline(img, x_right, y_top,  y_upper_end, thickness=thickness, val=val)
    # c: 右下竖线
    if "c" in segs:
        draw_vline(img, x_right, y_lower_st, y_bottom, thickness=thickness, val=val)
    # d: 底部水平线
    if "d" in segs:
        draw_hline(img, y_bottom, x_left, x_right, thickness=thickness, val=val)
    # e: 左下竖线
    if "e" in segs:
        draw_vline(img, x_left, y_lower_st, y_bottom, thickness=thickness, val=val)
    # f: 左上竖线
    if "f" in segs:
        draw_vline(img, x_left, y_top, y_upper_end, thickness=thickness, val=val)
    # g: 中间水平线
    if "g" in segs:
        draw_hline(img, y_mid, x_left, x_right, thickness=thickness, val=val)

def save_pgm_p5(path, img_2d):
    """保存为 PGM (P5, binary, 8-bit)"""
    h, w = len(img_2d), len(img_2d[0])
    with open(path, "wb") as f:
        header = f"P5\n{w} {h}\n255\n".encode("ascii")
        f.write(header)
        # 逐行写入
        for row in img_2d:
            f.write(bytearray(row))

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # 在 28x28 内部留一个 20x20 的绘制区域，保证笔画不挤边缘
    box_side = min(H, W) - 8   # 28 -> 20
    top  = (H - box_side) // 2
    left = (W - box_side) // 2
    box  = (top, left, top + box_side - 1, left + box_side - 1)

    thickness = max(1, box_side // 12)  # 线条粗细

    for d in range(10):
        img = make_canvas(H, W, bg=255)  # 白底
        draw_7seg_digit(img, d, box, thickness=thickness, val=0)  # 数字笔画=0（黑）
        out_path = os.path.join(OUT_DIR, f"{d}.pgm")
        save_pgm_p5(out_path, img)
        print(f"Wrote: {out_path}")

    print("Done. Put --datadir=/data/mnist 给你的可执行程序即可。")

if __name__ == "__main__":
    main()
