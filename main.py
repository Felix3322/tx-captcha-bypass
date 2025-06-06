#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os
import cv2
import numpy as np

TOP_CUT = 0.18  # 上分隔
BOT_CUT = 0.15  # 下分割
SV_TH = (70, 70)  # 背景灰度参数
CLICK_N = 4  # 一般是4，被风控了就不知道了


# ──────────────────────────────────────────────────────────────

def high_sv_mask(img, sv_thresh):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    s, v = hsv[:, :, 1], hsv[:, :, 2]
    mask = (s > sv_thresh[0]) & (v > sv_thresh[1])
    return mask.astype("uint8") * 255


def merge_boxes(boxes, gap=8):
    """合并相互靠近/重叠的 bbox，返回合并后 bbox 列表"""
    merged = []
    for x, y, w, h in sorted(boxes, key=lambda b: b[0]):
        if not merged:
            merged.append([x, y, w, h])
            continue
        mx, my, mw, mh = merged[-1]
        # 分字
        if x <= mx + mw + gap and y <= my + mh + gap and my <= y + h + gap:
            # merge
            nx = min(mx, x)
            ny = min(my, y)
            nw = max(mx + mw, x + w) - nx
            nh = max(my + mh, y + h) - ny
            merged[-1] = [nx, ny, nw, nh]
        else:
            merged.append([x, y, w, h])
    return merged


def recognize_digit(roi, num_range=range(1, 5)):
    """简单模板匹配识别数字，返回识别到的数字或 None"""
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (30, 30))
    _, gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    best = None
    best_score = -1
    for n in num_range:
        tmpl = np.zeros((30, 30), dtype=np.uint8)
        cv2.putText(tmpl, str(n), (2, 27), cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 2, cv2.LINE_AA)
        res = cv2.matchTemplate(gray, tmpl, cv2.TM_CCOEFF_NORMED)
        _, score, _, _ = cv2.minMaxLoc(res)
        if score > best_score:
            best_score = score
            best = n
    if best_score < 0.2:  # 阈值太低认为没识别出来
        return None
    return best


def detect_click_points(img):
    h, w = img.shape[:2]
    # 切割
    y0 = int(h * TOP_CUT)
    y1 = h - int(h * BOT_CUT)
    main = img[y0:y1, :]

    # sv掩码
    mask = high_sv_mask(main, SV_TH)

    # 填洞 连通
    ker = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, ker, 2)
    mask = cv2.dilate(mask, ker, 1)

    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for c in cnts:
        x, y, wc, hc = cv2.boundingRect(c)
        if 300 < wc * hc < 8000:  # 过滤大小极端的块
            boxes.append([x, y, wc, hc])

    if len(boxes) < CLICK_N:
        raise RuntimeError(f"检测到 {len(boxes)} 个候选字符，调小 SV_TH 或放宽面积")

    # 合并被切开的笔画
    boxes = merge_boxes(boxes)

    if len(boxes) < CLICK_N:
        raise RuntimeError("合并后不足所需数量，调 gap 或面积阈值")

    # 按面积排序取前 N
    boxes = sorted(boxes, key=lambda b: b[2] * b[3], reverse=True)[:CLICK_N]

    # 根据数字识别结果排序，识别失败则按行列排序
    ordered = []
    for bx in boxes:
        x, y, wc, hc = bx
        roi = main[y:y + hc, x:x + wc]
        d = recognize_digit(roi)
        ordered.append((d if d is not None else float('inf'), bx))

    if all(d is not None for d, _ in ordered):
        ordered.sort(key=lambda t: t[0])
        boxes = [b for _, b in ordered]
    else:
        boxes = sorted(boxes, key=lambda b: (b[1], b[0]))

    # 7. 计算质心
    points = []
    for x, y, wc, hc in boxes:
        points.append((x + wc // 2, y + hc // 2 + y0))
    return points, mask, boxes, y0


def main():
    if len(sys.argv) != 2:
        print("用法: python solve_click_captcha.py <captcha.png>")
        sys.exit(1)

    path = sys.argv[1]
    img = cv2.imread(path)
    if img is None:
        print("无法读取图片:", path);
        sys.exit(1)

    pts, msk, bbs, off = detect_click_points(img)

    # 输出
    print("点击顺序坐标:")
    for i, (x, y) in enumerate(pts, 1):
        print(f"{i}: ({x}, {y})")

    # 可视化
    vis = img.copy()
    for i, (x, y) in enumerate(pts, 1):
        cv2.circle(vis, (x, y), 6, (0, 0, 255), -1)
        cv2.putText(vis, str(i), (x - 12, y - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    if os.environ.get("DISPLAY"):
        cv2.imshow("result", vis)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

