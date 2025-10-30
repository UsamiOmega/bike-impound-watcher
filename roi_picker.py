
# roi_picker.py
# 使用方法：
# 1) 先运行摄像头预览并按 's' 键保存一张截图：  python3 roi_picker.py --capture
# 2) 标定 ROI：                                     python3 roi_picker.py --mark frame.jpg
#    - 先标定“停车框 ROI”：依次点击四个角（顺时针/逆时针皆可），回退按 'z'，完成按 'ENTER'
#    - 接着标定“通道 ROI”：同样点击四个角；完成按 'ENTER'
# 3) 保存后会自动写入 config.json 中的 roi_park / roi_pass

import argparse, json, os, cv2, sys
from time import time,sleep

def load_config():
    cfg_path = "config.json"
    if not os.path.exists(cfg_path):
        print("未找到 config.json，请先放在当前目录"); sys.exit(1)
    with open(cfg_path, "r", encoding="utf-8") as f:
        return json.load(f), cfg_path

def live_capture(cfg):
    cap = cv2.VideoCapture(cfg.get("source", 0))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg["capwidth"])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg["capheight"])
    cap.set(cv2.CAP_PROP_FPS, cfg["capfps"])
    if not cap.isOpened():
        print("无法打开摄像头（source=%s）。请检查 config.json 的 source 值。" % str(cfg.get("source",0)))
        sys.exit(1)
    print("按 's' 保存当前帧为 frame.jpg，按 'q' 退出")
    last_cap_time = 0.0
    sep_of_cap = 1 / cfg["capfps"]
    
    while True:
        cur_time = time()
        if cur_time - last_cap_time < sep_of_cap:
            sleep(sep_of_cap); continue
        else:
            last_cap_time = cur_time
        ok, frame = cap.read()
        if not ok: continue
        cv2.imshow("preview (press 's' to save, 'q' to quit)", frame)
        k = cv2.waitKey(1) & 0xFF
        if k == ord('s'):
            cv2.imwrite("frame.jpg", frame)
            print("已保存 frame.jpg")
        elif k == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

def mark_polygon(img, title):
    cfg = load_config()[0]
    pts = []
    def on_mouse(event, x, y, flags, param):
        nonlocal pts, img_show
        
        if event == cv2.EVENT_LBUTTONDOWN:
            pts.append((x,y))
        elif event == cv2.EVENT_RBUTTONDOWN:
            if pts: pts.pop()
        # redraw
        img_show = img.copy()
        for i,p in enumerate(pts):
            cv2.circle(img_show, p, 4, (0,255,0), -1)
            if i>0: cv2.line(img_show, pts[i-1], p, (0,255,0), 2)
        if len(pts)>=2:
            cv2.line(img_show, pts[-1], pts[0], (0,255,0), 1)
        cv2.imshow(title, img_show)

    img_show = img.copy()
    cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    size_of_window = max(1024,cfg["capwidth"])
    cv2.resizeWindow(title, size_of_window, int(size_of_window/cfg["capwidth"]*cfg["capheight"]))
    cv2.setMouseCallback(title, on_mouse)
    print("→ %s：左键添加点，右键撤销，按 ENTER 完成，ESC 退出" % title)
    while True:
        cv2.imshow(title, img_show)
        k = cv2.waitKey(20) & 0xFF
        if k == 13:  # Enter
            if len(pts) < 3:
                print("至少需要3个点构成多边形")
                continue
            break
        elif k == 27:  # ESC
            pts = []
            break
    cv2.destroyWindow(title)
    return pts

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture", action="store_true", help="打开摄像头预览并保存截图")
    parser.add_argument("--mark", type=str, help="对指定图片进行ROI标定（如 frame.jpg）")
    args = parser.parse_args()

    cfg, cfg_path = load_config()

    if args.capture:
        live_capture(cfg)
        return

    if args.mark:
        if not os.path.exists(args.mark):
            print("未找到图片：", args.mark); sys.exit(1)
        img = cv2.imread(args.mark)
        if img is None:
            print("读取图片失败"); sys.exit(1)
        else:
            img = cv2.resize(img, (cfg["capwidth"], cfg["capheight"]))
        roi_parks,roi_passes,t = [],[],0
        while True:
            roi_park = mark_polygon(img, "Mark Parking-lot ROI:")
            if not roi_park:
                print("结束停车点标定");break
            else:
                roi_parks.append(roi_park)
        if roi_parks == []:
            print("未标定，不进行保存")
            sys.exit(0)
        while True:
            roi_pass = mark_polygon(img, "Mark Passway ROI:")
            if not roi_pass:
                print("结束过道标定");break
            else:
                roi_passes.append(roi_pass)
        if roi_passes == []:
            print("未标定，不进行保存")
            sys.exit(0)
        cfg["roi_park"] = roi_parks
        cfg["roi_pass"] = roi_passes

        with open(cfg_path, "w", encoding="utf-8") as f:
            json.dump(cfg, f, ensure_ascii=False, indent=2)
        print("已写入 config.json：roi_park / roi_pass")
        return

    print("未指定参数。示例：\n",
          "  python3 roi_picker.py --capture   # 预览并保存截图\n",
          "  python3 roi_picker.py --mark frame.jpg  # 在截图上标定数个ROI")

if __name__ == "__main__":
    main()
