import os, sys
import json, time, cv2, pandas as pd
import numpy as np
from shapely.geometry import Polygon, box
from ultralytics import YOLO
from datetime import datetime
import threading
import queue

#change weight&settings path
base_path = os.path.dirname(os.path.abspath(sys.argv[0]))

config_path = os.path.join(base_path, "config_files")
cache_path = os.path.join(base_path, "weights")

os.makedirs(config_path, exist_ok=True)
os.makedirs(cache_path, exist_ok=True)

os.environ["ULTRALYTICS_CFG"] = config_path
os.environ["ULTRALYTICS_CACHE"] = cache_path
#change weight&settings path done

cfg = json.load(open("config.json", "r"))
model = YOLO("yolo11n.pt")

# names 映射，兼容不同 ultralytics 版本
try:
    names = model.names
except Exception:
    try:
        names = model.model.names
    except Exception:
        names = {}

#Capture setup
cap = cv2.VideoCapture(cfg["source"])
cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg["capwidth"])
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg["capheight"])
#Capture setup done

park_polys = [Polygon(b) for b in cfg["roi_park"]]
pass_polys = [Polygon(b) for b in cfg["roi_pass"]]

# IoU 阈值，便于统一修改
IOU_THRESH = 0.1

log_rows = []
t_last_log = time.time()
last_block_on = 0.0
last_outside_on = 0.0
blocked = False
outside = False

def iou_with_poly(xyxy, poly: Polygon):
    x1, y1, x2, y2 = xyxy
    det = box(float(x1), float(y1), float(x2), float(y2))
    inter = det.intersection(poly).area
    union = det.area + poly.area - inter
    return inter/union if union > 0 else 0.0

def flush_log():
    global log_rows
    if not log_rows: return
    df = pd.DataFrame(log_rows)
    try:
        df.to_csv("logs.csv", mode="a", header=not os.path.exists("logs.csv"), index=False, encoding="utf-8")
    except Exception:
        # fallback: rewrite complete file
        df.to_csv("logs.csv", index=False, encoding="utf-8")
    log_rows = []

os.makedirs("snaps", exist_ok=True)

# 线程化相关
FRAME_QUEUE_SIZE = cfg.get("queue_size", 4)
frame_queue = queue.Queue(maxsize=FRAME_QUEUE_SIZE)
result_queue = queue.Queue(maxsize=1)  # 只保留最新结果
stop_event = threading.Event()
sep_of_inference = 1.0 / float(cfg["inffps"])

def inference_worker():
    """
    推理线程：从 frame_queue 取帧，做模型推理，把序列化的检测结果放入 result_queue（覆盖旧结果）。
    线程会尽量按 cfg['inffps'] 限速（处理时间 + sleep）。
    """
    print("Inference worker started")
    while not stop_event.is_set():
        try:
            ts, frame = frame_queue.get(timeout=0.5)
        except queue.Empty:
            continue
        start = time.time()
        try:
            res = model(frame, imgsz=((cfg["capwidth"]//32*32),(cfg["capheight"]//32*32)), conf=cfg["conf"], verbose=False)[0]
        except Exception as e:
            print("Inference error:", e)
            continue

        boxes = []
        for b in res.boxes:
            try:
                cls = int(b.cls[0])
                conf = float(b.conf[0]) if hasattr(b, "conf") else None
                xyxy = b.xyxy[0].cpu().numpy().tolist()
                boxes.append({"xyxy": xyxy, "cls": cls, "conf": conf})
            except Exception:
                continue

        result = {"time": time.time(), "boxes": boxes}

        # 写入 result_queue，若满则替换旧结果（保留最新）
        try:
            result_queue.put_nowait(result)
        except queue.Full:
            try:
                _ = result_queue.get_nowait()
            except queue.Empty:
                pass
            try:
                result_queue.put_nowait(result)
            except queue.Full:
                pass

        elapsed = time.time() - start
        to_sleep = sep_of_inference - elapsed
        if to_sleep > 0:
            time.sleep(to_sleep)
    print("Inference worker stopped")

inf_thread = threading.Thread(target=inference_worker, daemon=True)
inf_thread.start()

print("Running... Press q to stop.")
print(f'Using Res: {cfg["capwidth"]//32*32} by {cfg["capheight"]//32*32} at {cfg["inffps"]}fps')
if cfg["capwidth"]%32!=0 or cfg["capheight"]%32!=0:
    print("[WARN] Res must be the multiplier of 32!")

lag_counts = 0
max_lag_num = cfg["lagtolerance"]
fps_count = 0

try:
    while True:
        ok, frame = cap.read()
        if not ok:
            time.sleep(0.1)
            continue

        # 生产者：非阻塞入队（队列满则丢帧）
        try:
            frame_queue.put_nowait((time.time(), frame.copy()))
        except queue.Full:
            lag_counts += 1
            if lag_counts >= max_lag_num:
                print(f"{lag_counts} Lagged frames completed at {datetime.now()}")

        # 从 result_queue 取最新的推理结果（若有）
        latest_result = None
        try:
            latest_result = result_queue.get_nowait()
        except queue.Empty:
            latest_result = None

        # 使用最新推理结果更新计数与事件逻辑
        in_park = in_pass = out_of_park = 0
        if latest_result:
            for det in latest_result["boxes"]:
                cls = det["cls"]
                # 获取类别名（兼容 dict/list）
                try:
                    name = names[cls]
                except Exception:
                    name = str(cls)
                if name not in cfg["classes"]:
                    continue
                xyxy = det["xyxy"]
                is_in_park = any(iou_with_poly(xyxy, poly) > IOU_THRESH for poly in park_polys)
                is_in_pass = any(iou_with_poly(xyxy, poly) > IOU_THRESH for poly in pass_polys)

                if is_in_park:
                    in_park += 1
                else:
                    out_of_park += 1
                if is_in_pass:
                    in_pass += 1

        now = time.time()
        # Blocked
        if in_pass > 0:
            if not blocked:
                last_block_on = now
                blocked = True
        else:
            blocked = False
        # Not_in_place
        if out_of_park > 0 and in_park == 0:
            if not outside:
                last_outside_on = now
                outside = True
        else:
            outside = False

        events = []
        if blocked and now - last_block_on >= cfg["block_secs"]:
            events.append("block")
            last_block_on = now
            cv2.imwrite(f"snaps/block_{int(now)}.jpg", frame)
        if outside and now - last_outside_on >= cfg["block_secs"]:
            events.append("outside")
            last_outside_on = now
            cv2.imwrite(f"snaps/outside_{int(now)}.jpg", frame)

        if now - t_last_log >= cfg["log_interval"]:
            ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            log_rows.append({"time": ts, "in_park": in_park, "in_pass": in_pass, "events": "|".join(events)})
            flush_log()
            t_last_log = now

        # 预览：绘制 ROI 和最新检测框（如果有）
        if cfg.get("preview", False):
            for bpark in cfg["roi_park"]:
                cv2.polylines(frame, [np.array(bpark, np.int32)], True, (0,255,0), 1)
            for bpass in cfg["roi_pass"]:
                cv2.polylines(frame, [np.array(bpass, np.int32)], True, (0,0,255), 1)

            if latest_result:
                for det in latest_result["boxes"]:
                    cls = det["cls"]
                    try:
                        name = names[cls]
                    except Exception:
                        name = str(cls)
                    if name not in cfg["classes"]:
                        continue
                    xyxy = det["xyxy"]
                    x1, y1, x2, y2 = map(int, xyxy)
                    h, w = frame.shape[:2]
                    x1 = max(0, min(w-1, x1))
                    x2 = max(0, min(w-1, x2))
                    y1 = max(0, min(h-1, y1))
                    y2 = max(0, min(h-1, y2))
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255,0,0), 1)

            cv2.imshow("bikewatch", frame)
            if cv2.waitKey(1) in (113,81,27):  # q/Q/Esc
                break

except KeyboardInterrupt:
    pass
finally:
    # 停止推理线程并清理
    stop_event.set()
    inf_thread.join(timeout=2)
    cap.release()
    cv2.destroyAllWindows()
    flush_log()