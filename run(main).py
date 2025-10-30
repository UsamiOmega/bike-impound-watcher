import os,sys
import json, time, cv2, pandas as pd
import numpy as np
from shapely.geometry import Polygon, box
from ultralytics import YOLO
from datetime import datetime
import torch
import signal
#change weight&settings path
base_path = os.path.dirname(os.path.abspath(sys.argv[0]))

config_path = os.path.join(base_path, "config_files")
cache_path = os.path.join(base_path, "weights")
os.makedirs("snaps", exist_ok=True)

os.makedirs(config_path, exist_ok=True)
os.makedirs(cache_path, exist_ok=True)

os.environ["ULTRALYTICS_CFG"] = config_path
os.environ["ULTRALYTICS_CACHE"] = cache_path
#change weight&settings path done

cfg = json.load(open("config.json","r"))            #Load settings

#Model setup
model = YOLO(cfg.get("model","yolo11n.pt"))
device = 'cuda' if (cfg.get("usecuda", False) and torch.cuda.is_available()) else 'cpu'
if cfg.get("usecuda", False) and (not torch.cuda.is_available()):
    print("[WARN] Device not supporting CUDA!")
#Model setup done
#Capture setup

cap = cv2.VideoCapture(cfg["source"])
cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg["capwidth"])
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg["capheight"])
#fourcc = cfg["fourcc"]
#cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'{}'.format(fourcc)))


#Capture setup done


#Capture exit signals start
def signal_handler(sig, frame):
    print("[INFO] Received interrupt signal, exiting...")
    cap.release()
    cv2.destroyAllWindows()
    flush_log()  # 确保退出前保存日志
    sys.exit(0)
#Capture exit signals done


signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

park_polys = [Polygon(b) for b in cfg["roi_park"]]
pass_polys = [Polygon(b) for b in cfg["roi_pass"]]

# IoU 阈值，便于统一修改
IOU_thresh = cfg.get("IOU",0.1)

log_rows = []
t_last_log = time.time()
last_block_on = 0.0
last_outside_on = 0.0
blocked = False
outside = False

def iou_with_poly(xyxy, poly: Polygon):
    x1,y1,x2,y2 = xyxy
    det = box(float(x1), float(y1), float(x2), float(y2))
    inter = det.intersection(poly).area
    union = det.area + poly.area - inter
    return inter/union if union>0 else 0.0

def flush_log():
    global log_rows
    if not log_rows: return
    df = pd.DataFrame(log_rows)
    try:
        df.to_csv("logs.csv", mode="a", header=not os.path.exists("logs.csv"), index=False, encoding="utf-8")
    except Exception:
        # fallback: rewrite complete file
        try:
            df.to_csv("logs.csv", index=False, encoding="utf-8")
        except PermissionError:
            print("[ERROR] logs.csv unwrittable!")
    log_rows = []




print("[INFO] Running...",end = "")
if cfg.get("preview",False): print(" Press q to stop.") 
else: print(" Press Ctrl+C to stop.")
print(f'[INFO] Using Res: {cfg["capwidth"]//32*32} by {cfg["capheight"]//32*32} at {cfg["inffps"]}fps')
if cfg["capwidth"]%32!=0 or cfg["capheight"]%32!=0:
    print("[WARN] Res must be a multiple of 32!")
last_inference_time = 0.0
sep_of_inference = 1 / float(cfg["inffps"])
lag_counts = 0
max_lag_num = cfg["lagtolerance"]
fps_count = 0
while True:
    current_time = time.time()
    if current_time - last_inference_time < sep_of_inference:
        if lag_counts >= max_lag_num: print(f"[INFO] {lag_counts} Lagged frames completed at {datetime.now()}")
        lag_counts = 0
        time.sleep(sep_of_inference-(current_time - last_inference_time)); continue;
    else:
        lag_counts += 1
    ok, frame = cap.read()
    if not ok:
        print("[ERROR] Failed to capture frame!")
        time.sleep(0.1); continue
    
    res = model(frame, imgsz=((cfg["capwidth"]//32*32),(cfg["capheight"]//32*32)), conf=cfg["conf"], verbose=False, device=device)[0]
    
    if lag_counts >= max_lag_num:
        print(f"[WARN] Overload! {lag_counts}/{max_lag_num} frames queued!!!curfps at {1/(current_time - last_inference_time):.1f}/{cfg['inffps']:.1f}")
    if fps_count < 5*cfg["inffps"]:
        fps_count += 1
    elif lag_counts < max_lag_num:
        fps_count = 0
        print(f"[INFO] curfps at {1/(current_time - last_inference_time):.1f}/{cfg['inffps']:.1f}")
    else:
        fps_count = 0
    
    last_inference_time = current_time
    in_park = in_pass = out_of_park = 0

    # ROI-impound Check
    for b in res.boxes:
        try:
            cls = int(b.cls[0])
            name = res.names[cls]
        except Exception:
            continue
        if name not in cfg["classes"]:
            continue
        xyxy = b.xyxy[0].cpu().numpy()

        # Check if in "park"
        is_in_park = any(iou_with_poly(xyxy, poly) > IOU_thresh for poly in park_polys)
        # Check if in "pass"
        is_in_pass = any(iou_with_poly(xyxy, poly) > IOU_thresh for poly in pass_polys)

        if is_in_park:
            in_park += 1
        else:
            out_of_park += 1

        if is_in_pass:
            in_pass += 1

    now = time.time()
    # Blocked
    if in_pass > 0:
        if not blocked: last_block_on = now; blocked = True
    else:
        blocked = False
    # Not_in_place
    if out_of_park > 0 and in_park == 0:
        if not outside: last_outside_on = now; outside = True
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

    if cfg.get("preview", False):
        #Mark identification range
        for bpark in cfg["roi_park"]:
            cv2.polylines(frame, [np.array(bpark,np.int32)], True, (0,255,0), 1)
        for bpass in cfg["roi_pass"]:
            cv2.polylines(frame, [np.array(bpass,np.int32)], True, (0,0,255), 1)
        #Mark identified bikes
        for bimg in res.boxes:
            try:
                clas = int(bimg.cls[0])
                name = res.names[clas]
                #print(name)
            except Exception:
                continue
            try:
                confb = float(bimg.conf[0])
            except Exception:
                confb = 99.99
            if name not in cfg["classes"]:
                pass
                #continue
            
            xyxy = bimg.xyxy[0].cpu().numpy()
            x1, y1, x2, y2 = map(int, xyxy)
            h, w = frame.shape[:2]
            x1 = max(0, min(w-1, x1))
            x2 = max(0, min(w-1, x2))
            y1 = max(0, min(h-1, y1))
            y2 = max(0, min(h-1, y2))
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255,0,0), 1)
            cv2.putText(frame, f"{name}: {confb:.2f}", (x1,y1+12),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,0),2)
        cv2.imshow("bikewatch", frame)
        if cv2.waitKey(1) in (113,81,27):             #ord("q")==113,ord("Q")==81,ord(Esc)==27
            print("[INFO] Received interrupt signal, exiting...")
            break

cap.release()
cv2.destroyAllWindows()
flush_log()
sys.exit(0)