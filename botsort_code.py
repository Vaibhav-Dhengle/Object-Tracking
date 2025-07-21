import cv2
import numpy as np
from ultralytics import YOLO
from shapely.geometry import Point, Polygon
import yaml
import os
import csv


# ---------------------- CONFIGURATION ----------------------
video_path = "C:/Computer Vision/Internship Project/train/circular_2025-07-01_16-03.mp4"
model = YOLO("yolo11x.pt")
roi_yaml_path = "roi_config_botsort.yaml"
output_video_path = "botsort_output.mp4"
target_class = "person"

# ---------------------- ROI SELECTION ----------------------
roi_points = []
roi_dict = {}  # {roi_name: Polygon}


def click_event(event, x, y, flags, param):
    """Mouse click callback for ROI creation"""
    global roi_points
    if event == cv2.EVENT_LBUTTONDOWN:
        roi_points.append((x, y))
        print(f"Point added: {x},{y}")
    elif event == cv2.EVENT_RBUTTONDOWN:
        if len(roi_points) >= 3:
            name = input("Enter ROI name: ") or f"ROI{len(roi_dict) + 101}"
            roi_dict[name] = Polygon(roi_points)
            print(f"Polygon '{name}' saved: {roi_points}\n")
            roi_points.clear()
        else:
            print("Need at least 3 points!")


def interactive_roi_selection(frame):
    """Allows user to select ROIs and assign names"""
    global roi_points, roi_dict
    roi_points = []
    roi_dict = {}

    print("\nInstructions:")
    print("- Left Click: Add point")
    print("- Right Click: Finish ROI and enter name")
    print("- ENTER: Finish all selections\n")

    cv2.namedWindow("Select ROI - Press Enter when done")
    cv2.setMouseCallback("Select ROI - Press Enter when done", click_event)

    while True:
        temp_frame = frame.copy()

        # Draw current ROI being created
        for i, pt in enumerate(roi_points):
            cv2.circle(temp_frame, pt, 5, (0, 255, 0), -1)
            if i > 0:
                cv2.line(temp_frame, roi_points[i - 1], pt, (255, 0, 0), 2)
        if len(roi_points) > 2:
            cv2.line(temp_frame, roi_points[-1], roi_points[0], (0, 0, 255), 2)

        # Draw saved ROIs with names
        for name, poly in roi_dict.items():
            pts = np.array(poly.exterior.coords[:-1], np.int32).reshape((-1, 1, 2))
            cv2.polylines(temp_frame, [pts], True, (0, 255, 255), 2)
            cx, cy = np.mean(pts[:, 0, :], axis=0).astype(int)
            cv2.putText(temp_frame, name, (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        cv2.imshow("Select ROI - Press Enter when done", temp_frame)
        key = cv2.waitKey(1)
        if key == 13:  # ENTER
            break

    cv2.destroyAllWindows()
    return roi_dict


def save_rois_to_yaml(rois, path):
    """Save ROIs with names to YAML file"""
    yaml_dict = {"rois": {}}
    for name, poly in rois.items():
        coords_list = [[float(x), float(y)] for x, y in poly.exterior.coords[:-1]]
        yaml_dict["rois"][name] = coords_list

    with open(path, "w") as f:
        yaml.safe_dump(yaml_dict, f, default_flow_style=True, sort_keys=False, width=10_000)
    print(f"ROIs saved to {path}")


def load_rois_from_yaml(path):
    """Load ROIs from YAML file"""
    with open(path, "r") as f:
        data = yaml.safe_load(f) or {}
    roi_data = data.get("rois", {})
    return {name: Polygon(coords) for name, coords in roi_data.items()}


# ---------------------- ROI LOADING / CREATION ----------------------
cap = cv2.VideoCapture(video_path)
ret, frame = cap.read()
cap.release()

if not ret:
    print("Failed to read video")
    exit()

if os.path.exists(roi_yaml_path):
    rois = load_rois_from_yaml(roi_yaml_path)
    if not rois:
        print("YAML empty → Select ROIs manually")
        rois = interactive_roi_selection(frame)
        save_rois_to_yaml(rois, roi_yaml_path)
else:
    rois = interactive_roi_selection(frame)
    save_rois_to_yaml(rois, roi_yaml_path)

print(f"Loaded ROIs: {list(rois.keys())}")

# ---------------------- TRACKING ----------------------
object_log = {}  # {id: {"class":..., "roi":..., "sessions":[{enter_time, exit_time, duration}]}}
cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
frame_num = 0
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
print("Output video will be saved at:", os.path.abspath(output_video_path))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame_num += 1
    current_time = frame_num / fps  # in seconds

    results = model.track(frame, tracker="botsort.yaml", persist=True, verbose=False)
    boxes = results[0].boxes

    if boxes is not None:
        for box in boxes:
            cls_id = int(box.cls[0])
            class_name = model.names[cls_id]
            if class_name != target_class:
                continue

            obj_id = int(box.id[0]) if box.id is not None else -1
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            point = Point(cx, cy)

            # Determine if inside an ROI
            inside_roi_name = None
            for name, poly in rois.items():
                if poly.contains(point):
                    inside_roi_name = name
                    break

            if inside_roi_name:
                if obj_id not in object_log:
                    object_log[obj_id] = {
                        "class": class_name,
                        "sessions": [{"roi": inside_roi_name, "enter_time": current_time, "exit_time": None}]
                    }
                else:
                    sessions = object_log[obj_id]["sessions"]
                    last_session = sessions[-1]

                    if last_session["exit_time"] is None:
                        if last_session["roi"] != inside_roi_name:
                            # Close previous ROI session
                            last_session["exit_time"] = current_time
                            last_session["duration"] = round(current_time - last_session["enter_time"], 2)
                            # Start new ROI session
                            sessions.append({"roi": inside_roi_name, "enter_time": current_time, "exit_time": None})
                    else:
                        # Last session is closed, so start a new session (even if it's the same ROI)
                        sessions.append({"roi": inside_roi_name, "enter_time": current_time, "exit_time": None})
            else:
                # If person is not in any ROI but last session is open, close it
                if obj_id in object_log:
                    sessions = object_log[obj_id]["sessions"]
                    last_session = sessions[-1]
                    if last_session["exit_time"] is None:
                        last_session["exit_time"] = current_time
                        last_session["duration"] = round(current_time - last_session["enter_time"], 2)

            # Draw detection and ROI info
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"ID:{obj_id}"
            if inside_roi_name:
                label += f" {inside_roi_name}"
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    # Draw ROIs
    for name, poly in rois.items():
        pts = np.array(poly.exterior.coords[:-1], np.int32).reshape((-1, 1, 2))
        cv2.polylines(frame, [pts], True, (0, 0, 255), 2)
        cx, cy = np.mean(pts[:, 0, :], axis=0).astype(int)
        cv2.putText(frame, name, (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow("Tracking", frame)
    out.write(frame)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()




# ---------------------- BUILD & SAVE SUMMARY (ALL IN roi_config.yaml) ----------------------

# 1. Close any still-open sessions at video end (safety)
video_end_time = frame_num / fps if fps else None
for info in object_log.values():
    if info["sessions"]:
        last = info["sessions"][-1]
        if last["exit_time"] is None and video_end_time is not None:
            last["exit_time"] = video_end_time
            last["duration"] = round(video_end_time - last["enter_time"], 2)

# 2. Build per-object summary (all visits, ROI included)
objects_summary = {}
for obj_id, obj_info in object_log.items():
    obj_sessions = []
    for s in obj_info["sessions"]:
        enter = round(s["enter_time"], 2)
        if s["exit_time"] is not None:
            exit_t = round(s["exit_time"], 2)
            dur = round(
                s.get("duration", s["exit_time"] - s["enter_time"]),
                2,
            )
        else:
            exit_t = None
            dur = None
        obj_sessions.append(
            {
                "roi": s["roi"],
                "enter_time": enter,
                "exit_time": exit_t,
                "duration": dur,
            }
        )

    objects_summary[obj_id] = {
        "class": obj_info["class"],
        "sessions": obj_sessions,
    }

# 3. Build ROI-wise reverse index (optional, but helps you read by ROI)
roi_summary = {roi_name: [] for roi_name in rois.keys()}
for obj_id, obj_info in objects_summary.items():
    for sess in obj_info["sessions"]:
        roi_summary[sess["roi"]].append(
            {
                "person": f"P{obj_id}",
                "enter_time": sess["enter_time"],
                "exit_time": sess["exit_time"],
                "duration": sess["duration"],
            }
        )

# 4. Reconstruct ROIs block (coords) so we don't lose them
rois_block = {}
for name, poly in rois.items():
    coords_list = [[float(x), float(y)] for x, y in poly.exterior.coords[:-1]]
    rois_block[name] = coords_list

# 5. Combine everything in ONE YAML + write
yaml_out = {
    "rois": rois_block,
    "objects": objects_summary
}

with open(roi_yaml_path, "w") as f:
    yaml.safe_dump(yaml_out, f, sort_keys=False)

print(f"\n Tracking summary (ROIs + objects + roi_summary) saved to {roi_yaml_path}")




# ---------------------- EXPORT TO CSV (Stacked IN/OUT per ROI) ----------------------
csv_path = "tracking_summary_botsort.csv"
roi_names = sorted(rois.keys())

# Build header row
header = ["Person"]
for roi in roi_names:
    header += [f"{roi}_IN", f"{roi}_OUT"]

rows = []

for obj_id, info in object_log.items():
    # Collect visits per ROI
    roi_visits = {roi: [] for roi in roi_names}
    for sess in info["sessions"]:
        roi_visits[sess["roi"]].append((
            round(sess["enter_time"], 2),
            round(sess["exit_time"], 2) if sess["exit_time"] else ""
        ))

    # Determine maximum visits across all ROIs for this person
    max_visits = max(len(v) for v in roi_visits.values()) if roi_visits else 1

    # Add rows for each visit index
    for i in range(max_visits):
        row = [f"P{obj_id}" if i == 0 else ""]
        for roi in roi_names:
            if i < len(roi_visits[roi]):
                row += [roi_visits[roi][i][0], roi_visits[roi][i][1]]
            else:
                row += ["", ""]
        rows.append(row)

# Save CSV
import csv
with open(csv_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(header)
    writer.writerows(rows)

print(f" CSV saved at {csv_path} in grouped stacked IN/OUT format.")

