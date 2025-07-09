import cv2
import json
import numpy as np
from shapely.geometry import Point, Polygon
from ultralytics import YOLO
import os
import streamlit as st
import tempfile
#from main_fixed import terminate

REGION_FILE = "regions.json"
MODEL_PATH = "yolo11n.pt"

def load_regions():
    with open(REGION_FILE, "r") as f:
        data = json.load(f)
    shapely_polys = [Polygon([(pt['x'], pt['y']) for pt in poly]) for poly in data['polygons']]
    raw_polys = data['polygons']
    return shapely_polys, raw_polys

def get_thresholds_streamlit():
    st.header("🔧 Threshold Settings")
    car_thresh = st.sidebar.number_input("Car Threshold", min_value=0, value=10)
    truck_thresh = st.sidebar.number_input("Truck Threshold", min_value=0, value=5)
    bus_thresh = st.sidebar.number_input("Bus Threshold", min_value=0, value=3)
    overall_thresh = st.sidebar.number_input("Overall Vehicle Threshold", min_value=0, value=20)
    return car_thresh, truck_thresh, bus_thresh, overall_thresh

def draw_stats_box(frame, car_count, truck_count, bus_count, car_thresh, truck_thresh, bus_thresh, overall_count, overall_thresh, status_text, status_color):
    box_x, box_y, box_w, box_h = 10, 10, 340, 160
    overlay = frame.copy()
    cv2.rectangle(overlay, (box_x, box_y), (box_x + box_w, box_y + box_h), (50, 50, 50), -1)
    frame = cv2.addWeighted(overlay, 0.4, frame, 0.6, 0)

    cv2.putText(frame, f"Cars: {car_count} / {car_thresh}", (box_x + 10, box_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    cv2.putText(frame, f"Trucks: {truck_count} / {truck_thresh}", (box_x + 10, box_y + 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    cv2.putText(frame, f"Buses: {bus_count} / {bus_thresh}", (box_x + 10, box_y + 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    cv2.putText(frame, f"Total: {overall_count} / {overall_thresh}", (box_x + 10, box_y + 105), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    cv2.putText(frame, f"Status: {status_text}", (box_x + 10, box_y + 140), cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
    return frame

def detect_vehicles_in_roi(video_path, scale=1.0, thresholds=(10, 5, 3, 20)):
    rois, raw_polygons = load_regions()
    model = YOLO(MODEL_PATH)
    car_thresh, truck_thresh, bus_thresh, overall_thresh = thresholds

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        st.error("Unable to open video.")
        return None

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) * scale)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) * scale)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_path = os.path.join(os.path.dirname(video_path), "output_result.mp4")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_skip = 4
    frame_counter = 0

    stframe = st.empty()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_counter += 1
        if frame_counter % frame_skip != 0:
            continue

        if scale != 1.0:
            frame = cv2.resize(frame, (width, height))

        results = model(frame)[0]
        boxes = results.boxes.xyxy.cpu().numpy()
        class_ids = results.boxes.cls.cpu().numpy().astype(int)

        car_count = truck_count = bus_count = 0
        vehicle_classes = {2: 'car', 3: 'motorcycle', 5: 'bus', 7: 'truck'}

        for box, cls_id in zip(boxes, class_ids):
            if cls_id not in vehicle_classes:
                continue
            x1, y1, x2, y2 = map(int, box)
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            pt = Point(cx, cy)
            if any(poly.contains(pt) for poly in rois):
                label = vehicle_classes[cls_id]
                if label == 'car':
                    car_count += 1
                elif label == 'truck':
                    truck_count += 1
                elif label == 'bus':
                    bus_count += 1

        overall_count = car_count + truck_count + bus_count
        status_color = (0, 255, 0)
        status_text = "Smooth"
        if (car_count > car_thresh or truck_count > truck_thresh or
            bus_count > bus_thresh or overall_count > overall_thresh):
            status_color = (0, 0, 255)
            status_text = "Congested"

        overlay = frame.copy()
        alpha = 0.2
        for poly in raw_polygons:
            pts = np.array([[pt['x'], pt['y']] for pt in poly], np.int32).reshape((-1, 1, 2))
            cv2.fillPoly(overlay, [pts], status_color)
        frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

        for poly in raw_polygons:
            pts = np.array([[pt['x'], pt['y']] for pt in poly], np.int32).reshape((-1, 1, 2))
            cv2.polylines(frame, [pts], True, status_color, 2)

        frame = draw_stats_box(frame, car_count, truck_count, bus_count,
                               car_thresh, truck_thresh, bus_thresh,
                               overall_count, overall_thresh,
                               status_text, status_color)

        out.write(frame)
        stframe.image(frame, channels="BGR")

    cap.release()
    out.release()
    return output_path
    while True:
        # ✅ Check if termination was requested
        if st.session_state.get("terminate", False):
            #terminate()
            st.warning("🚫 Detection manually terminated.")
            break


def main():
    st.title("🚦 Full Video ROI Detection with YOLOv8")

    video_file = st.file_uploader("Upload a video", type=["mp4", "avi", "mov", "mkv"])
    if not video_file:
        st.warning("Please upload a video to start.")
        return
    # if st.button("⏹ Terminate", key="terminate_button"):
    #     st.session_state["terminate"] = True
    # else:
    #     st.session_state["terminate"] = False

    #car_thresh, truck_thresh, bus_thresh, overall_thresh = get_thresholds_streamlit()

    if st.button("▶️ Run Full Detection"):
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(video_file.read())
        saved_path = detect_vehicles_in_roi(tfile.name, scale=1.0,
                                            thresholds=get_thresholds_streamlit())
        if saved_path:
            st.success(f"🎥 Detection complete! Download the result below:")
            st.video(saved_path)

if __name__ == "__main__":
    main()
