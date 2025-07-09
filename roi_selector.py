import cv2
import json
import numpy as np
import tkinter as tk
from tkinter import filedialog
import os
from main_fixed import upload_and_save_video
import streamlit as st
from vehicle_detection import detect_vehicles_in_roi, get_thresholds_streamlit

REGION_FILE = "regions.json"
tmp_points = []
polygons = []
scale = 1.0  # scale factor


# Save polygons and scale to JSON
def save_regions():
    data = {
        "scale": scale,
        "polygons": [
            [{"x": int(x), "y": int(y)} for x, y in poly]
            for poly in polygons
        ]
    }
    with open(REGION_FILE, "w") as f:
        json.dump(data, f, indent=2)
    print(f"[+] Saved {len(polygons)} region(s) to {REGION_FILE} with scale={scale}")


# Draw saved and in-progress polygons
def draw_overlays(frame):
    for poly in polygons:
        pts = np.array(poly, np.int32).reshape((-1, 1, 2))
        cv2.polylines(frame, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
    if tmp_points:
        pts = np.array(tmp_points, np.int32).reshape((-1, 1, 2))
        cv2.polylines(frame, [pts], isClosed=False, color=(0, 0, 255), thickness=1)
        for x, y in tmp_points:
            cv2.circle(frame, (x, y), radius=3, color=(0, 0, 255), thickness=-1)


# Mouse callback
def mouse_callback(event, x, y, flags, param):
    global tmp_points, polygons
    if event == cv2.EVENT_LBUTTONDOWN:
        tmp_points.append((x, y))
    elif event == cv2.EVENT_RBUTTONDOWN:
        if len(tmp_points) >= 3:
            polygons.append(tmp_points.copy())
            print(f"[+] Region #{len(polygons)} saved ({len(tmp_points)} points)")
        tmp_points = []


def draw_roi_regions(VIDEO_PATH):
    """Function to handle ROI drawing"""
    global scale
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print("ERROR: Could not open video.")
        return False

    ret, frame = cap.read()
    cap.release()
    if not ret:
        print("ERROR: Could not read video frame.")
        return False

    # Determine scaling if video is too large
    max_width = 1280
    max_height = 720
    height, width = frame.shape[:2]
    scale_x = max_width / width
    scale_y = max_height / height
    scale = min(1.0, scale_x, scale_y)

    if scale < 1.0:
        frame = cv2.resize(frame, (int(width * scale), int(height * scale)))

    window_name = "Draw Regions (L-click add, R-click finish, U=undo, S=save, Q=quit)"
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, mouse_callback)

    while True:
        display_frame = frame.copy()
        draw_overlays(display_frame)
        cv2.imshow(window_name, display_frame)
        key = cv2.waitKey(10) & 0xFF

        if key == ord('u'):
            if tmp_points:
                tmp_points.pop()
        elif key == ord('s'):
            if tmp_points and len(tmp_points) >= 3:
                polygons.append(tmp_points.copy())
            save_regions()
            cv2.destroyAllWindows()
            return True  # ROI saved successfully
        elif key == ord('q'):
            print("Exiting without saving.")
            cv2.destroyAllWindows()
            return False

    return False


# Main flow
def main():
    global scale

    # Step 1: Upload video
    VIDEO_PATH = upload_and_save_video()

    if VIDEO_PATH is None:
        return

    # Step 2: Draw ROI regions
    if st.button("Select/Draw ROI"):
        # Initialize session state for ROI completion
        if 'roi_completed' not in st.session_state:
            st.session_state.roi_completed = False

        # Draw ROI in separate window
        roi_success = draw_roi_regions(VIDEO_PATH)

        if roi_success:
            st.session_state.roi_completed = True
            st.success("✅ ROI regions saved successfully!")
        else:
            st.error("❌ ROI selection cancelled or failed.")

    # Step 3: Get thresholds (only show if ROI is completed)
    if st.session_state.get('roi_completed', False):
        st.markdown("---")
        st.markdown("### Step 2: Set Detection Thresholds")

        # Get thresholds from user
        car_thresh, truck_thresh, bus_thresh, overall_thresh = get_thresholds_streamlit(key_prefix='main_detection')

        # Step 4: Start detection
        if st.button("🚀 Start Vehicle Detection"):
            st.info("Starting vehicle detection... This may take a while.")

            # Store thresholds in session state to pass to detection function
            st.session_state.thresholds = {
                'car_thresh': car_thresh,
                'truck_thresh': truck_thresh,
                'bus_thresh': bus_thresh,
                'overall_thresh': overall_thresh
            }

            # Call detection function
            detect_vehicles_in_roi(VIDEO_PATH, scale, st.session_state.thresholds)
            st.success("✅ Detection completed! Check the output video file.")


if __name__ == '__main__':
    main()