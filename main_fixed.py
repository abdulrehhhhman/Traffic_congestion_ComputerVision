# video_uploader.py

import streamlit as st
import os
#from roi_selector import main
#from vehicle_detection import get_thresholds_streamlit
SAVE_DIR = "uploaded_videos"
os.makedirs(SAVE_DIR, exist_ok=True)


def upload_and_save_video(widget_key="video_upload_main"):
    """
    Displays a Streamlit video uploader, saves the uploaded video to disk,
    and returns the path where the video was saved (or None if not uploaded).

    Returns:
        str or None: The path to the saved video file or None if no file was uploaded.
    """
    st.title("Video Upload and Save App")

    uploaded_video = st.file_uploader(
        "Upload a video file",
        key=widget_key,
        type=["mp4", "mpv", "mov", "avi", "mkv", "flv", "webm"],
        accept_multiple_files=False,

    )

    if uploaded_video is not None:
        st.success(f"Uploaded: {uploaded_video.name}")
        st.video(uploaded_video)

        save_path = os.path.join(SAVE_DIR, uploaded_video.name)

        with open(save_path, "wb") as f:
            f.write(uploaded_video.read())

        st.success(f"Saved video to: {save_path}")
        return save_path

    else:
        st.info("Please upload a video file.")
        return None




st.title("🚦 ROI Vehicle Detection App")




# Step 2: Thresholds


#car_thresh, truck_thresh, bus_thresh, overall_thresh = get_thresholds_streamlit()



def terminate():
    if st.button("⏹ Terminate", key="terminate_button"):
        st.session_state["terminate"] = True
    else:
        st.session_state["terminate"] = False


# Create a button that calls the imported function
#if st.button("Select/Draw ROI"):
#    st.success("hi")
