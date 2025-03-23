import streamlit as st
import cv2
import numpy as np
import google.generativeai as genai
from PIL import Image
import time
from config import *
from collections import deque
import threading
from queue import Queue, Empty, Full

from prompt_utils import *

# Initialize Gemini API
genai.configure(api_key=GOOGLE_API_KEY)

# Global variables
camera = None
recognition_thread = None
frame_queue = Queue(maxsize=1)  # Limit queue size to 1, keep only the latest frame
result_queue = Queue()  # For passing recognition results between threads
is_running = True
result_lock = threading.Lock()  # For synchronizing access to recognition results
model = None
persistent_chat = None  # Persistent chat context
last_api_call_time = 0  # Record the time of last API call

def initialize_model():
    """Initialize model and persistent chat context"""
    global model, persistent_chat
    print("Initializing model...")  # Debug info
    model = genai.GenerativeModel('gemini-1.5-flash')
    
    # Create persistent chat context and set base prompt
    persistent_chat = model.start_chat(history=[])
    persistent_chat.send_message(BASE_PROMPT)
    print("Model and base context initialization complete")  # Debug info
    return model

def process_images(frames):
    """Process images and return recognition results"""
    try:
        global last_api_call_time, persistent_chat
        current_time = time.time()
        
        # Calculate time interval since last API call
        if last_api_call_time > 0:
            interval = current_time - last_api_call_time
            print(f"Time since last API call: {interval:.2f} seconds")
        
        # Update last API call time
        last_api_call_time = current_time
        
        # Use latest 6 frames
        latest_frames = frames[-6:] if len(frames) > 6 else frames
        
        # Adjust frame sizes for horizontal display of 6 frames
        resized_frames = []
        for frame in latest_frames:
            width = frame.shape[1] // 6  # Adjust width to 1/6 of original
            height = frame.shape[0]
            resized = cv2.resize(frame, (width, height))
            resized_frames.append(resized)
            
        pil_images = [Image.fromarray(frame) for frame in latest_frames]
        
        # Use persistent chat context but create new history each time
        response = persistent_chat.send_message(
            [DETAILED_PROMPT, *pil_images],
            generation_config=genai.types.GenerationConfig(
                candidate_count=1,
                temperature=0,  # Use deterministic output
                max_output_tokens=10,  # Limit output length
            )
        )
        
        # Clear this conversation's history but keep base prompt
        persistent_chat.history = persistent_chat.history[:1]  # Keep only first message (BASE_PROMPT)
        
        end_time = time.time()  # End timing
        print(f"Current API call duration: {end_time - current_time:.2f} seconds")  # Print duration
        
        # Return results and resized frames
        return response.text.strip(), resized_frames
    except Exception as e:
        print(f"API call error: {str(e)}")
        return "UNKNOWN", None

def recognition_worker():
    """Sign language recognition worker thread"""
    print("Recognition thread started")  # Debug info
    while is_running:
        try:
            # Get frames from queue with 1-second timeout
            try:
                frames = frame_queue.get(timeout=1)
                if frames is None:  # Exit signal
                    print("Received exit signal, recognition thread ending")  # Debug info
                    break
                
                print("Processing new frames")  # Debug info
                # Perform recognition
                result, used_frames = process_images(frames)
                print(f"Recognition result: {result}")  # Debug info
                
                # Update recognition results in thread-safe way
                with result_lock:
                    # Put results and frames into queue
                    result_queue.put((result, used_frames))
                    print(f"Put result {result} into result queue")  # Debug info
                
            except Empty:
                # Queue is empty, continue waiting
                continue
                
        except Exception as e:
            print(f"Recognition thread error: {str(e)}")  # Debug info
            continue

def start_recognition_thread():
    """Start recognition thread"""
    global recognition_thread
    if recognition_thread is None or not recognition_thread.is_alive():
        print("Starting recognition thread...")  # Debug info
        recognition_thread = threading.Thread(target=recognition_worker)
        recognition_thread.daemon = True
        recognition_thread.start()
        print("Recognition thread started")  # Debug info

def initialize_camera():
    """Initialize camera and model"""
    global camera, model
    if camera is None:
        print("Initializing camera...")  # Debug info
        camera = cv2.VideoCapture(0)
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
        camera.set(cv2.CAP_PROP_FPS, FRAME_RATE)
        print("Camera initialization complete")  # Debug info
        
        # Initialize model
        model = initialize_model()
        print("Model initialization complete")  # Debug info

def cleanup():
    """Clean up resources"""
    global camera, recognition_thread, is_running
    print("Starting resource cleanup...")  # Debug info
    is_running = False
    frame_queue.put(None)
    if recognition_thread:
        recognition_thread.join()
    if camera:
        camera.release()
    camera = None
    recognition_thread = None
    print("Resource cleanup complete")  # Debug info

# Initialize session state
if 'recognized_words' not in st.session_state:
    st.session_state.recognized_words = []
if 'last_gesture_time' not in st.session_state:
    st.session_state.last_gesture_time = 0
if 'current_sentence' not in st.session_state:
    st.session_state.current_sentence = ""
if 'frames' not in st.session_state:
    st.session_state.frames = deque(maxlen=10)  # Increase queue length to store more frames
if 'last_frame_time' not in st.session_state:
    st.session_state.last_frame_time = 0
if 'frame_count' not in st.session_state:
    st.session_state.frame_count = 0
if 'recognition_results' not in st.session_state:
    st.session_state.recognition_results = deque(maxlen=5)  # Store recognition results
if 'last_recognition_time' not in st.session_state:
    st.session_state.last_recognition_time = 0

def main():
    st.title("Real-time Sign Language Recognition System")
    st.write("Real-time display of camera feed and sign language recognition results")

    # Create two-column layout with adjusted width ratio
    col1, col2 = st.columns([1, 1.5])  # Increase right column width to accommodate more frames

    with col1:
        st.subheader("Live Camera Feed")
        camera_placeholder = st.empty()

    with col2:
        st.subheader("Frames Used for API Call")
        frames_container = st.container()
        with frames_container:
            frames_placeholder = st.empty()
            result_text = st.empty()
        st.subheader("Recognition Results")
        results_placeholder = st.empty()

    # Control buttons area
    button_col1, button_col2 = st.columns(2)
    with button_col1:
        if st.button("Clear Results", key="clear_frames_button"):
            print("Clear results button clicked")  # Debug info
            with result_lock:  # Use lock to protect clearing operation
                st.session_state.frames.clear()
                st.session_state.recognition_results.clear()
            # Clear frame queue and result queue
            while not frame_queue.empty():
                try:
                    frame_queue.get_nowait()
                except Empty:
                    break
            while not result_queue.empty():
                try:
                    result_queue.get_nowait()
                except Empty:
                    break
            frames_placeholder.empty()
            result_text.empty()
            results_placeholder.empty()
    
    with button_col2:
        if st.button("Exit", key="exit_button"):
            print("Exit button clicked")  # Debug info
            cleanup()

    # Initialize camera and start recognition thread
    initialize_camera()
    start_recognition_thread()

    # Track API call status
    api_call_in_progress = False
    
    # Main loop
    while is_running:
        ret, frame = camera.read()
        if not ret:
            st.error("Cannot read camera feed")
            break

        # Adjust frame size
        frame = cv2.resize(frame, (CAMERA_WIDTH, CAMERA_HEIGHT))
        
        # Update frames
        st.session_state.frames.append(frame)
        
        # Display live feed
        camera_placeholder.image(frame, channels="BGR")
        
        # Check for new recognition results
        try:
            result, used_frames = result_queue.get_nowait()
            api_call_in_progress = False  # Mark API call as complete
            
            with result_lock:
                # Display frame sequence
                if used_frames and len(used_frames) > 0:
                    # Create horizontally stacked frame sequence
                    combined_frame = np.hstack(used_frames)
                    frames_placeholder.image(combined_frame, channels="BGR", use_container_width=True)
                    result_text.markdown(f"**Recognition Result:** {result}")
                else:
                    frames_placeholder.write("Waiting for sufficient frames...")
                
                # Save result
                if result != "UNKNOWN":
                    st.session_state.recognition_results.append(result)
                    # Keep result list no longer than 5
                    while len(st.session_state.recognition_results) > 5:
                        st.session_state.recognition_results.popleft()
        except Empty:
            pass
        
        # Only send new frames when no API call is in progress
        if not api_call_in_progress and len(st.session_state.frames) >= 6:
            try:
                recent_frames = list(st.session_state.frames)[-6:]  # Get latest 6 frames
                frame_queue.put(recent_frames, timeout=0.1)
                api_call_in_progress = True  # Mark API call as started
                print("Sending new frames for recognition")  # Debug info
            except Full:
                print("Frame queue is full, skipping current frame")
        
        # Display historical recognition results
        if st.session_state.recognition_results:
            results_text = ["Historical Recognition Results:"]
            for i, result in enumerate(st.session_state.recognition_results):
                results_text.append(f"{i+1}. {result}")
            results_placeholder.markdown("\n".join(results_text))
        
        # Control frame rate
        time.sleep(0.0167)  # Approximately 60fps

if __name__ == "__main__":
    main() 