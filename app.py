from flask import Flask, Response
import cv2
from flask import render_template, send_file
import google.generativeai as genai
import os
from prompt_utils import BASE_PROMPT, DETAILED_PROMPT, return_instruction
from dotenv import load_dotenv
from text_to_speech import text_to_speech

# Load environment variables
load_dotenv()

app = Flask(__name__)
model = genai.GenerativeModel('gemini-1.5-flash')

def generate_frames():
    # You can replace this with your video source
    camera = cv2.VideoCapture(0)  # 0 for webcam
    
    while True:
        success, frame = camera.read()
        if not success:
            break
        
        # Encode the frame as JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        
        # Yield the frame in byte format
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/generate_and_play/<text>')
def generate_and_play(text):
    try:
        # Your text-to-speech code here
        text_to_speech(text, "audio/output.mp3")
        return render_template('index.html')
    except Exception as e:
        return str(e), 400
    
@app.route('/get_audio')
def get_audio():
    try:
        return send_file('audio/output.mp3', mimetype='audio/mpeg')
    except Exception as e:
        return str(e), 400

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
