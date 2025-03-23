from flask import Flask, Response
import cv2
from flask import render_template

app = Flask(__name__)

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

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
