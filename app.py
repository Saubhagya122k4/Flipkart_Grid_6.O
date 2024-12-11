from flask import Flask, render_template, send_from_directory
from flask_socketio import SocketIO
import cv2
import google.generativeai as genai
from PIL import Image
import base64
import numpy as np
import threading
import time
import os

app = Flask(__name__, template_folder=os.path.abspath('templates'))
socketio = SocketIO(app, cors_allowed_origins="*")

# Configure the Gemini API
import os
genai.configure(api_key=os.getenv('GENAI_API_KEY')) # Replace with your API key

# Initialize the Gemini model
model = genai.GenerativeModel('gemini-1.5-flash')

analyzing = False
capture_thread = None

def analyze_image(image, option):
    try:
        if option == 'brand':
            prompt = """Please perform the following tasks on this image:
            1. Brand Recognition: Identify any visible brand names or logos.
            
            Provide your response in the following format:
            Brand: [identified brand name]
            
            If no brand is detected, state 'None detected'."""
        
        elif option == 'expiry_mrp':
            prompt = """Please perform OCR on this image and extract the following information:
            1. MRP (Maximum Retail Price)
            2. Expiry Date or Best Before Date

            Provide your response in the following format:
            MRP: [extracted MRP]
            Date: [extracted expiry date or best before date]

            If any information is not found, state 'Not found' for that category."""
        
        elif option == 'auto_capture':
            prompt =""" Please perform OCR on this image and extract the following information:
            1. Brand details (extract all text from the label or package, not just keywords)
            2. Pack Size (extract pack size or weight explicitly, if mentioned)
            3. Brand name

            Provide your response in the following format:
            Brand details: [extracted brand details]
            Pack Size: [extracted pack size or weight]
            Brand name: [extracted brand name]

            If any information is not found, state 'Not found' for that category."""

        elif option == 'fruit_freshness':  # Fruit freshness detection prompt
            prompt = """Analyze this image and determine if it contains a fruit or vegetables. If a fruit or vegetables is present, estimate its freshness on a scale of 0% (completely rotten) to 100% (perfectly fresh).

            Provide your response in the following format:
            Fruit or Vegetable Detected: [Yes/No]
            Fruit or Vegetable Type: [Name of the fruit or vegetable, if detected]
            Freshness: [Percentage]

            If no fruit or vegetable is detected, state 'No fruit or vegetable detected' for all categories."""
        
        # Generate a response from the model based on the prompt
        response = model.generate_content([prompt, image])
        return response.text

    except genai.types.generation_types.BlockedPromptException as e:
        # Handle API rate limit exceptions
        if option == 'brand':
            return "Brand: Not detected"
        elif option == 'expiry_mrp':
            return "MRP: Not found\nDate: Not found"
        elif option == 'auto_capture':
            return "Brand details: Not found\nPack Size: Not found\nBrand name: Not found"
        elif option == 'fruit_freshness':
            return "Fruit or Vegetable Detected: No\nFruit or Vegetable Type: Not found\nFreshness: 0%\nExplanation: Rate limit exceeded"

    except Exception as e:
        # Check if the error matches the specific quota error
        error_message = str(e)
        if "429 Resource has been exhausted" in error_message:
            if option == 'brand':
                return "Brand: Not detected"
            elif option == 'expiry_mrp':
                return "MRP: Not found\nDate: Not found"
            elif option == 'auto_capture':
                return "Brand details: Not found\nPack Size: Not found\nBrand name: Not found"
            elif option == 'fruit_freshness':
                return "Fruit or Vegetable Detected: No\nFruit or Vegetable Type: Not found\nFreshness: 0%\nExplanation: Rate limit exceeded"
        else:
            return f"An error occurred during image analysis: {error_message}"

def continuous_analysis():
    global analyzing
    while analyzing:
        socketio.emit('request_frame')
        time.sleep(5)  # Analyze every 5 seconds

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/brand_recognition')
def brand_recognition():
    return render_template('brand_recognition.html')

@app.route('/expiry_mrp')
def expiry_mrp():
    return render_template('expiry_and_mrp.html')

@app.route('/branddetails_and_packsize')
def branddetails_and_packsize():
    return render_template('branddetails_and_packsize.html')

@app.route('/fruit_freshness')
def fruit_freshness():
    return render_template('freshness_detetction.html')

@app.route('/<path:filename>')
def serve_static(filename):
    return send_from_directory(app.template_folder, filename)

@socketio.on('connect')
def handle_connect():
    print('Client connected')

@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected')
    global analyzing
    analyzing = False

@socketio.on('start_analysis')
def handle_start_analysis(data):
    global analyzing, capture_thread
    analyzing = True
    capture_thread = threading.Thread(target=continuous_analysis)
    capture_thread.daemon = True
    capture_thread.start()
    print(f"Analysis started with option: {data.get('option')}")

@socketio.on('stop_analysis')
def handle_stop_analysis():
    global analyzing
    analyzing = False
    print("Analysis stopped")

@socketio.on('frame')
def handle_frame(data):
    try:
        # Extract the base64 image data
        image_data = data['image'].split(',')[1]
        image_bytes = base64.b64decode(image_data)
        
        # Convert to numpy array
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Convert to PIL Image
        pil_image = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        
        # Analyze the image based on selected option
        analysis_result = analyze_image(pil_image, data['option'])
        
        # Send results back to client
        socketio.emit('analysis_result', {'result': analysis_result})
        
    except Exception as e:
        print(f"Error processing frame: {str(e)}")
        socketio.emit('analysis_result', {'error': str(e)})

if __name__ == '__main__':
    print(f"Templates directory: {app.template_folder}")
    socketio.run(app, debug=True, allow_unsafe_werkzeug=True)
