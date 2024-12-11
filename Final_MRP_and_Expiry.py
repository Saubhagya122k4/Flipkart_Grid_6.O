import cv2
import google.generativeai as genai
import os
from PIL import Image
import time

# Configure the Gemini API
genai.configure(api_key='AIzaSyCJj4fekxyhZrcqgK7ltol6AgT7kyFedxg')

# Initialize the Gemini model
model = genai.GenerativeModel('gemini-1.5-pro')

def analyze_image(image_path):
    # Load the image
    image = Image.open(image_path)
    
    # Create a prompt for OCR and specific information extraction
    prompt = """Please perform OCR on this image and extract the following information:
    1. MRP (Maximum Retail Price)
    2. Expiry Date or Best Before Date

    Provide your response in the following format:
    MRP: [extracted MRP]
    Date: [extracted expiry date or best before date]

    If the expiry date is not found, look for a "Best Before" date instead.
    If any information is not found, state 'Not found' for that category."""
    
    try:
        # Generate content using Gemini
        response = model.generate_content([prompt, image])
        
        # Extract the analysis from the response
        analysis = response.text
        
        print("Image Analysis:")
        print(analysis)
        
        # Save the analysis to a file (optional)
        with open("image_analysis.txt", "w") as file:
            file.write(analysis)
    except Exception as e:
        print(f"An error occurred during image analysis: {str(e)}")

def automatic_capture_and_analyze():
    cap = cv2.VideoCapture(0)  # Start video capture from the webcam
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return None

    # Define the static bounding box coordinates (x_start, y_start, x_end, y_end)
    x_start, y_start = 100, 100  # Top-left corner
    x_end, y_end = 400, 300  # Bottom-right corner

    print("Capturing and analyzing images automatically every 7 seconds. Press 'q' to quit.")

    last_capture_time = time.time()
    capture_interval = 7  # Capture every 7 seconds

    while True:
        ret, frame = cap.read()  # Read a frame from the webcam
        if not ret:
            print("Error: Could not read frame.")
            break

        # Draw the static bounding box on the frame
        cv2.rectangle(frame, (x_start, y_start), (x_end, y_end), (0, 255, 0), 2)
        cv2.imshow('Webcam', frame)  # Display the current frame

        current_time = time.time()
        if current_time - last_capture_time >= capture_interval:
            print("\nCapturing image...")
            # Crop the image using the bounding box coordinates
            cropped_image = frame[y_start:y_end, x_start:x_end]
            cropped_img_path = 'cropped_image.jpg'
            cv2.imwrite(cropped_img_path, cropped_image)
            cv2.imshow('Cropped Image', cropped_image)  # Show the cropped image
            
            # Analyze the cropped image using Gemini
            analyze_image(cropped_img_path)
            
            last_capture_time = current_time

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):  # Press 'q' to exit
            print("Exiting...")
            break

    cap.release()  # Release the webcam
    cv2.destroyAllWindows()  # Close all OpenCV windows

# Example usage
automatic_capture_and_analyze()