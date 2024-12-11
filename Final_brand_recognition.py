import cv2
import google.generativeai as genai
import os
from PIL import Image
import time

# Configure the Gemini API
genai.configure(api_key='AIzaSyBcPuWoRgmri2nvzD87VW_fmezdNOh5ziE')

# Initialize the Gemini model
model = genai.GenerativeModel('gemini-1.5-pro')

def analyze_image(image_path):
    # Load the image
    image = Image.open(image_path)
    
    # Create a prompt for OCR and brand recognition
    prompt = """Please perform the following tasks on this image:
    1. OCR: Extract any visible text.
    2. Brand Recognition: Identify any visible brand names or logos.
    
    Provide your response in the following format:
    OCR Text: [extracted text]
    Brand: [identified brand name]
    
    If no text or brand is detected, state 'None detected' for that category."""
    
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

def capture_and_analyze_image(frame, x_start, y_start, x_end, y_end):
    print("\nCapturing image...")
    img_path = 'captured_image.jpg'
    cv2.imwrite(img_path, frame)
    print("Image captured and saved as:", img_path)

    # Crop the image using the bounding box coordinates
    cropped_image = frame[y_start:y_end, x_start:x_end]
    cropped_img_path = 'cropped_image.jpg'
    cv2.imwrite(cropped_img_path, cropped_image)
    cv2.imshow('Cropped Image', cropped_image)  # Show the cropped image

    # Analyze the cropped image using Gemini
    analyze_image(cropped_img_path)

def capture_image_on_demand():
    cap = cv2.VideoCapture(0)  # Start video capture from the webcam
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return None

    # Define the static bounding box coordinates (x_start, y_start, x_end, y_end)
    x_start, y_start = 100, 100  # Top-left corner
    x_end, y_end = 400, 300      # Bottom-right corner

    print("Press 'c' to capture an image manually. Images will be automatically captured every 7 seconds. Press 'q' to quit.")

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
            capture_and_analyze_image(frame, x_start, y_start, x_end, y_end)
            last_capture_time = current_time

        key = cv2.waitKey(1) & 0xFF
        if key == ord('c'):
            capture_and_analyze_image(frame, x_start, y_start, x_end, y_end)
            last_capture_time = current_time
        elif key == ord('q'):  # Press 'q' to exit
            print("Exiting...")
            break

    cap.release()  # Release the webcam
    cv2.destroyAllWindows()  # Close all OpenCV windows

# Example usage
capture_image_on_demand()