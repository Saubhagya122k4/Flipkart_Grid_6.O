import cv2
import google.generativeai as genai
import os
from PIL import Image
import time

# Configure the Gemini API
genai.configure(api_key='AIzaSyDc3T4p7ftiJftATQx_58NpM72iHxjm4gk')

# Initialize the Gemini model
model = genai.GenerativeModel('gemini-1.5-pro')

def analyze_image(image_path):
    # Load the image
    image = Image.open(image_path)
 
    # Create a prompt for OCR and specific information extraction
    prompt = """Please perform OCR on this image and extract the following information:
    1. Brand details
    2. Pack Size
    3. Brand size

    Provide your response in the following format:
    Brand details: [extracted brand details]
    Pack Size: [extracted pack size]
    Brand size: [extracted brand size]

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
        # return None # Define the static bounding box coordinat]edc key == ord('q'):  # Press 'q' to exit
        #     print("Exiting...")
        #     break

    cap.release()  # Release the webcam
    cv2.destroyAllWindows()  # Close all OpenCV windows

# Example usage
automatic_capture_and_analyze()