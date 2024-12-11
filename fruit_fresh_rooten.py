import cv2
import google.generativeai as genai
import os
from PIL import Image
import time

# Configure the Gemini API
genai.configure(api_key='AIzaSyDSxydyGgkIzpQDh3oIgsAP8Do5ma6bYHA')

# Initialize the Gemini model
model = genai.GenerativeModel('gemini-1.5-pro')

def analyze_image(image_path):
    # Load the image
    image = Image.open(image_path)
    
    # Create a prompt for fruit freshness detection
    prompt = """Analyze this image and determine if it contains a fruit. If a fruit is present, estimate its freshness on a scale of 0% (completely rotten) to 100% (perfectly fresh).

    Provide your response in the following format:
    Fruit Detected: [Yes/No]
    Fruit Type: [Name of the fruit, if detected]
    Freshness: [Percentage]
    Explanation: [Brief explanation of your assessment]

    If no fruit is detected, state 'No fruit detected' for all categories."""
    
    try:
        # Generate content using Gemini
        response = model.generate_content([prompt, image])
        
        # Extract the analysis from the response
        analysis = response.text
        
        print("\nFruit Analysis:")
        print(analysis)
        
        # Save the analysis to a file (optional)
        with open("fruit_analysis.txt", "w") as file:
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

    print("Press 'c' to capture an image and analyze fruit freshness. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()  # Read a frame from the webcam
        if not ret:
            print("Error: Could not read frame.")
            break

        # Draw the static bounding box on the frame
        cv2.rectangle(frame, (x_start, y_start), (x_end, y_end), (0, 255, 0), 2)
        cv2.imshow('Webcam', frame)  # Display the current frame

        key = cv2.waitKey(1) & 0xFF
        if key == ord('c'):
            capture_and_analyze_image(frame, x_start, y_start, x_end, y_end)
        elif key == ord('q'):  # Press 'q' to exit
            print("Exiting...")
            break

    cap.release()  # Release the webcam
    cv2.destroyAllWindows()  # Close all OpenCV windows

def analyze_uploaded_image(image_path):
    print(f"\nAnalyzing uploaded image: {image_path}")
    analyze_image(image_path)

# Example usage
print("Choose an option:")
print("1. Capture image from webcam")
print("2. Upload an image file")
choice = input("Enter your choice (1 or 2): ")

if choice == '1':
    capture_image_on_demand()
elif choice == '2':
    image_path = input("Enter the path to your image file: ")
    if os.path.exists(image_path):
        analyze_uploaded_image(image_path)
    else:
        print("Error: The specified file does not exist.")
else:
    print("Invalid choice. Exiting.")