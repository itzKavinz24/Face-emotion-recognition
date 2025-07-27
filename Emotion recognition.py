from facial_emotion_recognition import EmotionRecognition
import cv2

# Initialize EmotionRecognition model to run on CPU
er = EmotionRecognition(device='cpu')

# Initialize webcam (try 0, change to 1 if needed)
cam = cv2.VideoCapture(0)

# Check if webcam opened successfully
if not cam.isOpened():
    print("❌ Cannot open webcam.")
    exit()

while True:
    success, frame = cam.read()

    # If frame not captured correctly, skip
    if not success or frame is None:
        print("⚠️ Failed to read frame from webcam.")
        continue

    # Run emotion recognition
    try:
        frame = er.recognise_emotion(frame, return_type='BGR')
    except Exception as e:
        print(f"Error processing frame: {e}")
        continue

    # Display the processed frame
    cv2.imshow("Recognition", frame)

    # Break the loop if ESC key is pressed
    key = cv2.waitKey(1)
    if key == 27:  # ESC key
        break

# Release resources
cam.release()
cv2.destroyAllWindows()
