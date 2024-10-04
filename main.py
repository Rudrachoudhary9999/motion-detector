import cv2
import pygame

pygame.mixer.init()  # Initialize the Pygame mixer

# Load the sound file
sound_file = "D:\mixkit-classic-alarm-995.wav"
sound = pygame.mixer.Sound(sound_file)

# Initialize the video capture object
cap = cv2.VideoCapture(0)

# Initialize the motion detection parameters
threshold = 1000  # Minimum number of pixels for detecting motion
frame_area = 640 * 480  # Area of the frame

# Initialize the previous frame
prev_frame = None

while True:
    # Capture the current frame
    ret, frame = cap.read()

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Blur the frame to remove noise
    gray = cv2.GaussianBlur(gray, (21, 21), 0)

    # Initialize the motion flag to False
    motion_detected = False

    # If this is the first frame, initialize the previous frame
    if prev_frame is None:
        prev_frame = gray
    else:
        # Calculate the absolute difference between the current frame and the previous frame
        frame_diff = cv2.absdiff(prev_frame, gray)

        # Threshold the frame difference to highlight the regions with significant differences
        thresh = cv2.threshold(frame_diff, 25, 255, cv2.THRESH_BINARY)[1]

        # Find the contours in the thresholded image
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Loop over the contours
        for contour in contours:
            # If the contour is too small, ignore it
            if cv2.contourArea(contour) < threshold:
                continue

            # Motion detected
            motion_detected = True

            # Play the sound
            sound.play()

        # Update the previous frame
        prev_frame = gray

    # Display the frame
    cv2.imshow("Frame", frame)

    # Wait for a key press
    key = cv2.waitKey(1) & 0xFF

    # If the 'q' key is pressed, stop the loop
    if key == ord("q"):
        break

# Release the video capture object and close all windows
cap.release()
cv2.destroyAllWindows()
