import cv2
import numpy as np
import pyautogui

# Function to count the number of fingers
def count_fingers(thresholded, hand_segment):
    # Approximate the contour
    approx = cv2.approxPolyDP(hand_segment, 0.02 * cv2.arcLength(hand_segment, True), True)
    
    # Find convex hull
    hull = cv2.convexHull(approx, returnPoints=False)
    if len(hull) < 3:  # Skip if the hull does not have enough points
        return 0

    try:
        # Find convexity defects
        defects = cv2.convexityDefects(approx, hull)
    except cv2.error as e:
        return 0

    if defects is None:
        return 0

    # Count fingers
    finger_count = 0
    for i in range(defects.shape[0]):
        s, e, f, d = defects[i, 0]
        start = tuple(approx[s][0])
        end = tuple(approx[e][0])
        far = tuple(approx[f][0])

        a = np.linalg.norm(np.array(end) - np.array(start))
        b = np.linalg.norm(np.array(far) - np.array(start))
        c = np.linalg.norm(np.array(end) - np.array(far))
        
        angle = np.arccos((b**2 + c**2 - a**2) / (2*b*c))

        if angle <= np.pi / 2:
            finger_count += 1
    
    return finger_count

# Open a connection to the webcam
cap = cv2.VideoCapture(0)

# Background subtraction to detect hand
bg_subtractor = cv2.createBackgroundSubtractorMOG2(detectShadows=False)

while cap.isOpened():
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)  # Flip the frame horizontally
    frame_copy = frame.copy()

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply background subtraction
    bg_mask = bg_subtractor.apply(gray)

    # Apply some filters to remove noise
    blurred = cv2.GaussianBlur(bg_mask, (7, 7), 0)
    _, thresholded = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY)

    # Find contours
    contours, _ = cv2.findContours(thresholded.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        # Sort contours by area and keep the largest one
        hand_segment = max(contours, key=cv2.contourArea)

        # Draw the hand segment
        cv2.drawContours(frame_copy, [hand_segment], -1, (0, 255, 0), 1)

        # Count the number of fingers
        fingers = count_fingers(thresholded, hand_segment)

        # Display the number of fingers detected
        cv2.putText(frame_copy, f"Fingers: {fingers}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Determine which hand and perform the corresponding action
        hand_center_x = np.mean(hand_segment[:, 0, 0])
        frame_center_x = frame.shape[1] // 2

        if hand_center_x < frame_center_x:  # Left hand
            if fingers == 1:
                print("Left hand: One finger detected - Moving video backward")
                pyautogui.press('left')
        else:  # Right hand
            if fingers == 1:
                print("Right hand: One finger detected - Moving video forward")
                pyautogui.press('right')

    # Display the frame
    cv2.imshow('Hand Gesture Control', frame_copy)

    # Break loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close OpenCV windows
cap.release()
cv2.destroyAllWindows()
