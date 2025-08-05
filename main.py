import cv2
import numpy as np
import mediapipe as mp
import pyautogui
import time
import webbrowser
from threading import Timer

#Initialize webcam and MediaPipe Hands
cap = cv2.VideoCapture(0)
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Cooldown setup
last_action_time = 0
last_action = None
cooldown = 0.8  # seconds - slightly reduced for better responsiveness

# Program state
game_started = False
calibration_complete = False
hand_present = False
countdown_active = False
countdown_value = 3

# Calibration values
hand_size_samples = []
MIN_SAMPLES = 30

# Game URL
GAME_URL = "https://poki.com/en/g/subway-surfers"

def launch_game():
    """Open the Subway Surfers game in the default browser"""
    webbrowser.open(GAME_URL)
    global game_started
    game_started = True

def calculate_hand_size(landmarks, img_width, img_height):
    """Calculate the size of the hand based on landmarks"""
    x_coords = [landmark.x * img_width for landmark in landmarks.landmark]
    y_coords = [landmark.y * img_height for landmark in landmarks.landmark]
    
    x_min, x_max = min(x_coords), max(x_coords)
    y_min, y_max = min(y_coords), max(y_coords)
    
    width = x_max - x_min
    height = y_max - y_min
    
    return width * height

def draw_progress_bar(frame, progress, y_position):
    """Draw a progress bar on the frame"""
    width = int(400 * progress)
    cv2.rectangle(frame, (50, y_position), (450, y_position + 30), (255, 255, 255), 2)
    cv2.rectangle(frame, (50, y_position), (50 + width, y_position + 30), (0, 255, 0), -1)
    cv2.putText(frame, f"{int(progress * 100)}%", (460, y_position + 20), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

def start_countdown():
    """Start the countdown timer"""
    global countdown_active, countdown_value
    countdown_active = True
    countdown_value = 3

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)  # Mirror image for intuitive interaction
        height, width, _ = frame.shape
        cx, cy = width // 2, height // 2

        # Create a clean overlay for each frame
        overlay = frame.copy()
        
        # Process image and detect hands
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)
        
        # Check if hand is detected
        hand_present = result.multi_hand_landmarks is not None
        now = time.time()
        action = None

        # Define action zones using your original definitions
        left_poly = np.array([[0, 0], [200, 150], [200, 340], [0, height - 1]], np.int32)
        right_poly = np.array([[width - 1, 0], [430, 150], [430, 340], [width - 1, height - 1]], np.int32)
        up_poly = np.array([[0, 0], [200, 150], [430, 150], [width - 1, 0]], np.int32)
        down_poly = np.array([[0, height - 1], [200, 340], [430, 340], [width - 1, height - 1]], np.int32)
        center_rect = (200, 150, 230, 190)  # x, y, w, h

        # Process hand landmarks if detected
        if result.multi_hand_landmarks:
            for handLms in result.multi_hand_landmarks:
                # Draw hand landmarks
                mp_draw.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS)
                
                # Get center of the palm (landmark 9 is the middle of the palm)
                palm_center = handLms.landmark[9]
                hand_x, hand_y = int(palm_center.x * width), int(palm_center.y * height)
                
                # Draw palm center
                cv2.circle(frame, (hand_x, hand_y), 10, (255, 0, 0), -1)
                
                # Collect calibration data
                if not calibration_complete:
                    hand_size = calculate_hand_size(handLms, width, height)
                    hand_size_samples.append(hand_size)
                    
                    if len(hand_size_samples) >= MIN_SAMPLES:
                        # Calculate average hand size and complete calibration
                        avg_hand_size = sum(hand_size_samples) / len(hand_size_samples)
                        calibration_complete = True
                        print(f"Calibration complete! Average hand size: {avg_hand_size:.2f}")
                else:
                    # Determine action zone based on palm center position
                    if cv2.pointPolygonTest(left_poly, (hand_x, hand_y), False) >= 0:
                        action = 'left'
                        cv2.fillPoly(overlay, [left_poly], (0, 255, 255))
                    elif cv2.pointPolygonTest(right_poly, (hand_x, hand_y), False) >= 0:
                        action = 'right'
                        cv2.fillPoly(overlay, [right_poly], (0, 255, 255))
                    elif cv2.pointPolygonTest(up_poly, (hand_x, hand_y), False) >= 0:
                        action = 'up'
                        cv2.fillPoly(overlay, [up_poly], (0, 255, 255))
                    elif cv2.pointPolygonTest(down_poly, (hand_x, hand_y), False) >= 0:
                        action = 'down'
                        cv2.fillPoly(overlay, [down_poly], (0, 255, 255))
                    elif 200 <= hand_x <= 430 and 150 <= hand_y <= 340:
                        action = 'center'
                        cv2.rectangle(overlay, (200, 150), (430, 340), (0, 255, 255), -1)

        # Overlay with transparency
        alpha = 0.4
        frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

        # Display status based on program state
        if not calibration_complete:
            # Show calibration progress
            cv2.putText(frame, "Hand Calibration", (width//2 - 120, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            progress = min(1.0, len(hand_size_samples) / MIN_SAMPLES)
            draw_progress_bar(frame, progress, 70)
            
            if not hand_present:
                cv2.putText(frame, "Please show your hand to calibrate", (width//2 - 200, 150), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        elif not game_started:
            # Show start button
            start_button = (width//2 - 100, height//2 - 50, 200, 100)
            cv2.rectangle(frame, (start_button[0], start_button[1]), 
                         (start_button[0] + start_button[2], start_button[1] + start_button[3]), (0, 255, 0), -1)
            cv2.putText(frame, "START GAME", (start_button[0] + 10, start_button[1] + 60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            # Check if hand is in start button
            if hand_present and action == 'center' and not countdown_active:
                start_countdown()
                
            # Handle countdown
            if countdown_active:
                cv2.putText(frame, f"Starting in {countdown_value}...", 
                            (width//2 - 150, height//2 + 100), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
                
                # Update countdown
                if not hasattr(start_countdown, 'last_tick'):
                    start_countdown.last_tick = time.time()
                
                if time.time() - start_countdown.last_tick >= 1:
                    countdown_value -= 1
                    start_countdown.last_tick = time.time()
                    
                    if countdown_value <= 0:
                        countdown_active = False
                        launch_game()
        
        # Always draw the zone outlines exactly as specified
        cv2.rectangle(frame, (200, 150), (430, 340), (0, 255, 255), 2)
        cv2.polylines(frame, [left_poly], True, (0, 255, 255), 2)
        cv2.polylines(frame, [right_poly], True, (0, 255, 255), 2)
        cv2.polylines(frame, [up_poly], True, (0, 255, 255), 2)
        cv2.polylines(frame, [down_poly], True, (0, 255, 255), 2)

        # Draw labels exactly as specified
        cv2.putText(frame, "Center", (cx, cy + 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.putText(frame, "Left", (30, cy + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.putText(frame, "Right", (width - 150, cy + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2)
        cv2.putText(frame, "Up", (cx + 60, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        cv2.putText(frame, "Down", (cx + 50, height - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # Show game active status when game has started
        if game_started:
            cv2.putText(frame, "Game Controls Active", (30, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            if not hand_present:
                cv2.putText(frame, "Hand not detected!", (30, 60), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Perform action with cooldown (only if game has started)
        if action and game_started and (now - last_action_time > cooldown or action != last_action):
            if action != 'center':
                pyautogui.press(action)
            print(f"Action: {action}")
            last_action_time = now
            last_action = action

        # Display frame
        cv2.imshow("Subway Surfer Gesture Control", frame)
        
        # Check for exit key
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):  # Reset calibration
            calibration_complete = False
            hand_size_samples = []
            print("Calibration reset")
        elif key == ord('s') and calibration_complete and not game_started:  # Manual start
            launch_game()

except Exception as e:
    print(f"Error: {e}")
finally:
    cap.release()
    cv2.destroyAllWindows()
    print("Program exited.")