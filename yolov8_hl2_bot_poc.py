'''
Author: Daryl A.

This script is the second step in making a bot. It follows from this: https://github.com/allend2092/YOLO_object_recognition_for_video_games

Here I've implemented YOLOv8, but now the script takes very simple actions based on what was detected in the screen. I've reduced to OpenCV recording
window (and the game itself) to 640x480 resolution to boost the performance of model inference. 

When the bot detects a person at a confidence of 70%, it will walk forward and move the mouse in the general direction of the "person". 
I've put person in air quotes here because YOLOv8 detects things it thinks are people, but are not. However, this script proves that I can generate
some action from a script that only detects what is on the screen. 

Additionally, this script uses multiple threads to perform the mouse and keyboard inputs. This frees the main thread to continue with object recognition. I think I might
spawn object recognition to its own thread in a future code version.

'''

import cv2
import numpy as np
from PIL import ImageGrab
from screeninfo import get_monitors
from ultralytics import YOLO
from pynput.keyboard import Controller, Key
import pyautogui
import time
import threading

# Load YOLOv8 model
model = YOLO('yolov8x.pt')  # Ensure the 'yolov8x.pt' file is in the same directory as this script
keyboard = Controller()

# Shared variables and events
person_detected = threading.Event()
person_coordinates = (0, 0, 0, 0)
confidence = 0

# Lock for updating shared variables
lock = threading.Lock()


def capture_screen(region=None):
    screenshot = ImageGrab.grab(bbox=region)
    screenshot_np_array = np.array(screenshot)
    return cv2.cvtColor(screenshot_np_array, cv2.COLOR_RGB2BGR)


def move_mouse():
    global person_coordinates
    while True:
        person_detected.wait()
        with lock:
            x1, y1, x2, y2 = person_coordinates
            current_confidence = confidence

        if current_confidence >= 0.80:
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            screen_center_x = 640 // 2
            screen_center_y = (480 + 30) // 2  # Adjust for the window bar

            # Calculate the distance to move the mouse
            move_x = center_x - screen_center_x
            move_y = center_y - screen_center_y

            # Move the mouse slowly toward the person
            if abs(move_x) > 100 or abs(move_y) > 100:  # Increase the tolerance for aiming
                pyautogui.moveRel(move_x * 1, move_y * 1, duration=0.15)
            else:
                # If the person is in the center, fire the weapon
                pyautogui.click(button='left')
        else:
            # Stop mouse movement if no person is detected
            pyautogui.moveRel(0, 0, duration=0.1)

        time.sleep(0.1)  # Add a short sleep to prevent high CPU usage


def press_key():
    while True:
        person_detected.wait()
        with lock:
            current_confidence = confidence

        if current_confidence >= 0.70:
            keyboard.press('w')
            time.sleep(0.2)
            keyboard.release('w')
        else:
            # Release the key if no person is detected
            keyboard.release('w')

        time.sleep(0.1)  # Add a short sleep to prevent high CPU usage


def main():
    global person_coordinates, confidence
    # Set the region of the screen to capture (coordinates for upper right section, 640x480 + 30 for window bar)
    region = (1280, 0, 1920, 510)  # Adjust these values based on your screen resolution

    # Find the second monitor (not used in this example but kept for completeness)
    monitors = get_monitors()
    if len(monitors) < 2:
        print("Second monitor not found!")
        return

    second_monitor = monitors[1]

    # Create a window with the same size as the game window
    cv2.namedWindow('YOLOv8 Object Detection', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('YOLOv8 Object Detection', 640, 480)
    cv2.moveWindow('YOLOv8 Object Detection', second_monitor.x, second_monitor.y)

    # Start the mouse and key press threads
    threading.Thread(target=move_mouse, daemon=True).start()
    threading.Thread(target=press_key, daemon=True).start()

    try:
        while True:
            screenshot = capture_screen(region)
            resized_screenshot = cv2.resize(screenshot, (640, 480))  # Resize to the original resolution of the game

            results = model(resized_screenshot)

            person_detected.clear()
            found_person = False

            for result in results:
                if result.boxes is not None and len(result.boxes) > 0:
                    for box in result.boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                        conf = box.conf[0].item()
                        cls = int(box.cls[0].item())
                        label = f'{model.names[cls]} {conf:.2f}'

                        if model.names[cls] == 'person' and conf >= 0.70:
                            with lock:
                                person_coordinates = (x1, y1, x2, y2)
                                confidence = conf
                            person_detected.set()
                            found_person = True
                            print(
                                f"Detected person at coordinates: x1={x1}, y1={y1}, x2={x2}, y2={y2}, confidence={conf:.2f}")

                        # Draw the bounding boxes and labels on the screenshot
                        cv2.rectangle(screenshot, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(screenshot, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            if not found_person:
                with lock:
                    confidence = 0  # Reset confidence if no person is detected

            cv2.imshow('YOLOv8 Object Detection', screenshot)

            if cv2.waitKey(1) == ord('q'):
                break
    except KeyboardInterrupt:
        print("Interrupted by user.")
    finally:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()







