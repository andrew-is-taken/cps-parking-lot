import cv2
import time
import atexit
import signal
import sys
from gpiozero import LED
import drivers

led = LED(17)
led2 = LED(27)
display = drivers.Lcd()

FRAME_WIDTH = 640
FRAME_HEIGHT = 480
BACKGROUND_CAPTURE_DELAY_SEC = 3
THRESHOLD_SENSITIVITY = 70
MIN_CONTOUR_AREA = 1000

PARKING_SPACES = {
    "Space 1": [0, 380, 80, 100],
    "Space 2": [140, 380, 80, 100],
    "Space 3": [280, 380, 80, 100],
}

cap = None


def cleanup():
    """Clear LCD and release GUI resources no matter how we exit."""
    try:
        display.lcd_clear()
    except Exception:
        pass
    try:
        led.off()
    except Exception:
        pass
    try:
        led2.off()
    except Exception:
        pass
    try:
        if cap is not None:
            cap.release()
    except Exception:
        pass
    try:
        cv2.destroyAllWindows()
    except Exception:
        pass


def update_lcd(free_count):
    """Show the required message on the LCD."""
    display.lcd_clear()
    if free_count <= 0:
        display.lcd_display_string("No empty parking", 1)
        display.lcd_display_string("spaces :(", 2)
    else:
        display.lcd_display_string("Empty parking", 1)
        display.lcd_display_string(f"spaces: {free_count}.", 2)


def setup(cameraIndex):
    led.off()
    led2.off()

    atexit.register(cleanup)
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda s, f: sys.exit(0))

    cap = cv2.VideoCapture(cameraIndex)
    if not cap.isOpened():
        print(f"Error: Could not open camera {cameraIndex}.")
        print("Check if the camera is connected and not in use by another application.")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    print("--- Parking Lot Monitor Initializing ---")
    print(f"Please ensure all parking spaces are EMPTY for {BACKGROUND_CAPTURE_DELAY_SEC} seconds.")
    print("Capturing background in...")

    for i in range(BACKGROUND_CAPTURE_DELAY_SEC, 0, -1):
        print(f"{i}...")
        time.sleep(1)

    ret, background_frame = cap.read()
    if not ret:
        print("Error: Could not capture background frame.")
        sys.exit(1)

    background_gray = cv2.cvtColor(background_frame, cv2.COLOR_BGR2GRAY)
    background_blur = cv2.GaussianBlur(background_gray, (21, 21), 0)

    print("Background captured. Starting monitoring...")

    last_lcd_update = 0.0
    lcd_interval = 1.0
    last_free_count = None

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame. Exiting...")
                break

            current_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            current_blur = cv2.GaussianBlur(current_gray, (21, 21), 0)

            frame_diff = cv2.absdiff(background_blur, current_blur)
            _, thresh = cv2.threshold(frame_diff, THRESHOLD_SENSITIVITY, 255, cv2.THRESH_BINARY)
            dilated = cv2.dilate(thresh, None, iterations=2)
            contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            parking_status = {name: "FREE" for name in PARKING_SPACES}

            for name, (x, y, w, h) in PARKING_SPACES.items():
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                for contour in contours:
                    if cv2.contourArea(contour) < MIN_CONTOUR_AREA:
                        continue
                    (cx, cy, cw, ch) = cv2.boundingRect(contour)
                    if (cx < x + w) and (cx + cw > x) and (cy < y + h) and (cy + ch > y):
                        parking_status[name] = "OCCUPIED"
                        cv2.rectangle(frame, (cx, cy), (cx + cw, cy + ch), (0, 0, 255), 2)

            free_count = sum(1 for s in parking_status.values() if s == "FREE")

            overlay_text = ("No empty parking spaces :("
                            if free_count == 0
                            else f"Empty parking spaces: {free_count}.")
            cv2.putText(frame, overlay_text, (10, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

            now = time.time()
            if (now - last_lcd_update) >= lcd_interval and free_count != last_free_count:
                update_lcd(free_count)
                last_lcd_update = now
                last_free_count = free_count
            if free_count > 0:
                led.on()
                led2.off()
            else:
                led.off()
                led2.on()

            y_offset = 45
            for name, status in parking_status.items():
                color = (0, 255, 0) if status == "FREE" else (0, 0, 255)
                cv2.putText(frame, f"{name}: {status}", (10, y_offset),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                y_offset += 25

            cv2.imshow("Parking Lot Monitor", frame)
            cv2.imshow("Threshold (Debug)", thresh)

            if cv2.waitKey(1) & 0xFF == 27:
                break

    finally:
        cleanup()
        print("Monitoring stopped (LCD cleared).")
