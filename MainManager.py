import cv2
import time
import atexit
import signal
import sys
import threading
import queue
import imutils
import easyocr
from gpiozero import LED
import drivers
import RPi.GPIO as GPIO

# Leds
led_green = LED(17)
led_red = LED(27)
lcd_display = drivers.Lcd()

# Parking lot camera
PARKING_LOT_CAM_INDEX = 0
FRAME_WIDTH_PARK = 640
FRAME_HEIGHT_PARK = 480
BACKGROUND_CAPTURE_DELAY_SEC = 3
THRESHOLD_SENSITIVITY = 30
MIN_CONTOUR_AREA = 500
PARKING_SPACES = {
    "Space 1": [170, 50, 80, 100],
    "Space 2": [280, 50, 80, 100],
    "Space 3": [450, 230, 100, 80],
}

# License plate recognition
LPR_CAM_INDEX = 2
FRAME_WIDTH_LPR = 400
ROI_PERCENT_TOP = 0.5
OCR_EVERY_N_FRAMES = 10
CONF_THRESHOLD = 0.5
AUTHORIZED_PLATES = {'A1', 'X4'}

# Gate
SERVO_PIN = 18
OPEN_POSITION = 150
CLOSE_POSITION = 60
GATE_OPEN_DURATION_SEC = 5

# Ultrasonic Sensor
ULTRASONIC_TRIG_PIN = 23
ULTRASONIC_ECHO_PIN = 24
DETECTION_DISTANCE_CM = 7


class ParkingLotManager:
    """
    Manages the parking lot monitoring system.

    This class handles the video stream from the parking lot camera, performs
    motion detection to determine parking space occupancy, and updates the
    LCD display and status LEDs accordingly.
    """
    def __init__(self, camera_index, led_green, led_red, lcd_display):
        """
        Initializes the ParkingLotManager.

        Args:
            camera_index (int): The index of the camera to use for monitoring.
            led_green (LED): The green LED object for indicating free spaces.
            led_red (LED): The red LED object for indicating no free spaces.
            lcd_display (drivers.Lcd): The LCD display object.
        """
        self.cap = cv2.VideoCapture(camera_index)
        self.led_green = led_green
        self.led_red = led_red
        self.lcd = lcd_display
        self.background_blur = None
        self.last_lcd_update = 0.0
        self.last_free_count = -1

    def cleanup(self):
        """Releases camera resources and turns off LEDs and LCD."""
        print("Cleaning up ParkingLotManager...")
        try:
            self.lcd.lcd_clear()
        except Exception:
            pass
        self.led_green.off()
        self.led_red.off()
        if self.cap.isOpened():
            self.cap.release()
        cv2.destroyWindow("Parking Lot Monitor")
        cv2.destroyWindow("Threshold (Debug)")

    def update_lcd(self, free_count):
        """Updates the LCD display with the current number of free spaces."""
        self.lcd.lcd_clear()
        if free_count <= 0:
            self.lcd.lcd_display_string("No empty parking", 1)
            self.lcd.lcd_display_string("spaces :(", 2)
        else:
            self.lcd.lcd_display_string("Empty parking", 1)
            self.lcd.lcd_display_string(f"spaces: {free_count}.", 2)

    def setup(self):
        """
        Sets up the camera and captures the initial background frame.

        Returns:
            bool: True if setup is successful, False otherwise.
        """
        if not self.cap.isOpened():
            print(f"Error: Could not open camera {PARKING_LOT_CAM_INDEX}.")
            return False

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH_PARK)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT_PARK)

        print("--- Parking Lot Monitor Initializing ---")
        print("Please ensure all parking spaces are EMPTY...")
        for i in range(BACKGROUND_CAPTURE_DELAY_SEC, 0, -1):
            print(f"{i}...")
            time.sleep(1)

        ret, background_frame = self.cap.read()
        if not ret:
            print("Error: Could not capture background frame.")
            return False

        background_gray = cv2.cvtColor(background_frame, cv2.COLOR_BGR2GRAY)
        self.background_blur = cv2.GaussianBlur(background_gray, (21, 21), 0)
        print("Background captured. Starting monitoring...")
        return True

    def loop(self):
        """
        The main loop for the parking lot monitor.

        It captures frames, processes them for motion, checks parking space
        occupancy, updates the UI, and displays the results.

        Returns:
            str: "CONTINUE" to keep the loop running, "STOP" to exit.
        """
        try:
            ret, frame = self.cap.read()
            if not ret:
                print("Error: Could not read frame from parking camera. Exiting...")
                return "STOP"

            current_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            current_blur = cv2.GaussianBlur(current_gray, (21, 21), 0)
            frame_diff = cv2.absdiff(self.background_blur, current_blur)
            _, thresh = cv2.threshold(frame_diff, THRESHOLD_SENSITIVITY, 255, cv2.THRESH_BINARY)
            dilated = cv2.dilate(thresh, None, iterations=2)
            contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            parking_status = {name: "FREE" for name in PARKING_SPACES}

            for name, (x, y, w, h) in PARKING_SPACES.items():
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                for contour in contours:
                    if cv2.contourArea(contour) >= MIN_CONTOUR_AREA:
                        (cx, cy, cw, ch) = cv2.boundingRect(contour)
                        if (cx < x + w) and (cx + cw > x) and (cy < y + h) and (cy + ch > y):
                            parking_status[name] = "OCCUPIED"
                            cv2.rectangle(frame, (cx, cy), (cx + cw, cy + ch), (0, 0, 255), 2)

            free_count = sum(1 for s in parking_status.values() if s == "FREE")
            now = time.time()
            if now - self.last_lcd_update >= 1.0 and free_count != self.last_free_count:
                self.update_lcd(free_count)
                self.last_lcd_update = now
                self.last_free_count = free_count

            if free_count > 0:
                self.led_green.on()
                self.led_red.off()
            else:
                self.led_green.off()
                self.led_red.on()

            y_offset = 45
            for name, status in parking_status.items():
                color = (0, 255, 0) if status == "FREE" else (0, 0, 255)
                cv2.putText(frame, f"{name}: {status}", (10, y_offset),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                y_offset += 25

            cv2.imshow("Parking Lot Monitor", frame)
            cv2.imshow("Threshold (Debug)", thresh)

            if cv2.waitKey(1) & 0xFF == 27:
                return "STOP"
            return "CONTINUE"

        except Exception as e:
            print(f"Error in ParkingLotManager loop: {e}")
            return "STOP"


class LPRManager:
    """
    Manages the license plate recognition (LPR) task.

    It uses a separate camera to capture frames, extracts a region of interest,
    and sends it to an EasyOCR worker thread for processing. It also handles
    access control based on recognized license plates, signaling the GateManager.
    """

    def __init__(self, camera_index, gate_manager):
        """
        Initializes the LPRManager.

        Args:
            camera_index (int): The index of the camera for LPR.
            gate_manager (GateManager): The gate manager object to control access.
        """
        self.cap = cv2.VideoCapture(camera_index)
        self.frame_queue = queue.Queue(maxsize=1)
        self.result_queue = queue.Queue()
        self.stop_event = threading.Event()
        self.ocr_thread = None
        self.frame_count = 0
        self.plate_text = ""
        self.gate_manager = gate_manager
        self.last_open_time = 0

    def ocr_worker(self):
        """
        A worker thread for performing OCR on queued frames.

        This thread continuously checks the frame queue for new images and
        uses EasyOCR to read text, putting the results into the result queue.
        """
        reader = easyocr.Reader(['en'], gpu=False)
        print("OCR worker thread started.")
        while not self.stop_event.is_set():
            try:
                roi = self.frame_queue.get(timeout=1)
                results = reader.readtext(roi)
                best_match = ""
                best_conf = 0
                for (_, text, conf) in results:
                    cleaned = ''.join(filter(str.isalnum, text.upper()))
                    if conf > best_conf and 2 <= len(cleaned) <= 10:
                        best_match = cleaned
                        best_conf = conf

                if best_conf > CONF_THRESHOLD:
                    self.result_queue.put(best_match)
                else:
                    self.result_queue.put("")
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Error in OCR worker: {e}")

        print("OCR worker thread stopped.")

    def setup(self):
        """
        Starts the OCR worker thread and initializes the LPR camera.

        Returns:
            bool: True if setup is successful, False otherwise.
        """
        self.ocr_thread = threading.Thread(target=self.ocr_worker, daemon=True)
        self.ocr_thread.start()

        if not self.cap.isOpened():
            print(f"Error: Could not open camera {LPR_CAM_INDEX}.")
            self.stop_event.set()
            self.ocr_thread.join()
            return False

        print("LPR camera opened. Starting main loop.")
        return True

    def cleanup(self):
        """Stops the OCR thread and releases camera resources."""
        print("Cleaning up LPRManager...")
        self.stop_event.set()
        if self.ocr_thread and self.ocr_thread.is_alive():
            self.ocr_thread.join()
        if self.cap.isOpened():
            self.cap.release()
        cv2.destroyWindow("Fast LPR (Pi 5 Optimized)")

    def loop(self):
        """
        The main loop for the LPR manager.

        It captures frames, resizes them, periodically sends a region of interest
        to the OCR thread, and checks the results. If an authorized plate is
        detected, it triggers the gate to open. It also handles closing the gate
        after a set duration.

        Returns:
            str: "CONTINUE" to keep the loop running, "STOP" to exit.
        """
        ret, frame = self.cap.read()
        if not ret:
            print("Failed to grab frame from LPR camera.")
            return "STOP"

        frame = imutils.resize(frame, width=FRAME_WIDTH_LPR)

        try:
            self.plate_text = self.result_queue.get_nowait()
        except queue.Empty:
            pass

        self.frame_count += 1
        if self.frame_count % OCR_EVERY_N_FRAMES == 0 and self.frame_queue.empty():
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            h, w = gray.shape
            roi = gray[int(h * ROI_PERCENT_TOP):, :]
            self.frame_queue.put(roi)

        if self.plate_text:
            cv2.putText(frame, f"Plate: {self.plate_text}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
            if self.plate_text in AUTHORIZED_PLATES:
                cv2.putText(frame, "ACCESS GRANTED", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                # Open the gate when access is granted and reset timer
                if not self.gate_manager.is_open:
                    self.gate_manager.open_gate()
                    self.last_open_time = time.time()
            else:
                cv2.putText(frame, "UNKNOWN PLATE", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        if self.gate_manager.is_open and time.time() - self.last_open_time > GATE_OPEN_DURATION_SEC:
            self.gate_manager.close_gate()
            self.last_open_time = 0

        cv2.imshow("Fast LPR (Pi 5 Optimized)", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            return "STOP"
        return "CONTINUE"


class GateManager:
    """
    Manages the physical gate using a servo motor.

    This class handles the initialization of the servo and provides methods
    to open and close the gate.
    """
    def __init__(self, servo_pin):
        """
        Initializes the GateManager.

        Args:
            servo_pin (int): The GPIO pin number connected to the servo's signal wire.
        """
        self.servo_pin = servo_pin
        self.pwm = None
        self.is_open = False

    def setup(self):
        """
        Initializes the servo motor with PWM control.

        Returns:
            bool: True if setup is successful, False otherwise.
        """
        try:
            GPIO.setmode(GPIO.BCM)
            GPIO.setup(self.servo_pin, GPIO.OUT)
            self.pwm = GPIO.PWM(self.servo_pin, 50)
            self.pwm.start(0)
            print("GateManager: Servo initialized. Moving to close position.")
            self.set_angle(CLOSE_POSITION)
            time.sleep(1)  # Give the servo time to move
            self.set_angle(CLOSE_POSITION)
            return True
        except Exception as e:
            print(f"Error initializing GateManager: {e}")
            return False

    def set_angle(self, angle):
        """
        Sets the servo's angle based on the provided angle in degrees.
        """
        duty = angle / 18 + 2
        self.pwm.ChangeDutyCycle(duty)
        time.sleep(0.5)
        self.pwm.ChangeDutyCycle(0)

    def open_gate(self):
        """Moves the servo to the open position."""
        if not self.is_open:
            print("GateManager: Opening gate...")
            self.set_angle(OPEN_POSITION)
            self.is_open = True

    def close_gate(self):
        """Moves the servo to the closed position."""
        if self.is_open:
            print("GateManager: Closing gate...")
            self.set_angle(CLOSE_POSITION)
            self.is_open = False

    def cleanup(self):
        """Stops the PWM and cleans up GPIO."""
        print("GateManager: Cleaning up GPIO...")
        if self.pwm:
            self.pwm.stop()


class UltrasonicSensorManager:
    """
    Manages the ultrasonic sensor for vehicle detection.

    This class uses a separate thread to continuously measure distance. If an
    object is detected within a certain range, it signals the gate manager to open.
    """
    def __init__(self, trig_pin, echo_pin, gate_manager):
        """
        Initializes the UltrasonicSensorManager.

        Args:
            trig_pin (int): The GPIO pin for the sensor's trigger.
            echo_pin (int): The GPIO pin for the sensor's echo.
            gate_manager (GateManager): The gate manager object.
        """
        self.trig_pin = trig_pin
        self.echo_pin = echo_pin
        self.gate_manager = gate_manager
        self.stop_event = threading.Event()
        self.sensor_thread = None

    def get_distance(self):
        """
        Measures the distance using the ultrasonic sensor.

        Returns:
            float: The distance in cm, or a large number on timeout.
        """
        GPIO.output(self.trig_pin, False)
        time.sleep(0.000002)

        GPIO.output(self.trig_pin, True)
        time.sleep(0.00001)
        GPIO.output(self.trig_pin, False)

        pulse_start = time.time()
        pulse_end = time.time()

        timeout_start = time.time()
        while GPIO.input(self.echo_pin) == 0:
            pulse_start = time.time()
            if time.time() - timeout_start > 1:
                return 99999

        timeout_start = time.time()
        while GPIO.input(self.echo_pin) == 1:
            pulse_end = time.time()
            if time.time() - timeout_start > 1:
                return 99999

        pulse_duration = pulse_end - pulse_start
        distance = (pulse_duration * 34300) / 2
        return distance

    def sensor_worker(self):
        """
        A worker thread for continuous distance measurement.

        If an object is detected within the threshold, it signals the gate
        to open.
        """
        print("UltrasonicSensorManager: Worker thread started.")
        while not self.stop_event.is_set():
            distance = self.get_distance()
            print(f"Ultrasonic Distance: {distance:.2f} cm")

            if distance < DETECTION_DISTANCE_CM and not self.gate_manager.is_open:
                print("🚨 Ultrasonic sensor detected an object. Opening the gate!")
                self.gate_manager.open_gate()

            time.sleep(0.5)

        print("UltrasonicSensorManager: Worker thread stopped.")

    def setup(self):
        """
        Initializes GPIO pins and starts the sensor worker thread.

        Returns:
            bool: True if setup is successful, False otherwise.
        """
        try:
            GPIO.setmode(GPIO.BCM)
            GPIO.setup(self.trig_pin, GPIO.OUT)
            GPIO.setup(self.echo_pin, GPIO.IN)
            self.sensor_thread = threading.Thread(target=self.sensor_worker, daemon=True)
            self.sensor_thread.start()
            print("UltrasonicSensorManager: Sensor initialized.")
            return True
        except Exception as e:
            print(f"Error initializing UltrasonicSensorManager: {e}")
            return False

    def cleanup(self):
        """Stops the sensor thread."""
        print("Cleaning up UltrasonicSensorManager...")
        self.stop_event.set()
        if self.sensor_thread and self.sensor_thread.is_alive():
            self.sensor_thread.join()


# --- Main Manager Class ---
class MainManager:
    """
    The central orchestrator for the entire smart parking system.

    This class initializes all component managers (ParkingLotManager, LPRManager,
    GateManager, UltrasonicSensorManager), handles their setup and cleanup, and
    runs the main application loop. It also manages signal handling for graceful
    shutdown.
    """
    def __init__(self):
        self.stop_event = threading.Event()
        self.all_managers = []
        self.loop_managers = []

    def run(self):
        """
        The main entry point for the application.

        It registers cleanup functions, sets up signal handlers, initializes all
        component managers, and starts the main application loop.
        """
        atexit.register(self.cleanup)
        for sig in (signal.SIGINT, signal.SIGTERM):
            signal.signal(sig, lambda s, f: self.stop_event.set())

        gate_manager = GateManager(SERVO_PIN)
        lpr_manager = LPRManager(LPR_CAM_INDEX, gate_manager)
        park_manager = ParkingLotManager(PARKING_LOT_CAM_INDEX, led_green, led_red, lcd_display)
        ultrasonic_manager = UltrasonicSensorManager(ULTRASONIC_TRIG_PIN, ULTRASONIC_ECHO_PIN, gate_manager)

        if not gate_manager.setup() or \
                not park_manager.setup() or \
                not lpr_manager.setup() or \
                not ultrasonic_manager.setup():
            print("One of the managers failed to set up. Exiting.")
            return

        self.loop_managers = [park_manager, lpr_manager]
        self.all_managers = [gate_manager, park_manager, lpr_manager, ultrasonic_manager]

        print("Starting parallel loops...")

        try:
            while not self.stop_event.is_set():
                for manager in self.loop_managers:
                    if manager.loop() == "STOP":
                        self.stop_event.set()
                        break
                time.sleep(0.01)

        except KeyboardInterrupt:
            print("Received keyboard interrupt. Stopping.")
        finally:
            self.cleanup()

    def cleanup(self):
        """
        Performs a graceful shutdown of all managers and GPIO.

        This method is called on exit to ensure all resources are properly
        released, threads are stopped, and GPIO is cleaned up.
        """
        if not self.stop_event.is_set():
            self.stop_event.set()

        print("Starting MainManager cleanup...")
        for manager in self.all_managers:
            manager.cleanup()

        GPIO.cleanup()

        cv2.destroyAllWindows()
        print("MainManager cleanup complete.")


if __name__ == "__main__":
    main_app = MainManager()
    main_app.run()