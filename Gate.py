import RPi.GPIO as GPIO
import time

SERVO_PIN = 18
pwm = GPIO.PWM(SERVO_PIN, 50)


def set_angle(angle, delay):
    duty = angle / 18 + 2
    pwm.ChangeDutyCycle(duty)
    time.sleep(0.5)
    pwm.ChangeDutyCycle(0)
    time.sleep(delay)


def setup():
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(SERVO_PIN, GPIO.OUT)

    pwm.start(0)

# try:
#     print("System ready.")
#     print("Type 'g' and press Enter to move gate left.")
#     print("Press Ctrl+C to exit.")
#
#     set_angle(DOWN_POSITION)
#
#     while True:
#         cmd = input("Command: ").strip().lower()
#         if cmd == 'g':
#             print("Gate moving left...")
#             set_angle(LEFT_POSITION, 5)
#
#             print("Returning to down position...")
#             set_angle(DOWN_POSITION, 0)
#         else:
#             print("Unknown command. Type 'g' to activate the gate.")
#
# except KeyboardInterrupt:
#     print("\nStopped by User.")
#
# finally:
#     pwm.stop()
#     GPIO.cleanup()
#     print("GPIO cleaned up. Exiting.")
