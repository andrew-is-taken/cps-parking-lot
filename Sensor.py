import lgpio as GPIO
import time

TRIG = 23
ECHO = 24

h = GPIO.gpiochip_open(0)
GPIO.gpio_claim_output(h, TRIG)
GPIO.gpio_claim_input(h, ECHO)


def get_distance():
    """
    Measures the distance using the HC-SR04 ultrasonic sensor.
    """
    GPIO.gpio_write(h, TRIG, 0)
    time.sleep(2)

    GPIO.gpio_write(h, TRIG, 1)
    time.sleep(0.00001)
    GPIO.gpio_write(h, TRIG, 0)

    pulse_start = time.time()
    pulse_end = time.time()

    while GPIO.gpio_read(h, ECHO) == 0:
        pulse_start = time.time()
        if time.time() - pulse_start > 0.1:
            return -1

    while GPIO.gpio_read(h, ECHO) == 1:
        pulse_end = time.time()
        if time.time() - pulse_end > 0.1:
            return -1

    pulse_duration = pulse_end - pulse_start

    distance = pulse_duration * 17150
    distance = round(distance, 2)

    return distance


def setup():
    try:
        print("HC-SR04 Distance Measurement")
        print("----------------------------")
        while True:
            dist = get_distance()
            if dist != -1:
                print("Measured Distance = {:.2f} cm".format(dist))
            else:
                print("Measurement failed or out of range. Retrying...")
            time.sleep(1)

    except KeyboardInterrupt:
        print("\nMeasurement stopped by User")
    finally:
        GPIO.gpiochip_close(h)
        print("GPIO resources released.")
