import Gate
import LicensePlateRecognitionCamera
import ParkingCamera
import Sensor

parkingCameraIndex = 0
licensePlateCameraIndex = 1

if __name__ == "__main__":
    Gate.setup()
    ParkingCamera.setup(parkingCameraIndex)
    LicensePlateRecognitionCamera.setup(licensePlateCameraIndex)
    Sensor.setup()
