import cv2
import imutils
import easyocr
import threading
import queue
import Gate

FRAME_WIDTH = 400
ROI_PERCENT_TOP = 0.5
OCR_EVERY_N_FRAMES = 10
CONF_THRESHOLD = 0.5
AUTHORIZED_PLATES = {'A1'}

OPEN_POSITION = 90
CLOSED_POSITION = 180

def ocr_worker(frame_queue, result_queue, stop_event):
    """
    This function runs in a separate thread and performs OCR on frames it receives.
    """
    reader = easyocr.Reader(['en'], gpu=False)
    print("OCR worker thread started.")
    while not stop_event.is_set():
        try:
            roi = frame_queue.get(timeout=1)
            results = reader.readtext(roi)
            best_match = ""
            best_conf = 0

            for (_, text, conf) in results:
                cleaned = ''.join(filter(str.isalnum, text.upper()))
                if conf > best_conf and 2 <= len(cleaned) <= 10:
                    best_match = cleaned
                    best_conf = conf
            
            if best_conf > CONF_THRESHOLD:
                result_queue.put(best_match)
            else:
                result_queue.put("")

        except queue.Empty:
            continue
        except Exception as e:
            print(f"Error in OCR worker: {e}")
            
    print("OCR worker thread stopped.")


def setup(cameraIndex):
    frame_q = queue.Queue(maxsize=1)
    result_q = queue.Queue()
    stop_event = threading.Event()

    ocr_thread = threading.Thread(target=ocr_worker, args=(frame_q, result_q, stop_event))
    ocr_thread.start()

    cap = cv2.VideoCapture(cameraIndex)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        stop_event.set()
        ocr_thread.join()
        exit()

    print("Camera opened. Starting main loop.")
    frame_count = 0
    plate_text = ""

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break

        frame = imutils.resize(frame, width=FRAME_WIDTH)

        try:
            plate_text = result_q.get_nowait()
        except queue.Empty:
            pass

        frame_count += 1
        if frame_count % OCR_EVERY_N_FRAMES == 0 and frame_q.empty():
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            h, w = gray.shape
            roi = gray[int(h * ROI_PERCENT_TOP):, :]
            frame_q.put(roi)

        if plate_text:
            cv2.putText(frame, f"Plate: {plate_text}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
            
            if plate_text in AUTHORIZED_PLATES:
                cv2.putText(frame, "ACCESS GRANTED", (10, 60), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

                Gate.set_angle(OPEN_POSITION, 5)
                Gate.set_angle(OPEN_POSITION, 0)
            else:
                cv2.putText(frame, "UNKNOWN PLATE", (10, 60), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        cv2.imshow("Fast LPR (Pi 5 Optimized)", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    print("Stopping application...")
    stop_event.set()
    ocr_thread.join()
    cap.release()
    cv2.destroyAllWindows()
