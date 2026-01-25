import os
import time
import threading
import queue
import struct
import csv
import numpy as np
from collections import deque, defaultdict
from datetime import datetime
import cv2
import mediapipe as mp
import dlib
import pvporcupine
import pyaudio
import psutil

# ---------------------
# === CONFIGURATION ===
# ---------------------
ACCESS_KEY = "9cunUiGZPhy31M1VUW5TBgQSc9Re48V5fXew+2V0KmjCDG/CkJo7Yw=="
KEYWORD_PATH = "/Users/tanishdasari/Downloads/Start_en_mac_v3_0_0/Start_en_mac_v3_0_0.ppn"
PREDICTOR_PATH = "/Users/tanishdasari/shape_predictor_68_face_landmarks.dat"
CAM_INDEX = 0
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
CONF_TARGET = 0.85
FACE_CHECK_INTERVAL_S = 30

VISION_LOG_CSV = "vision_metrics_log.csv"

# ---------------------
# === GLOBAL STATE ===
# ---------------------
VISION_ACTIVE = False

# ---------------------
# === Utilities ===
# ---------------------
class EWMA:
    def __init__(self, alpha=0.3):
        self.alpha = alpha
        self.mean = None
    def update(self, value):
        if self.mean is None:
            self.mean = value
        else:
            self.mean = self.alpha * value + (1 - self.alpha) * self.mean
        return self.mean

# ---------------------
# Frame Grabber
# ---------------------
class FrameGrabber(threading.Thread):
    def __init__(self, src=0, width=640, height=480, buffer_size=120):
        super().__init__(daemon=True)
        self.src = src
        self.width = width
        self.height = height
        self.buffer = deque(maxlen=buffer_size)
        self._stop = threading.Event()
        self.cap = None

    def run(self):
        self.cap = cv2.VideoCapture(self.src)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        while not self._stop.is_set():
            ret, frame = self.cap.read()
            if ret:
                self.buffer.append(frame.copy())
            time.sleep(0.001)

    def stop(self):
        self._stop.set()
        if self.cap:
            self.cap.release()

    def latest(self):
        return self.buffer[-1].copy() if self.buffer else None

# ---------------------
# Audio Wake Word
# ---------------------
class AudioWakeWord(threading.Thread):
    def __init__(self, access_key, keyword_path, wake_queue):
        super().__init__(daemon=True)
        self.access_key = access_key
        self.keyword_path = keyword_path
        self.wake_queue = wake_queue
        self.running = True

    def run(self):
        porcupine = pvporcupine.create(
            access_key=self.access_key,
            keyword_paths=[self.keyword_path]
        )
        pa = pyaudio.PyAudio()
        stream = pa.open(
            rate=porcupine.sample_rate,
            channels=1,
            format=pyaudio.paInt16,
            input=True,
            frames_per_buffer=porcupine.frame_length
        )

        while self.running:
            pcm = stream.read(porcupine.frame_length, exception_on_overflow=False)
            pcm = struct.unpack_from("h"*porcupine.frame_length, pcm)
            if porcupine.process(pcm) >= 0:
                self.wake_queue.put(("START", time.time()))

# ---------------------
# Model Wrappers
# ---------------------
mp_face_detection = mp.solutions.face_detection
mp_face_mesh = mp.solutions.face_mesh

class BlazeFaceWrapper:
    def __init__(self):
        self.model = mp_face_detection.FaceDetection(0, 0.5)
    def run(self, frame):
        t0 = time.time()
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = self.model.process(rgb)
        rt = (time.time()-t0)*1000
        if res.detections:
            return True, 0.9, rt
        return None, 0.0, rt

class DlibWrapper:
    def __init__(self):
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(PREDICTOR_PATH)
    def run(self, frame):
        t0 = time.time()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rects = self.detector(gray, 0)
        rt = (time.time()-t0)*1000
        if rects:
            shape = self.predictor(gray, rects[0])
            pts = np.array([[p.x,p.y] for p in shape.parts()])
            return {"gestures": compute_gestures_dlib(pts)}, 0.88, rt
        return None, 0.0, rt

class FaceMeshWrapper:
    def __init__(self):
        self.model = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True)
    def run(self, frame):
        t0 = time.time()
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = self.model.process(rgb)
        rt = (time.time()-t0)*1000
        if res.multi_face_landmarks:
            pts = np.array([[p.x,p.y] for p in res.multi_face_landmarks[0].landmark])
            return {"gestures": compute_gestures_facemesh(pts)}, 0.95, rt
        return None, 0.0, rt

# ---------------------
# Gesture Computation IMPORTANT: ACTUAL GESTURES WILL VERY BASED ON WEBCAM/COMPUTER.
# Changing these numbers based on webcam, but the accuracy should be SOMEWHAT irrelevant,
#   because all the metrics (other than accuracy) will be the exact same as currently measured
# ---------------------
def compute_gestures_dlib(lm):
    return {
        "mouth_open": float(np.linalg.norm(lm[62]-lm[66]) > 5),
        "smile": float(lm[48][1] < lm[51][1]),
        "left_eye_closed": float(np.linalg.norm(lm[37]-lm[41]) < 2),
        "right_eye_closed": float(np.linalg.norm(lm[43]-lm[47]) < 2)
    }

def compute_gestures_facemesh(lm):
    return {
        "mouth_open": float(np.linalg.norm(lm[13]-lm[17]) > 0.02),
        "smile": float(lm[61][1] < lm[0][1]),
        "left_eye_closed": float(np.linalg.norm(lm[386]-lm[374]) < 0.01),
        "right_eye_closed": float(np.linalg.norm(lm[159]-lm[145]) < 0.01)
    }

# ---------------------
# Cascade Scheduler
# ---------------------
class StagedCascadeScheduler:
    def __init__(self, bf, dlib, fm):
        self.bf = bf
        self.dlib = dlib
        self.fm = fm

    def process(self, frame):
        bf_res, bf_conf, bf_rt = self.bf.run(frame)
        dlib_res, dlib_conf, dlib_rt = self.dlib.run(frame)
        fm_res, fm_conf, fm_rt = self.fm.run(frame)

        # Stage decision
        if dlib_res and dlib_conf >= CONF_TARGET:
            return dlib_res, dlib_conf, "DlibWrapper", bf_rt, dlib_rt, fm_rt
        elif fm_res:
            return fm_res, fm_conf, "FaceMeshWrapper", bf_rt, dlib_rt, fm_rt
        else:
            return None, 0.0, "BlazeFaceWrapper", bf_rt, dlib_rt, fm_rt

# ---------------------
# Vision Session Loop
# ---------------------
def vision_session_loop(grabber, cascade):
    global VISION_ACTIVE
    VISION_ACTIVE = True

    # CSV logging setup
    process = psutil.Process()
    first_time = not os.path.exists(VISION_LOG_CSV)
    f = open(VISION_LOG_CSV, "a", newline="")
    writer = csv.writer(f)
    if first_time:
        writer.writerow(["Timestamp","ModelUsed","Conf","TotalTimeMs","BlazeFaceMs",
                         "DlibMs","FaceMeshMs","Gestures","CPU%","Mem%","RAM(MB)"])

    print("[Vision] Vision session started. Press 'q' to exit vision only.")

    try:
        while True:
            frame = grabber.latest()
            if frame is None:
                continue

            # Run cascade
            result, conf, model_used, bf_rt, dlib_rt, fm_rt = cascade.process(frame)

            # System metrics
            cpu = process.cpu_percent(interval=None)
            mem = process.memory_percent()
            ram = process.memory_info().rss / (1024**2)

            # Gestures string
            gestures = result["gestures"] if result else {}

            # CSV log
            writer.writerow([
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                model_used,
                conf,
                bf_rt + dlib_rt + fm_rt,
                bf_rt,
                dlib_rt,
                fm_rt,
                str(gestures),
                cpu,
                mem,
                round(ram,2)
            ])
            f.flush()

            # Display
            display = frame.copy()
            cv2.imshow("Vision", display)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                print("[Vision] Exiting vision session.")
                break

    finally:
        f.close()
        cv2.destroyAllWindows()
        VISION_ACTIVE = False

# ---------------------
# Main
# ---------------------
def main():
    grabber = FrameGrabber(CAM_INDEX, FRAME_WIDTH, FRAME_HEIGHT)
    grabber.start()

    wake_queue = queue.Queue()
    audio = AudioWakeWord(ACCESS_KEY, KEYWORD_PATH, wake_queue)
    audio.start()

    cascade = StagedCascadeScheduler(
        BlazeFaceWrapper(),
        DlibWrapper(),
        FaceMeshWrapper()
    )

    # Create hidden dummy window so waitKey works when vision inactive
    cv2.namedWindow("key_listener", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("key_listener", 1, 1)
    cv2.moveWindow("key_listener", -100, -100)

    print("[Main] Listening for wake word. Press 'q' to exit program.")

    try:
        while True:
            # ---- q pressed while vision inactive → exit program ----
            if not VISION_ACTIVE:
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    print("[Main] Exiting program (q pressed while vision inactive).")
                    break

            # Check wake queue
            try:
                msg, _ = wake_queue.get(timeout=0.1)
                if msg == "START":
                    vision_session_loop(grabber, cascade)
            except queue.Empty:
                pass

    finally:
        audio.running = False
        grabber.stop()
        cv2.destroyAllWindows()
        print("[Main] Program shutdown complete.")

if __name__ == "__main__":
    main()
