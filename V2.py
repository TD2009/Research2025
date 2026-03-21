import os
import time
import threading
import queue
import struct
import csv
import numpy as np
from collections import deque
from datetime import datetime
import cv2
import mediapipe as mp
import dlib
import pvporcupine
import pyaudio
import psutil

ACCESS_KEY = ""
KEYWORD_PATH = ""
PREDICTOR_PATH = ""
CAM_INDEX = 0
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
CONF_TARGET = 0.85
FACE_CHECK_INTERVAL_S = 30

VISION_LOG_CSV = "vision_metrics_log.csv"

EAR_CLOSED_THRESH  = 0.21
MAR_OPEN_THRESH    = 0.50
SMILE_RATIO_THRESH = 0.40

VISION_ACTIVE = False


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


def _ear(eye_pts):
    A = np.linalg.norm(eye_pts[1] - eye_pts[5])
    B = np.linalg.norm(eye_pts[2] - eye_pts[4])
    C = np.linalg.norm(eye_pts[0] - eye_pts[3])
    return (A + B) / (2.0 * C + 1e-6)


def _mar(mouth_pts):
    A = np.linalg.norm(mouth_pts[2]  - mouth_pts[10])
    B = np.linalg.norm(mouth_pts[4]  - mouth_pts[8])
    C = np.linalg.norm(mouth_pts[0]  - mouth_pts[6])
    return (A + B) / (2.0 * C + 1e-6)


def _conf_from_ratio(ratio, threshold, direction="above"):
    margin = abs(ratio - threshold) / (threshold + 1e-6)
    raw = min(margin, 1.0)
    if direction == "above":
        active = ratio > threshold
    else:
        active = ratio < threshold
    return round(0.5 + 0.5 * raw, 4) if active else round(0.5 - 0.5 * raw, 4)


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
            pcm = struct.unpack_from("h" * porcupine.frame_length, pcm)
            if porcupine.process(pcm) >= 0:
                self.wake_queue.put(("START", time.time()))


mp_face_detection = mp.solutions.face_detection
mp_face_mesh      = mp.solutions.face_mesh


class BlazeFaceWrapper:
    def __init__(self):
        self.model = mp_face_detection.FaceDetection(
            model_selection=0, min_detection_confidence=0.5
        )

    def run(self, frame):
        t0 = time.time()
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = self.model.process(rgb)
        rt = (time.time() - t0) * 1000
        if res.detections:
            score = max(d.score[0] for d in res.detections)
            return True, round(float(score), 4), rt
        return None, 0.0, rt


class DlibWrapper:
    def __init__(self):
        self.detector  = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(PREDICTOR_PATH)

    def run(self, frame):
        t0    = time.time()
        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rects = self.detector(gray, 0)
        rt    = (time.time() - t0) * 1000
        if not rects:
            return None, 0.0, rt
        shape = self.predictor(gray, rects[0])
        lm    = np.array([[p.x, p.y] for p in shape.parts()])
        gestures, confidence = compute_gestures_dlib(lm)
        return {"gestures": gestures}, confidence, rt


class FaceMeshWrapper:
    def __init__(self):
        self.model = mp_face_mesh.FaceMesh(
            max_num_faces=1, refine_landmarks=True
        )

    def run(self, frame):
        t0  = time.time()
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = self.model.process(rgb)
        rt  = (time.time() - t0) * 1000
        if not res.multi_face_landmarks:
            return None, 0.0, rt
        lm = np.array(
            [[p.x, p.y] for p in res.multi_face_landmarks[0].landmark]
        )
        gestures, confidence = compute_gestures_facemesh(lm)
        return {"gestures": gestures}, confidence, rt


def compute_gestures_dlib(lm):
    right_eye = lm[36:42]
    left_eye  = lm[42:48]
    r_ear = _ear(right_eye)
    l_ear = _ear(left_eye)

    mouth_outer = lm[48:60]
    mar = _mar(mouth_outer)

    mouth_width  = np.linalg.norm(lm[54] - lm[48])
    inter_eye    = np.linalg.norm(lm[39] - lm[42])
    smile_ratio  = mouth_width / (inter_eye + 1e-6)

    r_brow_height = np.mean(lm[17:22, 1])
    l_brow_height = np.mean(lm[22:27, 1])
    r_eye_height  = np.mean(lm[36:42, 1])
    l_eye_height  = np.mean(lm[42:48, 1])
    face_height   = np.linalg.norm(lm[8] - lm[27])
    r_brow_raise  = (r_eye_height - r_brow_height) / (face_height + 1e-6)
    l_brow_raise  = (l_eye_height - l_brow_height) / (face_height + 1e-6)
    BROW_RAISE_THRESH = 0.18

    gestures = {
        "right_eye_closed": float(r_ear < EAR_CLOSED_THRESH),
        "left_eye_closed":  float(l_ear < EAR_CLOSED_THRESH),
        "mouth_open":       float(mar   > MAR_OPEN_THRESH),
        "smile":            float(smile_ratio > SMILE_RATIO_THRESH),
        "right_brow_raise": float(r_brow_raise > BROW_RAISE_THRESH),
        "left_brow_raise":  float(l_brow_raise > BROW_RAISE_THRESH),
        "_r_ear": round(r_ear, 4),
        "_l_ear": round(l_ear, 4),
        "_mar":   round(mar,   4),
    }

    confs = [
        _conf_from_ratio(r_ear,        EAR_CLOSED_THRESH,  "below"),
        _conf_from_ratio(l_ear,        EAR_CLOSED_THRESH,  "below"),
        _conf_from_ratio(mar,          MAR_OPEN_THRESH,    "above"),
        _conf_from_ratio(smile_ratio,  SMILE_RATIO_THRESH, "above"),
        _conf_from_ratio(r_brow_raise, BROW_RAISE_THRESH,  "above"),
        _conf_from_ratio(l_brow_raise, BROW_RAISE_THRESH,  "above"),
    ]
    confidence = round(float(np.mean(confs)), 4)
    return gestures, confidence


def compute_gestures_facemesh(lm):
    left_eye_idx  = [362, 385, 387, 263, 373, 380]
    right_eye_idx = [33,  160, 158, 133, 153, 144]
    l_ear = _ear(lm[left_eye_idx])
    r_ear = _ear(lm[right_eye_idx])
    FM_EAR_THRESH = 0.20

    mouth_idx = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 375]
    mar = _mar(lm[mouth_idx])
    FM_MAR_THRESH = 0.45

    mouth_width  = np.linalg.norm(lm[61] - lm[291])
    inter_eye    = np.linalg.norm(lm[133] - lm[362])
    smile_ratio  = mouth_width / (inter_eye + 1e-6)
    FM_SMILE_THRESH = 0.85

    r_brow_y     = np.mean(lm[[70, 63, 105, 66, 107], 1])
    l_brow_y     = np.mean(lm[[336, 296, 334, 293, 300], 1])
    r_eye_y      = np.mean(lm[right_eye_idx, 1])
    l_eye_y      = np.mean(lm[left_eye_idx,  1])
    face_height  = abs(float(lm[152, 1]) - float(lm[6, 1]))
    r_brow_raise = (r_eye_y - r_brow_y) / (face_height + 1e-6)
    l_brow_raise = (l_eye_y - l_brow_y) / (face_height + 1e-6)
    FM_BROW_THRESH = 0.12

    nose_y       = float(lm[1, 1])
    cheek_mean_y = float(np.mean([lm[234, 1], lm[454, 1]]))
    nod_ratio    = (nose_y - cheek_mean_y) / (face_height + 1e-6)
    FM_NOD_DOWN_THRESH =  0.10
    FM_NOD_UP_THRESH   = -0.10

    gestures = {
        "right_eye_closed": float(r_ear < FM_EAR_THRESH),
        "left_eye_closed":  float(l_ear < FM_EAR_THRESH),
        "mouth_open":       float(mar   > FM_MAR_THRESH),
        "smile":            float(smile_ratio > FM_SMILE_THRESH),
        "right_brow_raise": float(r_brow_raise > FM_BROW_THRESH),
        "left_brow_raise":  float(l_brow_raise > FM_BROW_THRESH),
        "head_nod_down":    float(nod_ratio >  FM_NOD_DOWN_THRESH),
        "head_nod_up":      float(nod_ratio <  FM_NOD_UP_THRESH),
        "_r_ear":     round(r_ear,      4),
        "_l_ear":     round(l_ear,      4),
        "_mar":       round(mar,        4),
        "_nod_ratio": round(nod_ratio,  4),
    }

    confs = [
        _conf_from_ratio(r_ear,        FM_EAR_THRESH,   "below"),
        _conf_from_ratio(l_ear,        FM_EAR_THRESH,   "below"),
        _conf_from_ratio(mar,          FM_MAR_THRESH,   "above"),
        _conf_from_ratio(smile_ratio,  FM_SMILE_THRESH, "above"),
        _conf_from_ratio(r_brow_raise, FM_BROW_THRESH,  "above"),
        _conf_from_ratio(l_brow_raise, FM_BROW_THRESH,  "above"),
    ]
    confidence = round(float(np.mean(confs)), 4)
    return gestures, confidence


class StagedCascadeScheduler:
    def __init__(self, bf, dlib_w, fm):
        self.bf   = bf
        self.dlib = dlib_w
        self.fm   = fm

    def process(self, frame):
        bf_res, bf_conf, bf_rt = self.bf.run(frame)
        if not bf_res:
            return None, 0.0, "BlazeFaceWrapper", bf_rt, 0, 0

        dlib_res, dlib_conf, dlib_rt = self.dlib.run(frame)
        if dlib_res and dlib_conf >= CONF_TARGET:
            return dlib_res, dlib_conf, "DlibWrapper", bf_rt, dlib_rt, 0

        fm_res, fm_conf, fm_rt = self.fm.run(frame)
        if fm_res:
            return fm_res, fm_conf, "FaceMeshWrapper", bf_rt, dlib_rt, fm_rt

        return None, 0.0, "FaceMeshWrapper", bf_rt, dlib_rt, fm_rt


def vision_session_loop(grabber, cascade):
    global VISION_ACTIVE
    VISION_ACTIVE = True

    proc       = psutil.Process()
    first_time = not os.path.exists(VISION_LOG_CSV)
    f          = open(VISION_LOG_CSV, "a", newline="")
    writer     = csv.writer(f)

    if first_time:
        writer.writerow([
            "Timestamp", "ModelUsed", "Conf", "TotalTimeMs",
            "BlazeFaceMs", "DlibMs", "FaceMeshMs",
            "Gestures", "CPU%", "Mem%", "RAM(MB)"
        ])

    print("[Vision] Vision session started. Press 'q' to exit vision only.")

    try:
        while True:
            frame = grabber.latest()
            if frame is None:
                continue

            result, conf, model_used, bf_rt, dlib_rt, fm_rt = cascade.process(frame)

            cpu = proc.cpu_percent(interval=None)
            mem = proc.memory_percent()
            ram = proc.memory_info().rss / (1024 ** 2)

            gestures = result["gestures"] if result else {}

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
                round(ram, 2)
            ])
            f.flush()

            cv2.imshow("Vision", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("[Vision] Exiting vision session.")
                break

    finally:
        f.close()
        cv2.destroyAllWindows()
        VISION_ACTIVE = False


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

    cv2.namedWindow("key_listener", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("key_listener", 1, 1)
    cv2.moveWindow("key_listener", -100, -100)

    print("[Main] Listening for wake word. Press 'q' to exit program.")

    try:
        while True:
            if not VISION_ACTIVE:
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    print("[Main] Exiting program.")
                    break
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
        print("[Main] Shutdown complete.")


if __name__ == "__main__":
    main()
