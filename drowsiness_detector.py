"""
drowsiness_detector.py
======================
Real-time drowsiness detection using webcam.
Detects: eye closure (EAR), yawning, head nodding.
Plays escalating alarm when drowsiness is detected.

Requirements:
    pip install opencv-python mediapipe pygame numpy

Run:
    python drowsiness_detector.py
"""

import cv2
import mediapipe as mp
import numpy as np
import pygame
import time
import math
from collections import deque

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
EAR_THRESHOLD       = 0.20   # below this = eye closed
EAR_CONSEC_FRAMES   = 6      # frames eye must be closed to count (at ~20fps ≈ 0.3s)
DROWSY_EAR_SECONDS  = 10.0    # seconds of closed eyes → ALARM
YAWN_THRESHOLD      = 0.6    # mouth aspect ratio above this = yawn
YAWN_CONSEC_SECONDS = 2.0    # seconds of yawning → warning
NOD_THRESHOLD       = 20     # degrees of head pitch down = nod
NOD_CONSEC_SECONDS  = 1.5
BREAK_REMINDER_MIN  = 90     # remind break after X minutes of session
LOG_FILE            = "drowsiness_log.txt"

# ─────────────────────────────────────────────
# MEDIAPIPE LANDMARK INDICES
# ─────────────────────────────────────────────
# Eye landmarks (from MediaPipe 468-point face mesh)
LEFT_EYE  = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33,  160, 158, 133, 153, 144]

# Mouth landmarks for yawn detection
MOUTH_TOP    = 13
MOUTH_BOTTOM = 14
MOUTH_LEFT   = 78
MOUTH_RIGHT  = 308

# Nose tip & chin for head pose
NOSE_TIP  = 1
CHIN      = 199
LEFT_EAR_POINT  = 234
RIGHT_EAR_POINT = 454

# ─────────────────────────────────────────────
# ALARM SYSTEM
# ─────────────────────────────────────────────
pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=512)

def generate_beep(frequency=880, duration_ms=300, volume=0.5):
    """Generate a beep sound programmatically (no audio file needed)."""
    sample_rate = 44100
    n_samples = int(sample_rate * duration_ms / 1000)
    t = np.linspace(0, duration_ms / 1000, n_samples, False)
    wave = np.sin(2 * np.pi * frequency * t)
    # Apply fade in/out to avoid clicks
    fade = 50
    wave[:fade] *= np.linspace(0, 1, fade)
    wave[-fade:] *= np.linspace(1, 0, fade)
    wave = (wave * volume * 32767).astype(np.int16)
    # Stereo
    stereo = np.column_stack([wave, wave])
    sound = pygame.sndarray.make_sound(stereo)
    return sound

# Pre-generate sounds at different volumes
pygame.mixer.set_num_channels(4)
BEEP_SOFT   = generate_beep(660,  200, 0.3)
BEEP_MED    = generate_beep(880,  300, 0.6)
BEEP_LOUD   = generate_beep(1100, 400, 1.0)
BEEP_YAWN   = generate_beep(440,  150, 0.4)

def play_alarm(level=1):
    """level 1=soft, 2=medium, 3=LOUD"""
    pygame.mixer.stop()
    if level == 1:
        BEEP_SOFT.play()
    elif level == 2:
        BEEP_MED.play()
    else:
        BEEP_LOUD.play()
        # repeat loud 3 times
        pygame.time.delay(450)
        BEEP_LOUD.play()
        pygame.time.delay(450)
        BEEP_LOUD.play()

# ─────────────────────────────────────────────
# MATH HELPERS
# ─────────────────────────────────────────────
def euclidean(p1, p2):
    return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

def eye_aspect_ratio(landmarks, eye_indices, w, h):
    """
    EAR = (|p2-p6| + |p3-p5|) / (2 * |p1-p4|)
    A fully open eye ≈ 0.3, closed eye ≈ 0.0
    """
    pts = [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in eye_indices]
    A = euclidean(pts[1], pts[5])
    B = euclidean(pts[2], pts[4])
    C = euclidean(pts[0], pts[3])
    return (A + B) / (2.0 * C) if C > 0 else 0

def mouth_aspect_ratio(landmarks, w, h):
    """MAR: vertical / horizontal mouth opening ratio."""
    top    = (landmarks[MOUTH_TOP].x * w,    landmarks[MOUTH_TOP].y * h)
    bottom = (landmarks[MOUTH_BOTTOM].x * w, landmarks[MOUTH_BOTTOM].y * h)
    left   = (landmarks[MOUTH_LEFT].x * w,   landmarks[MOUTH_LEFT].y * h)
    right  = (landmarks[MOUTH_RIGHT].x * w,  landmarks[MOUTH_RIGHT].y * h)
    vertical   = euclidean(top, bottom)
    horizontal = euclidean(left, right)
    return vertical / horizontal if horizontal > 0 else 0

def head_pitch_angle(landmarks, w, h):
    """Estimate head pitch (nodding) from nose tip and chin vertical distance."""
    nose = (landmarks[NOSE_TIP].x * w,  landmarks[NOSE_TIP].y * h)
    chin = (landmarks[CHIN].x * w,      landmarks[CHIN].y * h)
    l_ear = (landmarks[LEFT_EAR_POINT].x * w,  landmarks[LEFT_EAR_POINT].y * h)
    r_ear = (landmarks[RIGHT_EAR_POINT].x * w, landmarks[RIGHT_EAR_POINT].y * h)
    # Face height reference
    face_height = euclidean(nose, chin)
    # Midpoint between ears = face center top
    ear_mid = ((l_ear[0]+r_ear[0])/2, (l_ear[1]+r_ear[1])/2)
    # Angle of chin relative to ear midpoint
    dy = chin[1] - ear_mid[1]
    dx = chin[0] - ear_mid[0]
    angle = math.degrees(math.atan2(dy, dx))
    return angle  # ~90° = upright, drops when nodding

# ─────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────
def log_event(event_type, detail=""):
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(LOG_FILE, "a") as f:
        f.write(f"[{timestamp}] {event_type}: {detail}\n")
    print(f"  📝 Logged: {event_type}")

# ─────────────────────────────────────────────
# HUD DRAWING
# ─────────────────────────────────────────────
def draw_bar(frame, x, y, value, max_val, label, color, w=150, h=12):
    """Draw a small progress bar."""
    cv2.rectangle(frame, (x, y), (x+w, y+h), (50,50,50), -1)
    fill = int(min(value / max_val, 1.0) * w)
    cv2.rectangle(frame, (x, y), (x+fill, y+h), color, -1)
    cv2.putText(frame, label, (x, y-4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200,200,200), 1)

def draw_hud(frame, state):
    h, w = frame.shape[:2]
    # Semi-transparent overlay panel
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (300, 240), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    # Title
    cv2.putText(frame, "DROWSINESS DETECTOR", (18, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 180), 2)

    # EAR bar
    ear_color = (0,200,100) if state['ear'] > EAR_THRESHOLD else (0,80,255)
    draw_bar(frame, 18, 55,  state['ear'],  0.35, f"EAR: {state['ear']:.2f}", ear_color)

    # MAR bar
    mar_color = (0,200,100) if state['mar'] < YAWN_THRESHOLD else (0,180,255)
    draw_bar(frame, 18, 85,  state['mar'],  1.0,  f"MAR: {state['mar']:.2f}", mar_color)

    # Head pitch
    pitch_norm = max(0, state['pitch'] - 60) / 30
    draw_bar(frame, 18, 115, pitch_norm,    1.0,  f"Nod: {state['pitch']:.0f}°", (180,100,255))

    # Session time
    mins = int(state['session_secs'] // 60)
    secs = int(state['session_secs'] % 60)
    cv2.putText(frame, f"Session: {mins:02d}:{secs:02d}", (18, 155),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

    # Alert counter
    cv2.putText(frame, f"Alerts: {state['alert_count']}", (18, 175),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

    # Status
    status_text  = state['status']
    status_color = state['status_color']
    cv2.putText(frame, status_text, (18, 210),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)

    # Controls hint
    cv2.putText(frame, "Q: quit  |  R: reset session", (10, h-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (120,120,120), 1)

def draw_eye_contours(frame, landmarks, w, h, ear):
    color = (0, 255, 120) if ear > EAR_THRESHOLD else (0, 60, 255)
    for eye in [LEFT_EYE, RIGHT_EYE]:
        pts = np.array([(int(landmarks[i].x*w), int(landmarks[i].y*h)) for i in eye], np.int32)
        cv2.polylines(frame, [pts], True, color, 1)

# ─────────────────────────────────────────────
# MAIN LOOP
# ─────────────────────────────────────────────
def main():
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Cannot open webcam. Check your camera.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # ── State ──
    eye_closed_start  = None
    yawn_start        = None
    nod_start         = None
    last_alarm_time   = 0
    alarm_cooldown    = 3.0   # seconds between alarms
    alert_count       = 0
    session_start     = time.time()
    break_notified    = False
    baseline_pitch    = None  # calibrated upright pitch
    pitch_history     = deque(maxlen=30)

    state = {
        'ear': 0.3, 'mar': 0.0, 'pitch': 90.0,
        'status': 'MONITORING', 'status_color': (0,255,120),
        'session_secs': 0, 'alert_count': 0
    }

    print("✅ Drowsiness Detector running. Press Q to quit.")
    log_event("SESSION_START")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)

        now = time.time()
        state['session_secs'] = now - session_start

        # Break reminder
        if not break_notified and state['session_secs'] >= BREAK_REMINDER_MIN * 60:
            log_event("BREAK_REMINDER", f"After {BREAK_REMINDER_MIN} mins")
            play_alarm(1)
            break_notified = True

        if results.multi_face_landmarks:
            lm = results.multi_face_landmarks[0].landmark

            # ── EAR ──
            left_ear  = eye_aspect_ratio(lm, LEFT_EYE,  w, h)
            right_ear = eye_aspect_ratio(lm, RIGHT_EYE, w, h)
            ear = (left_ear + right_ear) / 2.0
            state['ear'] = ear

            # ── MAR ──
            mar = mouth_aspect_ratio(lm, w, h)
            state['mar'] = mar

            # ── Head pitch ──
            pitch = head_pitch_angle(lm, w, h)
            pitch_history.append(pitch)
            smooth_pitch = np.mean(pitch_history)
            state['pitch'] = smooth_pitch

            # Calibrate baseline in first 2 seconds
            if baseline_pitch is None and state['session_secs'] > 2:
                baseline_pitch = smooth_pitch
                print(f"  📐 Baseline pitch calibrated: {baseline_pitch:.1f}°")

            draw_eye_contours(frame, lm, w, h, ear)

            # ── DROWSINESS CHECK (eyes) ──
            status_set = False
            if ear < EAR_THRESHOLD:
                if eye_closed_start is None:
                    eye_closed_start = now
                closed_duration = now - eye_closed_start

                if closed_duration >= DROWSY_EAR_SECONDS:
                    state['status'] = f'⚠ DROWSY! {closed_duration:.1f}s'
                    state['status_color'] = (0, 0, 255)
                    status_set = True
                    if now - last_alarm_time > alarm_cooldown:
                        level = 3 if closed_duration > 4 else (2 if closed_duration > 2.5 else 1)
                        play_alarm(level)
                        last_alarm_time = now
                        alert_count += 1
                        state['alert_count'] = alert_count
                        log_event("DROWSY_EYES", f"{closed_duration:.1f}s closed, alarm level {level}")
                elif closed_duration > 0.5:
                    state['status'] = f'Eyes closing... {closed_duration:.1f}s'
                    state['status_color'] = (0, 165, 255)
                    status_set = True
            else:
                eye_closed_start = None

            # ── YAWN CHECK ──
            if mar > YAWN_THRESHOLD:
                if yawn_start is None:
                    yawn_start = now
                yawn_duration = now - yawn_start
                if yawn_duration >= YAWN_CONSEC_SECONDS and not status_set:
                    state['status'] = f'😮 YAWNING {yawn_duration:.1f}s'
                    state['status_color'] = (0, 165, 255)
                    status_set = True
                    if now - last_alarm_time > alarm_cooldown:
                        BEEP_YAWN.play()
                        last_alarm_time = now
                        alert_count += 1
                        state['alert_count'] = alert_count
                        log_event("YAWN", f"{yawn_duration:.1f}s")
            else:
                yawn_start = None

            # ── HEAD NOD CHECK ──
            if baseline_pitch is not None:
                pitch_drop = smooth_pitch - baseline_pitch  # positive = dropped forward
                if pitch_drop > NOD_THRESHOLD:
                    if nod_start is None:
                        nod_start = now
                    nod_duration = now - nod_start
                    if nod_duration >= NOD_CONSEC_SECONDS and not status_set:
                        state['status'] = f'😴 HEAD NOD {nod_duration:.1f}s'
                        state['status_color'] = (0, 80, 255)
                        status_set = True
                        if now - last_alarm_time > alarm_cooldown:
                            play_alarm(2)
                            last_alarm_time = now
                            alert_count += 1
                            state['alert_count'] = alert_count
                            log_event("HEAD_NOD", f"{nod_duration:.1f}s")
                else:
                    nod_start = None

            if not status_set:
                state['status'] = 'ALERT ✓'
                state['status_color'] = (0, 255, 120)

        else:
            state['status'] = 'No face detected'
            state['status_color'] = (100, 100, 100)
            eye_closed_start = None
            yawn_start = None
            nod_start = None

        draw_hud(frame, state)
        cv2.imshow("Drowsiness Detector", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            session_start = time.time()
            break_notified = False
            alert_count = 0
            state['alert_count'] = 0
            baseline_pitch = None
            print("  🔄 Session reset.")
            log_event("SESSION_RESET")

    cap.release()
    cv2.destroyAllWindows()
    log_event("SESSION_END", f"Total alerts: {alert_count}")
    print(f"\n✅ Done. {alert_count} alerts logged to {LOG_FILE}")

if __name__ == "__main__":
    main()