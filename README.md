# 😴 DrowsinessDetector


**First time only:** Right-click run.command → Open → Open anyway
After that, double-click works normally.

**Real-time drowsiness detection that screams at you before you fall asleep at your desk.**

Built with Python, OpenCV, and MediaPipe. Runs locally on your machine — no internet required, no data sent anywhere.

---

## 🎯 What it does

- 👁️ **Eye closure detection** — triggers alarm if eyes stay closed too long (EAR algorithm)
- 😮 **Yawn detection** — catches early drowsiness before your eyes even close
- 😴 **Head nodding detection** — detects your head drooping forward
- 🔊 **Escalating alarm** — starts soft, gets LOUD the longer you're drowsy
- ⏱️ **90-min break reminder** — nudges you to take a break after 90 minutes
- 📝 **Drowsiness log** — saves every event with timestamp to `drowsiness_log.txt`
- 🖥️ **Live HUD** — real-time overlay showing EAR/MAR bars, session timer, alert count

---

## 🚀 Quick Start

### Option A — Run from source (recommended)

**1. Clone the repo**
```bash
git clone https://github.com/YOUR_USERNAME/drowsiness-detector.git
cd drowsiness-detector
```

**2. Install dependencies**
```bash
pip install opencv-python mediapipe==0.10.13 pygame numpy
```

**3. Run**
```bash
python drowsiness_detector.py
```

### Option B — Download the app (macOS)
Download the latest `.app` from [Releases](../../releases) — no Python needed.

---

## ⌨️ Controls

| Key | Action |
|-----|--------|
| `Q` | Quit |
| `R` | Reset session timer |

---

## ⚙️ Tuning (edit the CONFIG section in the script)

```python
EAR_THRESHOLD      = 0.20   # lower = less sensitive to eye closure
DROWSY_EAR_SECONDS = 10.0   # seconds before alarm triggers
YAWN_THRESHOLD     = 0.6    # higher = less sensitive to yawning
BREAK_REMINDER_MIN = 90     # minutes before break reminder
```

---

## 🛠️ How it works

**EAR (Eye Aspect Ratio)** — a formula that measures eye openness using 6 facial landmarks per eye:

```
EAR = (|p2−p6| + |p3−p5|) / (2 × |p1−p4|)
```

Open eye ≈ 0.30 | Closed eye ≈ 0.0 | Threshold: 0.20

**MAR (Mouth Aspect Ratio)** — same idea for yawning, measures vertical vs horizontal mouth opening.

**Head nod** — calibrates your upright head position on startup, then detects forward pitch using nose/chin/ear landmarks.

All detection runs locally on CPU using [MediaPipe Face Mesh](https://developers.google.com/mediapipe/solutions/vision/face_landmarker) (468 landmarks).

---

## 📦 Requirements

- Python 3.9+
- Webcam
- macOS / Windows / Linux

```
opencv-python
mediapipe==0.10.13
pygame
numpy
```

---

## 🗺️ Roadmap

- [ ] Windows build
- [ ] Linux build  
- [ ] Web version (browser-based, no install)
- [ ] Mobile app (iOS/Android)
- [ ] Custom alarm sound support
- [ ] Dark/light mode HUD
- [ ] Stats dashboard (weekly drowsiness report)

---

## 🤝 Contributing

PRs welcome. If the thresholds don't work well for you, open an issue with your lighting conditions and I'll help tune it.

---

## 📄 License

MIT — free to use, modify, and distribute.

---

## 👤 Author

Built by RON HARVARD — a 16-year-old developer from India.  
Inspired by the problem of falling asleep while studying for JEE 💀

> *"If I'm going to suffer through exam prep, at least my computer should suffer with me."*

