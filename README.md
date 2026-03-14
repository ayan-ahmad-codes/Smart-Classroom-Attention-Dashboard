# 🎓 Smart Classroom Attention Detector

A real-time Python application that uses your laptop webcam, MediaPipe Face Mesh, and OpenCV to detect whether you are paying attention to the screen.

---

## ✨ Features

| Feature | Details |
|---|---|
| Eye Tracking | Detects iris position and maps gaze: Center / Left / Right / Down |
| Blink Detection | Eye Aspect Ratio (EAR) algorithm |
| Head-Pose Estimation | Classifies Forward / Left / Right / Down |
| Attention Score | 0–100 % (70% gaze + 20% blink rate + 10% head pose) |
| Status Label | **Attentive** / **Distracted** / **Sleepy** |
| Alert System | On-screen warning after 5 s of low attention |
| Data Logging | Appends a row to `attention_log.csv` every 5 seconds |

---

## 🗂️ Project Structure

```
attention_detector/
│
├── main.py            ← Entry point – run this
├── eye_tracker.py     ← Face mesh, EAR, gaze, head pose
├── attention_logic.py ← Score calculation & alert logic
├── utils.py           ← HUD drawing & CSV logger
├── requirements.txt   ← Python dependencies
└── README.md          ← This file
```

---

## 🛠️ Installation

### Step 1 – Make sure Python 3.9+ is installed
```bash
python --version     # should say 3.9 or higher
```

### Step 2 – (Recommended) Create a virtual environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### Step 3 – Install dependencies
```bash
pip install -r requirements.txt
```
This installs:
- **opencv-python** – webcam capture & image drawing
- **mediapipe** – Face Mesh (468 facial landmarks + iris)
- **numpy** – fast numerical calculations

---

## ▶️ Running the App

```bash
cd attention_detector
python main.py
```

A window titled **"Attention Detector"** will open showing your webcam feed with:
- Green/yellow/red attention score bar
- Gaze direction & head pose labels
- Blink rate per minute
- Colour-coded status text

### Keyboard shortcuts
| Key | Action |
|-----|--------|
| `Q` or `ESC` | Quit |
| `R` | Reset blink counter |
| `S` | Save screenshot |

---

## 📊 Reading the CSV Log

After running, open `attention_log.csv` in Excel or any text editor.  
Columns: `timestamp, attention_score, status, gaze_direction, head_pose, blink_rate_per_min`

Example rows:
```
2024-09-01 10:05:30,82,Attentive,Center,Forward,14.3
2024-09-01 10:05:35,45,Distracted,Right,Forward,17.0
2024-09-01 10:05:40,18,Sleepy,Down,Down,28.6
```

---

## 🔧 Troubleshooting

| Problem | Fix |
|---------|-----|
| Camera not opening | Change `CAMERA_INDEX = 1` (or 2) in `main.py` |
| Very low FPS | Close other apps; reduce `FRAME_WIDTH/HEIGHT` in `main.py` |
| Blinks not detected | Lower `EAR_THRESHOLD` to `0.18` in `main.py` |
| Too many false blinks | Raise `CONSEC_FRAMES` to `4` or `5` |
| No face detected | Improve lighting; move closer to camera |
| mediapipe install fails | Try `pip install mediapipe==0.10.9` |

---

## 🧠 How the Attention Score Works

```
Attention Score = (Gaze Score × 0.70)
               + (Blink Score × 0.20)
               + (Head Score  × 0.10)
```

### Gaze Score
| Direction | Score |
|-----------|-------|
| Center    | 100%  |
| Left/Right| 30%   |
| Down      | 20%   |

### Blink Score
- Normal rate (~15 blinks/min) → full score
- Very high rate (>25/min) or prolonged eye closure → low score

### Head Score
| Pose    | Score |
|---------|-------|
| Forward | 100%  |
| Left/Right | 40% |
| Down    | 25%   |

---

## 📝 License
MIT – free to use and modify for personal or educational projects.
