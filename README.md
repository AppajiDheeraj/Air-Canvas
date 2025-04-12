# 🎨 Air Canvas - Draw with Hand Gestures

An advanced **Air Canvas** app that lets you **draw in the air** using your hand and a webcam!  
Built using **OpenCV**, **MediaPipe**, and **Python**, this tool brings a real-world drawing experience with gesture-based controls and multiple drawing modes.

---

![Demo](primary_aircanvas-ezgif.com-optimize)

---

## 🎮 Features

- ✋ Real-time hand tracking using MediaPipe
- 🎨 Multiple drawing tools:
  - Freehand drawing
  - Line tool
  - Rectangle tool
  - Circle tool
- 🌈 Color palette selection
- 🧽 Eraser tool
- 🗑️ Clear screen option
- 🖼️ Smooth UI with live feedback

---

## 🧰 Tech Stack

- **Python**
- **OpenCV**
- **MediaPipe** (for hand detection and tracking)
- **NumPy**

---

## 📂 Folder Structure
📄 air_canvas.py — Main application script <br>
📄 README.md — You are here! <br>

---

## 🚀 How to Run

1. **Clone this repository** or download the files.
2. Install dependencies:

```bash
pip install opencv-python mediapipe numpy
```
3. Run the app:

```bash
python air_canvas.py
```
---

## ✋ Hand Gestures

| Mode               | Gesture                          | Action                           |
|--------------------|-----------------------------------|-----------------------------------|
| **✍️ Drawing Mode**   | Index finger up                   | ✏️ Draw on canvas                  |
| **🖐️ Selection Mode** | Index + Middle fingers up         | 🎨 Select color/tool               |
| **🤏 Shape Complete** | Thumb + Index finger pinch        | ✅ Complete shape (line, rect, etc)|
| **🧽 Eraser Mode**    | Select Eraser icon from toolbar   | 🧼 Erase with thick stroke         |
| **↩️ Undo**           | Select Undo icon from toolbar     | 🔙 Undo last stroke                |
| **🧹 Clear Canvas**   | Select Clear icon from toolbar    | 🗑️ Erase everything                |

---

## 🖼️ UI Overview <br>
The top of the screen contains tool and color icons <br>

Drawing happens below the header area <br>

Shapes are previewed live and finalized on pinch <br>

---

## 📸 Screenshots
<!-- Add some screenshots or gif demos here if you'd like -->
<br>

---

## 🔮 Possible Enhancements
✨ Add gesture for saving canvas as an image <br>

🎤 Integrate voice commands to change tools <br>

🌐 Build a web-based version using TensorFlow.js and WebRTC <br>

🖼️ Add dynamic brush textures or gradients <br>

---

## 🤝 Credits <br>
MediaPipe by Google <br>

OpenCV <br>

---
