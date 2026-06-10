# 🚀 Quick Start Guide - SLRS

Get started with the Sign Language Recognition System in 5 minutes!

---

## ⚡ Prerequisites

- Python 3.7 or higher
- Webcam (built-in or USB)
- 4GB RAM minimum (8GB recommended)

---

## 📥 Step 1: Installation

```bash
# Clone the repository
git clone https://github.com/trabelssi/-Sign-Language-Recognition-System-SLRS-.git
cd -Sign-Language-Recognition-System-SLRS-

# Install dependencies
pip install -r requirements.txt
```

---

## ▶️ Step 2: Run the Application

### Option A: Silent Mode (Recommended for first test)
```bash
python "CLASS WITHOUT AUDIO.py"
```

### Option B: With Audio Feedback
```bash
python "CLASS WITH AUDIO.py"
```

---

## 🎯 Step 3: Use the System

1. **Position yourself** 0.5-4 meters from the camera
2. **Make a gesture** from these 6 options:
   - ✋ **Palm** - Open hand, palm facing camera
   - ✌️ **Peace** - Two fingers in V shape
   - 🛑 **Stop** - Hand up, palm facing forward
   - 👌 **OK** - Thumb and index finger touching
   - 📞 **Call** - Phone call gesture
   - 🤫 **Mute** - Finger to lips

3. **See results** displayed on screen with confidence percentage
4. **Press 'q'** to quit

---

## 📊 Expected Results

- **Accuracy**: ~95% in good lighting
- **Speed**: 5-10 FPS on CPU, 30+ FPS on GPU
- **Confidence**: Results shown when >75% confident
- **Audio**: Announces when >90% confident (audio mode only)

---

## ⚠️ Troubleshooting

### Camera Not Working
```python
# Check camera index (try 0, 1, or 2)
cap = cv2.VideoCapture(0)  # Change 0 to 1 or 2
```

### Low FPS
- Reduce display frequency (edit `displayFreq = 5` to higher number)
- Use silent mode instead of audio mode
- Close other applications

### Model Not Loading
```bash
# Verify model file exists
ls -la model.h5  # Linux/Mac
dir model.h5     # Windows

# Should see: 245,053,600 bytes
```

### Import Errors
```bash
# Reinstall dependencies
pip install --upgrade -r requirements.txt
```

---

## 💡 Tips for Best Performance

1. **Lighting**: Use good indoor lighting, avoid backlighting
2. **Distance**: Stay 1-2 meters from camera for best results
3. **Background**: Use plain backgrounds when possible
4. **Gesture**: Hold gesture steady for 1-2 seconds
5. **Position**: Keep hand in center of camera frame

---

## 🎓 Understanding the Output

```
CLASS: peace
PROBABILITY: 98.45%
FPS: 25.3
```

- **CLASS**: Detected gesture name
- **PROBABILITY**: Model confidence (75-100%)
- **FPS**: Frames processed per second

---

## 🔧 Quick Configuration

Edit these values at the top of the Python files:

```python
frameSize = (320, 240)    # Camera resolution
brightness = 180          # Camera brightness (0-255)
threshold = 0.75          # Confidence threshold (0.75 = 75%)
displayFreq = 5           # Process every Nth frame
```

---

## 📱 Hardware Deployment

### Raspberry Pi 4
See [README.md](README.md) section "Hardware Implementation" for:
- Complete setup guide with images
- Network configuration
- VNC/SSH setup
- Performance optimization

---

## 📚 Learn More

- **Full Documentation**: [README.md](README.md)
- **Model Details**: [MODEL_INFO.md](MODEL_INFO.md)
- **Training Notebook**: `model_training_notebook.ipynb`

---

## 🐛 Still Having Issues?

1. Check Python version: `python --version` (need 3.7+)
2. Check TensorFlow: `python -c "import tensorflow; print(tensorflow.__version__)"`
3. Check camera: `python -c "import cv2; print(cv2.VideoCapture(0).read()[0])"`
4. Open an issue on GitHub with error details

---

## ✅ You're Ready!

Your system should now be recognizing sign language gestures in real-time. Enjoy! 🎉

**Need help?** Reach out via [GitHub](https://github.com/trabelssi) or [LinkedIn](https://www.linkedin.com/in/trabelsi-mohamed-amine/)  
**Want to train your own model?** See [README.md](README.md) Training section

---

*Quick Start Guide - Sign Language Recognition System (SLRS)*  
*Author: Mohamed Amine Trabelsi*  
*Last Updated: November 29, 2025*
