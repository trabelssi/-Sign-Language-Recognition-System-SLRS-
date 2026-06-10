# Model Information - model.h5

## Overview

This file contains the trained deep learning model for sign language gesture recognition. The model is based on the ResNet50V2 architecture with transfer learning and has been trained to recognize 6 different hand gestures.

## Model Specifications

### Architecture
- **Base Model**: ResNet50V2 (pre-trained on ImageNet)
- **Transfer Learning**: 60 layers frozen, remaining layers fine-tuned
- **Input Shape**: 224 × 224 × 3 (RGB images)
- **Output**: 6 classes (softmax activation)
- **Framework**: TensorFlow/Keras

### Training Details

#### Dataset
- **Source**: HaGRID (HAnd Gesture Recognition Image Dataset)
- **Total Images**: 10,380 RGB Full HD images
- **Training Set**: 8,280 images (1,380 per class) - 79.24%
- **Validation Set**: 2,100 images (350 per class) - 20.21%
- **Test Set**: 1,200 images (200 per class) - 11.55%

#### Gesture Classes (6 total)
0. **call** - Hand gesture for phone call
1. **mute** - Hand gesture for mute/silence
2. **peace** - Peace sign gesture
3. **ok** - OK hand gesture
4. **stop** - Stop/halt gesture
5. **palm** - Open palm gesture

#### Training Parameters
- **Epochs**: 20
- **Batch Size**: 8
- **Learning Rate**: 0.0001
- **Optimizer**: Adam
- **Loss Function**: Categorical Crossentropy
- **Frozen Layers**: 60 (from ResNet50V2 base)
- **Training Time**: ~7 hours on Google Colab GPU
- **Platform**: Google Colab with GPU/TPU acceleration

#### Data Augmentation
Applied to training and validation sets:
- Rotation
- Horizontal and vertical shifts
- Shear transformation
- Zoom
- Horizontal flip

### Performance Metrics

#### Overall Performance
- **Test Accuracy**: 95%
- **Validation Accuracy**: Maintained satisfactory level throughout training
- **Overfitting**: Minor overfitting observed after epoch 17

#### Per-Class Performance

| Gesture | Precision | Recall | F1-Score | Accuracy |
|---------|-----------|--------|----------|----------|
| Call    | 0.42      | 0.45   | 0.44     | 0.83     |
| Mute    | 1.00      | 0.02   | 0.04     | 0.97     |
| Ok      | 0.44      | 0.26   | 0.33     | 0.82     |
| Palm    | 0.71      | 0.81   | 0.75     | 0.86     |
| Peace   | 0.38      | 0.39   | 0.38     | 0.80     |
| Stop    | 0.70      | 0.63   | 0.67     | 0.92     |

**Note**: The "mute" and "ok" classes show some confusion in the confusion matrix, which is expected given their visual similarity.

## Usage

### Loading the Model

```python
from keras.models import load_model

# Load the pre-trained model
model = load_model('model.h5')
```

### Making Predictions

```python
import numpy as np
from PIL import Image

def preprocessing(img_path):
    img = Image.open(img_path)
    img = img.resize((224, 224))
    img = np.array(img)
    img = img.astype('float32') / 255.0  # Normalize
    img = img.reshape(1, 224, 224, 3)  # Reshape for model
    return img

# Make prediction
img = preprocessing('path/to/image.jpg')
predictions = model.predict(img)
class_index = np.argmax(predictions)
probability = np.amax(predictions)

# Map to gesture name
class_names = ['call', 'mute', 'peace', 'ok', 'stop', 'palm']
gesture = class_names[class_index]
print(f"Detected: {gesture} with {probability*100:.2f}% confidence")
```

### GPU Acceleration

```python
import tensorflow as tf

# Check GPU availability
if tf.test.is_gpu_available():
    print('Using GPU for inference')
else:
    print('Using CPU for inference')
```

## Model File Details

- **Filename**: `model.h5`
- **Format**: HDF5 (Hierarchical Data Format)
- **File Size**: 245 MB (245,053,600 bytes)
- **Last Modified**: November 28, 2025
- **Compatible With**: TensorFlow 2.x, Keras 2.x

## Deployment

### Local Deployment
- Use `CLASS WITH AUDIO.py` for audio feedback
- Use `CLASS WITHOUT AUDIO.py` for silent mode
- Requires: TensorFlow, OpenCV, NumPy, PIL

### Raspberry Pi Deployment
- Recommended: Raspberry Pi 4 with 4GB+ RAM
- USB Camera required
- See hardware implementation section in README
- Network configuration via SSH/VNC

### Cloud Deployment
- Trained on Google Colab
- Compatible with AWS, Azure, GCP
- GPU acceleration recommended for production

## Model Limitations

1. **Class Imbalance**: "Mute" class shows high precision but low recall
2. **Similar Gestures**: "Ok" and "mute" gestures may be confused
3. **Lighting Conditions**: Best performance in indoor conditions with good lighting
4. **Distance Range**: Optimal at 0.5-4 meters from camera
5. **Background**: Performance may vary with complex backgrounds

## Future Improvements

- Increase dataset size for underperforming classes
- Implement data balancing techniques
- Add more gesture classes
- Optimize model for mobile/edge devices
- Implement real-time pose estimation for better accuracy

## Citation

If you use this model in your research or project, please cite:

```
Sign Language Recognition System (SLRS)
Author: Mohamed Amine Trabelsi
GitHub: https://github.com/trabelssi
LinkedIn: https://www.linkedin.com/in/trabelsi-mohamed-amine/
PFE - Projet de Fin d'Études
Based on ResNet50V2 with Transfer Learning
HaGRID Dataset, 2025
```

## License

See LICENSE file for details.

---

**Author**: Mohamed Amine Trabelsi  
**GitHub**: [trabelssi](https://github.com/trabelssi)  
**LinkedIn**: [trabelsi-mohamed-amine](https://www.linkedin.com/in/trabelsi-mohamed-amine/)  
**Last Updated**: November 29, 2025  
**Model Version**: 1.0  
**Training Platform**: Google Colab
