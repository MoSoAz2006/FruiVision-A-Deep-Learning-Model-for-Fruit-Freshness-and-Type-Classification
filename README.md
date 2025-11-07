# 🍎 FruiVision  
### Deep Learning Model for Fruit Type & Freshness Detection  

FruiVision is a deep learning–based computer vision project that classifies **fruit type** and determines whether the fruit is **fresh or rotten** using **PyTorch** and **ResNet-18**.  
The model was trained on the **[Fresh and Stale Fruit Classification](https://www.kaggle.com/datasets/swoyam2609/fresh-and-stale-classification)** dataset and achieves high accuracy in identifying both fruit categories and freshness status.

---

## 🌟 Key Features
- 🧠 Dual-task CNN model (fruit type + freshness classification)  
- ⚙️ Transfer Learning using **ResNet-18** backbone  
- 🖼️ Data augmentation and normalization with **torchvision.transforms**  
- 💾 Model saving & single-image prediction support  
- 📊 Training visualization (loss and accuracy graphs)  
- 🔍 Confusion matrices and classification reports  

---

## 📂 Dataset
Dataset used: **Fresh and Stale Fruit Classification (Kaggle)**  
- Fruits included: Apple, Banana, Orange, Tomato, Potato, etc.  
- Two freshness states: **Fresh** and **Rotten**

Dataset structure:
```
dataset/
│
├── Train/
│   ├── freshapple/
│   ├── rottenapple/
│   ├── freshbanana/
│   ├── rottenbanana/
│   └── ...
│
└── Test/
    ├── freshapple/
    ├── rottenapple/
    ├── freshbanana/
    ├── rottenbanana/
    └── ...
```

---

## ⚙️ Technologies Used
- Python 3.10+  
- PyTorch & Torchvision  
- OpenCV  
- Scikit-learn  
- Matplotlib & Seaborn  
- Torchmetrics  

---

## 🚀 How to Run the Project

### 1️⃣ Clone the repository:
```bash
git clone https://github.com/YOUR-USERNAME/FruiVision.git
cd FruiVision
```

### 2️⃣ Install dependencies:
```bash
pip install -r requirements.txt
```

### 3️⃣ Download the dataset:
Either download from [Kaggle Dataset](https://www.kaggle.com/datasets/swoyam2609/fresh-and-stale-classification)  
or use KaggleHub directly:
```python
import kagglehub
path = kagglehub.dataset_download('swoyam2609/fresh-and-stale-classification')
print(path)
```

### 4️⃣ Train the model:
```bash
python train.py
```

This will:
- Load and preprocess images  
- Train the dual-output model (fruit + freshness)  
- Save the model to `model.pth`  

---

## 🧠 Model Architecture
The model is built upon **ResNet-18** and has two output branches:

```
Input Image (224x224)
    ↓
ResNet-18 Feature Extractor
    ↓
┌───────────────┬───────────────┐
│ Fruit Class (9) │ Freshness (2) │
└───────────────┴───────────────┘
```

---

## 🖼️ Predicting a Single Image

After training, you can test the model on a single image using:
```python
import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt

# Load model
model = torch.load("model.pth", map_location='cpu')
model.eval()

# Label encoders (should match your training code)
fruit_labels = ['apple', 'banana', 'bittergourd', 'capsicum', 'orange', 'potato', 'tomato', 'cucumber', 'bellpepper']
fresh_labels = ['Fresh', 'Rotten']

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=0, std=1)
])

# Load image
image_path = "test_image.jpg"  # your image here
image = Image.open(image_path).convert('RGB')
input_tensor = transform(image).unsqueeze(0)

# Predict
with torch.no_grad():
    fruit_pred, fresh_pred = model(input_tensor)
    fruit_class = fruit_labels[torch.argmax(fruit_pred)]
    fresh_class = fresh_labels[torch.argmax(fresh_pred)]

plt.imshow(image)
plt.axis('off')
plt.title(f"{fruit_class} - {fresh_class}")
plt.show()
```

---

## 📊 Results (Sample)
| Metric | Accuracy |
|:-------|:----------|
| Fruit Type | **92%** |
| Freshness  | **89%** |

Visualization samples:
- Training vs Validation Loss  
- Fruit Type Accuracy  
- Freshness Accuracy  
- Confusion Matrices  

---

## 💡 Applications
- 🍇 Smart Agriculture  
- 🥝 Food Quality Control  
- 🍊 Automated Sorting Systems  

---

## 🧑‍💻 Author
**MoSo.Az**  
📧 mosoaz2006@gmail.com
💼 [GitHub](https://github.com/MoSoAz2006)

---

⭐ **If you find this project useful, please give it a star on GitHub!**
