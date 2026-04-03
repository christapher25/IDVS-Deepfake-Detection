

# 🕵️‍♂️ I.D.V.S: Integrated Deepfake Verification System

![Python](https://img.shields.io/badge/Python-3.11-blue?style=for-the-badge&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![React](https://img.shields.io/badge/React-Vite-20232A?style=for-the-badge&logo=react&logoColor=61DAFB)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-white?style=for-the-badge&logo=opencv&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

> **Restoring digital trust through multi-modal AI.** <br>
> I.D.V.S. is a real-time, hybrid deepfake detection framework that combines deep spatial cryptographic analysis with biological behavioral heuristics to definitively classify synthetic media.

---

## 🚨 The Problem
Modern Generative Adversarial Networks (GANs) and diffusion models create hyper-realistic deepfakes that easily bypass traditional, single-method Convolutional Neural Networks (CNNs). While heavy deep learning models (like EfficientNetB7) or complex physiological trackers (like rPPG) exist, they are computationally expensive and highly vulnerable to video compression and poor lighting.

## 💡 Our Solution: The Hybrid Approach
I.D.V.S. bridges the gap between pixel-level forensics and human biology. By fusing a lightweight CNN for spatial analysis with a mathematical heuristic for temporal blink detection, we eliminate single-point algorithmic failures and process videos in real-time on standard consumer hardware.

### ✨ Core Features
* 🧬 **Biological Liveness Verification:** Utilizes **MediaPipe Face Mesh** to track 468 3D facial landmarks, computing the Eye Aspect Ratio (EAR) across temporal frames to detect physiologically impossible blinking behaviors.
* 🖼️ **Spatial Deep Learning Analysis:** Employs an optimized **EfficientNetB0** architecture to detect micro-compression artifacts and GAN blending boundaries.
* ⚖️ **Hybrid Fusion Decision Engine:** A custom "Council of Four" weighted soft-voting algorithm that dynamically resolves high-variance conflicts (e.g., perfect skin texture but zero blinks) to trigger anomaly alerts.
* ⚡ **Asynchronous Backend:** Powered by **FastAPI**, handling heavy `.mp4` payloads without blocking the main server thread or causing CUDA memory crashes.
*📱 **Social Media UI:** A responsive **React/Vite** frontend mimicking modern social feeds, allowing users to verify media naturally.

---

## 🏗️ System Architecture

1. **Media Ingestion:** User uploads an `.mp4` or `.avi` via the React frontend.
2. **Preprocessing:** OpenCV samples temporal frames at a 1:10 ratio.
3. **Face Isolation:** **MTCNN** isolates, geometrically aligns, and crops facial tensors, acting as a primary fail-safe.
4. **Parallel Inference:** * **Module A:** EfficientNetB0 evaluates spatial authenticity.
   * **Module B:** MediaPipe evaluates temporal liveness.
5. **Verdict Generation:** The Hybrid Decision Engine aggregates scores, applies conflict resolution protocols, and returns a transparent JSON payload with a confidence metric.

---

## 📂 Repository Structure


IDVS-Deepfake-Detection/
├── backend_api.py         # FastAPI routing and asynchronous endpoints

├── backend_logic.py       # Core inference orchestration and fusion logic

├── models/                # ML architecture definitions and training scripts
│   ├── 8_train_efficientnet.py
│   └── 4_inference_logic.py

├── frontend/              # React/Vite UI codebase
│   ├── src/
│   └── package.json

├── requirements.txt       # Python dependencies

└── README.md              # Project documentation


---

## 🚀 Installation & Setup

### Prerequisites
* **Python** 3.11 or higher
* **Node.js** (for frontend UI)
* *(Optional)* NVIDIA GPU with CUDA support for accelerated inference

### 1. Clone the Repository

git clone https://github.com/christapher25/IDVS-Deepfake-Detection.git
cd IDVS-Deepfake-Detection


### 2. Backend Setup (Python)
Create an isolated virtual environment and install the required machine learning dependencies:

# Create virtual environment
python -m venv .venv

# Activate on Windows:
.venv\Scripts\activate
# Activate on Mac/Linux:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt


### 3. Download Pre-trained Weights
To keep this repository lightweight, the heavily trained neural network weights are hosted externally.

### 4. Frontend Setup (React/Vite)
Open a new terminal window, navigate to the frontend folder, and install the Node modules:

cd frontend
npm install




## 💻 How to Run the Application

**Step 1: Start the FastAPI Backend**
Ensure your Python virtual environment is active, then run:

uvicorn backend_api:app --reload

*The API will boot up securely at `http://localhost:8000`*

**Step 2: Start the React Frontend**
In your frontend terminal, execute:

npm run dev

*The interactive dashboard will be available at `http://localhost:5173`*



## 🎓 Academic Context
This project was developed as a **B.Tech Computer Science and Engineering (AI/ML)** Mini-Project at St. Thomas College of Engineering & Technology under APJ Abdul Kalam Technological University.

**The Engineering Team:**
* **Aaron Abraham Mathew**
* **Abel Tijo**
* **Anitta Stalen**
* **Christapher Thomas Rajesh**

*Project Guided by: Prof. Anish George*

---

## 📄 License
This project is licensed under the **MIT License**. See the `LICENSE` file for more details. 
```
