# Deep Learning-Based Violence Detection for Efficient Surveillance Video Investigation

---

## 1. Project Overview & Deep Learning Motivation

Manual review of long surveillance footage is labor-intensive, error-prone, and slow. This project implements an automated, deep learning-based forensic investigation system capable of scanning pre-recorded surveillance videos, pinpointing violent incidents, archiving visual evidence frames with exact timestamps, and generating **Explainable AI (XAI)** reports.

### Benchmark Dataset: UBI-Fights (Section IV)
The proposed architecture is evaluated on the benchmark **UBI-Fights** surveillance dataset (`UBI_FIGHTSALL`):

| PARAMETER | DESCRIPTION |
| :--- | :--- |
| **Dataset Name** | UBI_FIGHTSALL |
| **Total Videos** | 1,000 videos |
| **Total Duration** | Approximately 80 hours |
| **Violent Videos** | 216 videos (containing physical fights) |
| **Non-Violent Videos** | 784 videos (everyday, normal surveillance scenes) |
| **Classes** | Violent, Non-Violent (Binary Classification) |
| **Annotation Level** | Frame-level temporal ground truth |

### 14-Step Overall Methodology (Section VI.L)
1. **Step 1:** User uploads a pre-recorded surveillance video.
2. **Step 2:** The uploaded video is read by the video reader (`cv2.VideoCapture`).
3. **Step 3:** Extract frames at pre-determined intervals (stride $K = \lfloor FPS \times \Delta t \rfloor$).
4. **Step 4:** Resized and normalized extracted frames ($224 \times 224$, ImageNet standardization).
5. **Step 5:** During training, augmentation techniques such as rotation, flipping, scale, and brightness adjustment are used.
6. **Step 6:** The pre-processed frames are passed to the classification model based on **MobileNetV2**.
7. **Step 7:** Classify each frame as Violent / Non-Violent.
8. **Step 8:** The prediction confidence score is extracted ($P = \sigma(z) \in [0, 1]$).
9. **Step 9:** In case of detection of violent activity, the corresponding timestamp is found ($T = \text{frame} / FPS$).
10. **Step 10:** Detected frame is kept as forensic evidence.
11. **Step 11:** The system further analyses the rest of the video to detect other violent events.
12. **Step 12:** Each detected event is recorded individually and sorted in chronological order.
13. **Step 13:** **Explainable AI (Grad-CAM):** Generates gradient-weighted visual attention heatmaps *(adapted per faculty guidance)*.
14. **Step 14:** Create a report in PDF or CSV format containing the events detected.

---

## 2. Why Deep Learning & MobileNetV2?
Traditional computer vision approaches relying on handcrafted features (e.g. Optical Flow, SIFT, HOG) suffer from high false-alarm rates under varying lighting, background clutter, and camera angles. Deep Learning automatically learns invariant spatial feature representations.

**MobileNetV2 Architectural Importance:**
* **Depthwise Separable Convolutions:** Splitting standard $3 \times 3$ convolutions into a spatial *Depthwise Convolution* ($3 \times 3$ per channel) and a *Pointwise Convolution* ($1 \times 1$ cross-channel) slashes computational FLOPs by **~8.5×** without sacrificing accuracy:
  $$\text{Complexity Reduction} \approx \frac{1}{N} + \frac{1}{D_k^2}$$
* **Inverted Residuals & Linear Bottlenecks:** Uses an expansion factor $t=6$ to process high-dimensional features while retaining a compact linear bottleneck to prevent non-linearities from destroying low-dimensional feature manifolds.
* **Edge & Surveillance Feasibility:** Weighing only **~3.4 million parameters** (compared to 25.6M in ResNet-50 or 138M in VGG-16), MobileNetV2 is lightweight enough to analyze long surveillance video streams efficiently on CPU or edge hardware.

---

## 2. Explainable AI (XAI) with Grad-CAM

In forensic and legal contexts, a black-box probability score is insufficient. We integrate **Gradient-weighted Class Activation Mapping (Grad-CAM)**:
* Uses the gradients of the violence class score flowing into the final convolutional layer of MobileNetV2 (`features[-1]`):
  $$\alpha_k^c = \frac{1}{Z} \sum_{i=1}^u \sum_{j=1}^v \frac{\partial y^c}{\partial A_{i,j}^k}$$
* The final class-discriminative localization heatmap is generated using a ReLU-rectified weighted sum:
  $$L_{\text{Grad-CAM}}^c = \text{ReLU}\left(\sum_k \alpha_k^c A^k\right)$$
* Warmer colors (Red/Orange) reveal the exact spatial cues (e.g., physical contact, aggressive postures, flailing limbs) that prompted the model's violence prediction.

---

## 3. Algorithms Employed at Respective Stages

| Stage # | Pipeline Stage | Algorithm / Technique Used | Mathematical / Functional Description |
| :---: | :--- | :--- | :--- |
| **1** | **Video Decoding** | OpenCV FFmpeg Stream Demuxer | Decodes video stream (H.264/MPEG-4); extracts FPS, resolution ($W \times H$), and frame count $N$. |
| **2** | **Temporal Subsampling** | Uniform Stride Subsampling | Samples frames at stride $K = \lfloor FPS \times \Delta t \rfloor$, reducing computational overhead by 95%. |
| **3** | **Spatial Preprocessing** | Bilinear Interpolation & Standardization | Resizes frames to $224 \times 224$; normalizes channels with ImageNet mean and standard deviation. |
| **4** | **Spatial Feature Extraction**| MobileNetV2 (18 Inverted Residual Blocks) | Extracts deep feature maps ($[B, 1280, 7, 7]$) using depthwise separable convolutions. |
| **5** | **Explainable AI (XAI)** | Grad-CAM Algorithm | Backpropagates violence class score to generate 2D spatial attention activation heatmaps. |
| **6** | **Classification Probability**| Global Average Pooling + Sigmoid Activation | Condenses feature maps to 1D vector and computes probability $P(\text{Violent}) = \frac{1}{1 + e^{-z}}$. |
| **7** | **Temporal Smoothing** | Sliding Window Moving Average Filter | Rolling window ($W=3$) smooths transient probabilities to suppress single-frame false alarms. |
| **8** | **Timestamp Synchronization**| Discrete Modulo Arithmetic | Calculates exact playback timestamp: $T_{\text{sec}} = \text{Frame Index} / FPS \rightarrow \text{HH:MM:SS}$. |
| **9** | **Incident Aggregation** | Temporal Window Clustering & Merge Sort | Groups adjacent detections ($\Delta t \le 3\text{s}$) into unified incident windows $[T_{\text{start}}, T_{\text{end}}]$. |
| **10**| **Investigation Reporting** | ReportLab Flowable Engine & RFC 4180 CSV | Compiles evidence frames, Grad-CAM heatmaps, and metadata into a forensic PDF report and CSV log. |

---

## 4. Directory Structure

```text
Violence_Detection/
├── app.py                      # Interactive Streamlit Investigation Console
├── requirements.txt            # Project dependencies
├── sample_test.py              # Automated pipeline test & sample video generator
├── models/
│   ├── model.py                # MobileNetV2 PyTorch architecture & custom classifier
│   └── train.py                # Training loop with BCE loss, Adam, & data augmentations
├── core/
│   ├── explainable_ai.py       # Grad-CAM forward/backward hooks & overlay generator
│   ├── video_processor.py      # Video demuxing, stride sampling, inference, evidence saver
│   └── report_generator.py     # Explainable AI PDF & CSV report compiler (ReportLab)
├── data/                       # Test surveillance video clips
└── outputs/
    ├── evidence_frames/        # Saved evidence frames & Grad-CAM heatmaps
    └── reports/                # Generated PDF & CSV investigation reports
```

---

## 5. Getting Started & How to Run

### Step 1: Install Dependencies
Open PowerShell or your terminal in the project directory:
```bash
pip install -r requirements.txt
```

### Step 2: Run End-to-End Pipeline Verification
To verify the deep learning model, Grad-CAM engine, sample video generation, and PDF reporting:
```bash
python sample_test.py
```

### Step 3: Launch the Streamlit Web Application
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501` to:
1. Upload any surveillance video file (`.mp4`, `.avi`, `.mov`).
2. Adjust detection threshold and frame sampling intervals in the sidebar.
3. Click **"Run Deep Learning Analysis"** to watch the scanning progress.
4. Inspect the **Explainable AI (Grad-CAM)** evidence gallery.
5. Download the official **Explainable AI PDF Report** and **CSV Incident Log**.

---

## 6. Training on Custom / UBI-Fights Dataset
To train or fine-tune MobileNetV2 with data augmentations:
1. Organize your dataset:
   ```text
   data/dataset/
   ├── train/
   │   ├── violent/
   │   └── non_violent/
   └── val/
       ├── violent/
       └── non_violent/
   ```
2. Execute the training script:
   ```bash
   python models/train.py --epochs 15 --batch_size 32 --lr 0.0001
   ```
   *The script automatically logs Accuracy, Precision, Recall, F1-score, and Confusion Matrix at every epoch and saves the best model checkpoint to `models/mobilenet_violence_best.pth`.*
