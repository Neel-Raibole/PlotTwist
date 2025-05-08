# PlotTwist Project

## 1. Introduction

Static charts such as pie charts, bar graphs, and line plots are everywhere in reports, publications, and dashboards, yet their underlying data often remains locked in pixels. This hinders automated analysis and forces researchers and analysts especially in high-stakes fields like pharmaceuticals to rely on slow, error-prone manual transcription. To bridge this gap, my PlotTwist project delivers a proof-of-concept (PoC) system that turns any static chart image into structured data and a concise, human-readable summary.

Over six months, I built a multi-stage pipeline combining:
- Computer Vision & Object Detection (YOLOv11) to locate chart components
- OCR (PaddleOCR) to read embedded text
- Image Processing (CLAHE, LAB & YCbCr color space, edge detection, contour analysis, color clustering, super resolution) to isolate visual regions
- LLM Summarization (Mistral via llama.cpp) to generate natural-language insights
- Streamlit UI for real-time interaction
- Docker for project deployment

As a PoC, PlotTwist already cuts manual chart-review time by over 70%. Its modular design paves the way for future enhancements supporting more chart types, real-time batch processing, deeper domain adaptation, and seamless integration into enterprise workflows.

## 2. Features

This project is an end-to-end pipeline that transforms static chart images into structured tabular data and natural language summaries. The system incorporates computer vision, OCR, image processing, and large language models, and is deployed using a Streamlit web app containerized via Docker.

### Chart Component Detection
- Supports simple vertical bar charts, single-line charts, and exploded pie charts
- Uses a custom-trained YOLOv11 model (Ultralytics) to detect chart regions like:
  - Titles, legends, axis ticks, and plotting areas
  - Bars in bar charts, wedges in pie charts, and line paths in line charts
- Designed to work across varying visual styles and formats

### Text Extraction & Association
- Integrates PaddleOCR with angle correction to extract chart text such as:
  - X/Y axis labels, legend text, values, and titles
- Applies super-resolution preprocessing before OCR to enhance cropped image clarity and improve recognition accuracy
- Associates extracted text with visual components using bounding box alignment and position heuristics

### Image Processing
- Improves component segmentation and detection using:
  - CLAHE (Contrast Limited Adaptive Histogram Equalization) for contrast enhancement
  - Conversion to YCbCr color space for accurate color-based segmentation (especially in pie charts)
  - Edge detection and contour analysis for isolating chart elements
- Applies a combination of:
  - Binary and adaptive thresholding for region separation
  - Morphological operations (opening and closing) to clean noise and refine shapes
  - Kernel sharpening to enhance edges before OCR

### Data Structuring
- Converts visual elements into structured data tables with inferred mappings
- Supports categorical x-axes (e.g., months, product categories) as well as numeric values
- Limitations:
  - Currently does not handle missing or irregular axis ticks
  - A T5-based axis reconstruction module was prototyped but not fully developed due to data constraints

### Natural Language Summarization
- Extracted tables are converted into prompts for Mistral LLM via llama.cpp
- Generates human-like summaries highlighting insights, patterns, and trends in the chart data
- Output is English-only for now
- Prompt engineering is used to guide the format and focus of the generated summaries (examples provided in the Results section)

### Web Interface & Deployment
- Built using Streamlit with:
  - File upload widget for chart images
  - Drop-down menu to select chart type
  - Real-time output of detected components, structured data, and summary
- Fully containerized using Docker for local deployment
  - Runs on default ports with basic volume mapping (details in Installation section)

## 3. Tech Stack

| Component | Technology Used | Purpose |
|-----------|----------------|---------|
| Chart Detection | YOLOv11 (Ultralytics) | Object detection of chart components |
| OCR | PaddleOCR | Extracting text from chart elements |
| Image Processing | OpenCV, NumPy | Preprocessing for clarity and segmentation |
| Super-Resolution | OpenCV DNN / ESRGAN (if used) | Enhancing cropped text regions pre-OCR |
| LLM Summarization | Mistral via llama.cpp | Natural language generation |
| Web Interface | Streamlit | User-friendly frontend for interaction |
| Deployment | Docker | Local containerized app execution |
| Annotation Tool | CVAT | Manual labeling of chart components |

## 4. Dataset

This project uses a custom-compiled dataset consisting of over 1,500 static chart images, including bar charts, line graphs, and pie charts. These images were sourced from publicly available online repositories and synthesized to reflect a diverse range of chart styles and qualities.

### Dataset Composition
- 500+ Bar Charts (Vertical)
- 500+ Line Charts (Single-Series)
- 500+ Pie Charts (Including exploded variants)

Each image was manually annotated using CVAT, identifying the following components:
- Chart title
- X and Y axis 
- X and Y axis labels
- X and Y axis ticks
- Legends
- Bars, lines, or wedges (as per chart type)
- Plotting area

Annotations are saved in YOLO format and were used to train the detection model.

### Sample Images
**Bar Chart Samples**
<p align="center">
  <img src="https://github.com/Neel-Raibole/DataZymes/blob/main/demo%20images/bar%20chart%20non-annotated.png" alt="Non-Annotated Bar Chart" width="475"/>
  <img src="https://github.com/Neel-Raibole/DataZymes/blob/main/demo%20images/bar%20chart%20annotated.jpg" alt="Annotated Bar Chart" width="475"/>
</p>

**Line Chart Samples**
<p align="center">
  <img src="https://github.com/Neel-Raibole/DataZymes/blob/main/demo%20images/line%20chart%20non-annotated.png" alt="Non-Annotated Line Chart" width="475"/>
  <img src="https://github.com/Neel-Raibole/DataZymes/blob/main/demo%20images/line%20chart%20annotated.jpg" alt="Annotated Line Chart" width="475"/>
</p>

**Pie Chart Samples**
<p align="center">
  <img src="https://github.com/Neel-Raibole/DataZymes/blob/main/demo%20images/pie%20chart%20non-annotated.png" alt="Non-Annotated Pie Chart" width="475"/>
  <img src="https://github.com/Neel-Raibole/DataZymes/blob/main/demo%20images/pie%20chart%20annotated.jpg" alt="Annotated Pie Chart" width="475"/>
</p>

## 5. Methodology

The system follows a modular pipeline:

1. **Data Collection & Annotation**
   - 500+ images per chart type sourced from public datasets and manually annotated using CVAT
   - Labeled elements: title, legend, axes, bars/lines/wedges, ticks

2. **Model Training – YOLOv11**
   - Annotated charts were used to train YOLOv11 with:
     - Augmentation (flip, scale), learning rate tuning, and batch adjustments
   - Outputs bounding boxes for chart components

3. **Image Preprocessing**
   - CLAHE used to enhance contrast
   - Morphological opening/closing to remove noise and refine components
   - Color space conversion to YCbCr for accurate segmentation in pie charts
   - Sharpening filters and thresholding applied before OCR

4. **Text Extraction – PaddleOCR**
   - Cropped bounding box regions passed through OCR
   - Pre-enhanced using super-resolution for improved accuracy
   - Outputs are associated with chart components by spatial alignment

5. **Data Structuring**
   - Components are mapped into structured tabular form
   - Categorical axis labels supported
   - Gaps in ticks or missing values not currently handled

6. **Insight Generation**
   - Structured tables passed into a prompt template
   - LLM (Mistral via llama.cpp) generates human-like summaries of chart data

7. **Web App Deployment – Streamlit + Docker**
   - Drag-and-drop image upload and chart type selection
   - Displays table output and generated summary
   - Hosted locally via Docker for modular deployment

## 6. Model Training (YOLOv11)

The YOLOv11 model was trained to detect chart components (titles, legends, axes, data regions) with high precision across varied chart styles.

### 6.1 Hyperparameters Tuned
- **Learning Rate**: Adjusted to balance convergence speed and stability
- **Weight Decay**: Regularization to prevent overfitting
- **Momentum**: Improved SGD convergence for image features
- **Optimizer**: Stochastic Gradient Descent (SGD) with momentum—found to outperform Adam for this vision task

### 6.2 Training Artifacts
- **Final Weights**:[best.pt File Link](https://github.com/Neel-Raibole/DataZymes/tree/main/run2/detect/final_model_continued/weights)
- **Training Logs & Metrics**: Stored alongside weights in the same directory

### 6.3 Example Training Invocation
Below is a representative command to launch YOLOv11 training on your annotated dataset. Adjust paths, epochs, and hyperparameters as needed:

Link to complete YOLO Training File: [best.pt File Link](https://github.com/Neel-Raibole/DataZymes/blob/main/Graphs_Repository/Complete%20Training/YOLO_Training.ipynb)

```python
results = model.train(
    data = os.path.join(ROOT_DIR, "config.yaml"), # Full dataset config file
    epochs = 100,# Full training epochs
    imgsz = 800, # Image resolution
    batch = 8, # Adjust based on your GPU memory
    lr0 = best_lr, # Best learning rate from HPO
    momentum = best_momentum, # Best momentum from HPO
    weight_decay = best_wd, # Best weight decay from HPO
    optimizer = "SGD", # Using SGD optimizer
    workers = 2, # Number of dataloader workers
    half = True, # Enable mixed precision for speed-up
    name = "final_model_with_best_params" # Unique experiment name for logging
)

# Evaluate the trained model on the validation set
metrics = model.val(data = os.path.join(ROOT_DIR, "config.yaml"))
print("Validation metrics:")
print(metrics.results_dict)
```

## 7. Project Pipeline

An end-to-end flow that turns a static chart image into structured data and a natural language summary:

1. **Component Detection & JSON Conversion**
   - Input full chart image into a custom-trained YOLOv11 model
   - Detects bounding boxes for: titles, legends, axis regions, plotting areas, data points
   - Converts detections into a JSON structure for downstream steps

2. **Element Cropping & Preprocessing**
   - Crop each detected region (e.g., plotting area, legend block)
   - On each crop, sequentially apply:
     1. Morphological opening/closing
     2. Sharpening filter
     3. Denoising
     4. Super-resolution (EDSR)
   - Dynamically apply additional steps (CLAHE, thresholding, blurring) based on the chart type

3. **Chart-Type-Specific Processing**
   - Line & Bar Charts:
     - Use contour analysis and pixel-by-pixel scaling to map axis ticks to values
     - If data-point labels exist, directly associate them to X-axis positions; otherwise infer via the plotted line/bar top
   - Pie Charts:
     - Convert legend crop to YCbCr, apply CLAHE, then compute |Cb–Cr|
     - Cluster pie-slice pixels using K-Means → GMM (k = number of legend entries)
     - Map each color cluster to its legend text via CIEDE2000 color distance

4. **Text Extraction & Association**
   - Enhance cropped text regions with super-resolution, then run PaddleOCR (with angle correction)
   - Assign OCRed strings (titles, axis labels, legend entries) back to their respective visual components by bounding-box alignment

5. **Data Structuring**
   - Build a tidy table: each data point's X and Y values (or category & percentage for pie)
   - Support both numeric axes and categorical labels (months, quarters)

6. **NLP Summarization**
   - Convert the structured table into a prompt template based on the chart type
   - Invoke Mistral via llama.cpp to generate an English summary describing trends and insights

7. **Output Delivery**
   - Users upload an image, select chart type, and receive instant, structured analysis and summary

This pipeline ensures each chart type is handled with both generalized preprocessing and chart-specific logic, yielding accurate data extraction and human-readable insights.

## 8. Deployment, Installation & Usage

### 1. Prerequisites
- **Python**: Any modern Python 3.x installation (only needed if you want to inspect code locally)
- **Docker Desktop**:
  1. Windows: https://docs.docker.com/desktop/setup/install/windows-install/
  2. Mac: https://docs.docker.com/desktop/setup/install/mac-install/
  3. Ubuntu: https://docs.docker.com/desktop/setup/install/linux/ubuntu/

### 2. Download the Code
- Grab the latest ZIP from Google Drive:[Google Drive Link](https://drive.google.com/file/d/1Imt-P7YEk9G_MV-bN0XV-YaP7CqU2XYA/view?usp=sharing)
- Unzip and cd into the extracted directory

### 3. Build the Docker Image
- From the project root (where the Dockerfile resides): `docker build -t streamlit-chart-app .`

### 4. Run the Docker Container
- Launch the container, mapping host port 8501 to the app's port: `docker run -p 8501:8501 streamlit-chart-app`

### 5. Access the App
- Open a browser and navigate to: http://localhost:8501/

### 6. Using the UI
- **Upload Image**: Drag-and-drop or browse to select a chart image
- **Chart Type**: Choose "Bar Chart," "Line Chart," or "Pie Chart"
- **Data Points**: Specify if the chart includes explicit data point markers
- **View Results**: The app displays detected components, a structured data table, and a generated summary

## 9. Project Structure

This repository contains multiple modules, data, and configuration files. Below is the top-level layout. For full details you can view [Project_Folder_Structure.txt](https://github.com/Neel-Raibole/DataZymes/blob/main/Project_Folder_Structure.txt) in the repo.

Note: The mistral LLM file is over 5GB hence it is being stored in Google Drive

```
PlotTwist Project/DataZymes
¦   .dockerignore
¦   Capstone_Project_Code.py
¦   dockerfile
¦   EDSR_x4.pb
¦   README.md
¦   mistral-7b-instruct-v0.1.Q5_K_M.gguf
¦   requirements.txt
¦   Roadmap.xlsx
¦   streamlit_app.py
¦   
+---Graphs_Repository: # Contains Graphs and YOLO Training Files
¦                   
+---paddle_models: # Contains PaddleOCR Build Files
¦           
+---run2 # Contains YOLO Model Training Weights & Results
¦
+---Test_Images #Contains Post Training Testing Example Chart Image Files
```

## 10. Results & Evaluation

This section presents the key quantitative results from YOLOv11 training and qualitative examples of the end-to-end pipeline outputs.

### 10.1 YOLOv11 Detection Performance

**Training Metrics Plot:**
![YOLO Training Results Image](https://github.com/Neel-Raibole/DataZymes/blob/main/demo%20images/results.png)

**Confusion Matrix:**
![YOLO training outputs](https://github.com/Neel-Raibole/DataZymes/blob/main/demo%20images/confusion_matrix.png)

#### Highlights:
- **Overall**: Achieved 0.973 Precision, 0.991 Recall, 0.989 mAP@50, and 0.79 mAP@50–95 across 1,281 instances.
- **Axes & Titles**:
  - X/Y Axes & Titles detected with >0.94 Precision, >0.96 Recall, and up to 0.926 mAP@50–95.
- **Chart Areas**:
  - Bar, Line, and Pie Charts all achieved >0.965 Precision/Recall, 0.995 mAP@50, and up to 0.995 mAP@50–95.
- **Legend & Title**: Nearly perfect with >0.98 Precision, 1.0 Recall, and ~0.995 mAP@50.
- **Data Points & Ticks**: High detection accuracy with slightly lower performance on strict mAP@50–95 due to small size and visual density.

### 10.2 Project Pipeline Outputs

Below are example screenshots demonstrating the end-to-end functionality:

**Bar Chart with Data Point Samples**
<p align="center">
  <img src="https://github.com/Neel-Raibole/DataZymes/blob/main/demo%20images/bc%20output%20wd%201.jpg" alt="Non-Annotated Bar Chart" width="475"/>
  <img src="https://github.com/Neel-Raibole/DataZymes/blob/main/demo%20images/bc%20output%20wd%202.jpg" alt="Annotated Bar Chart" width="475"/>
</p>

**Bar Chart without Data Point Samples**
<p align="center">
  <img src="https://github.com/Neel-Raibole/DataZymes/blob/main/demo%20images/bc%20output%20wod%201.jpg" alt="Non-Annotated Bar Chart" width="475"/>
  <img src="https://github.com/Neel-Raibole/DataZymes/blob/main/demo%20images/bc%20output%20wod%202.jpg" alt="Annotated Bar Chart" width="475"/>
</p>

**Line Chart with Data Point Samples**
<p align="center">
  <img src="https://github.com/Neel-Raibole/DataZymes/blob/main/demo%20images/lc%20output%20wd%201.jpg" alt="Non-Annotated Line Chart" width="475"/>
  <img src="https://github.com/Neel-Raibole/DataZymes/blob/main/demo%20images/lc%20output%20wd%202.jpg" alt="Annotated Line Chart" width="475"/>
</p>

**Line Chart without Data Point Samples**
<p align="center">
  <img src="https://github.com/Neel-Raibole/DataZymes/blob/main/demo%20images/lc%20output%20wod%201.jpg" alt="Non-Annotated Line Chart" width="475"/>
  <img src="https://github.com/Neel-Raibole/DataZymes/blob/main/demo%20images/lc%20output%20wod%202.jpg" alt="Annotated Line Chart" width="475"/>
</p>

**Pie Chart with Data Point Samples**
<p align="center">
  <img src="https://github.com/Neel-Raibole/DataZymes/blob/main/demo%20images/pc%20output%20wd%201.jpg" alt="Non-Annotated Pie Chart" width="475"/>
  <img src="https://github.com/Neel-Raibole/DataZymes/blob/main/demo%20images/pc%20output%20wd%202.jpg" alt="Annotated Pie Chart" width="475"/>
</p>

**Pie Chart without Data Point Samples**
<p align="center">
  <img src="https://github.com/Neel-Raibole/DataZymes/blob/main/demo%20images/pc%20output%20wod%201.jpg" alt="Non-Annotated Pie Chart" width="475"/>
  <img src="https://github.com/Neel-Raibole/DataZymes/blob/main/demo%20images/pc%20output%20wod%202.jpg" alt="Annotated Pie Chart" width="475"/>
</p>

## 11. Future Work

- **Expanded Chart Coverage**:
  Support a wider variety of visualizations, including stacked/clustered bar charts, multi-series and area line graphs, scatter plots, and heatmaps.

- **Model Optimization**:
  Prune and quantize detection, OCR, and super-resolution models for faster inference and lower resource usage on edge devices.

- **Interactive Q&A Interface**:
  Enable users to ask natural-language questions about a chart ("What was the highest value in Q3?") and receive precise, data-driven answers.

- **Reinforcement Learning from Human Feedback (RLHF)**:
  Incorporate user corrections and feedback loops to continuously improve detection accuracy, OCR quality, and summary relevance.

- **Dashboard & Infographic Analysis**:
  Extend the pipeline to process entire dashboards and infographics, automatically stitching together multiple charts into a coherent narrative and deeper insight story.
