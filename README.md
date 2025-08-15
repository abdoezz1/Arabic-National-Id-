# Arabic-National-Id-
# Arabic National ID Extraction System

## 📌 Overview
This project is an **AI-powered system** for extracting information from Egyptian national ID cards.  
It uses **two YOLO-based object detection models** and **OCR** to accurately detect and read fields such as:

- **First Name**
- **Last Name**
- **Address1**
- **Address2**
- **National ID number**
  
The pipeline is designed to handle **real-world card images** with varying lighting, angles, and backgrounds.

---

## 🚀 Features
- **Field Label Detection** – Detects predefined text fields (e.g., name, address) using YOLO.
- **Digit Detection** – Uses a separate YOLO model to detect each digit of the national ID number.
- **OCR Integration** – Uses EasyOCR for text extraction from detected fields.
- **Modular Pipeline** – Easy to extend for new fields or improved models.
- **High Accuracy** – Improved performance over baseline OCR methods.

---

## 🛠 Tech Stack
- **YOLOv11** – Object detection models for field and digit detection.
- **EasyOCR** – Optical Character Recognition for text extraction.
- **Python** – Core language for data processing and pipeline implementation.
- **OpenCV** – Image preprocessing and visualization.
- **PyTorch** – For training YOLO models.
- **Pandas** – For structured output storage.

---
