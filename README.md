# Chest X-ray Pneumonia Detection System

A deep learning-powered diagnostic tool developed using **TensorFlow/Keras**, **OpenCV**, and **Streamlit**, designed to detect pneumonia in chest X-ray images. The system offers explainability features, clinical recommendations, and report generation, supporting fast and accurate diagnosis in low-resource healthcare settings.

## Table of Contents

* [Overview](#overview)
* [Features](#features)
* [Technologies Used](#technologies-used)
* [Installation and Setup](#installation-and-setup)
* [Dataset and Preprocessing](#dataset-and-preprocessing)
* [Usage](#usage)
* [Project Structure](#project-structure)
* [Contributing](#contributing)
* [Authors and Acknowledgements](#authors-and-acknowledgements)

## Overview

The **Chest X-ray Pneumonia Detection System** aims to:

* Provide an accessible, AI-assisted method to classify chest X-rays as **Normal** or **Pneumonia**.
* Integrate a trained Convolutional Neural Network (CNN) with a **Streamlit** frontend for seamless user interaction.
* Offer clinical insights such as saliency maps, severity grading, and follow-up recommendations.
* Support remote diagnosis by exporting results and advice as a **PDF report** using the `fpdf` library.

## Features

* **Model Inference**

  * Classifies chest X-rays into *Normal* or *Pneumonia*.
  * Confidence scores and model accuracy displayed.

* **Clinical Decision Support**

  * Patient symptom descriptions and follow-up recommendations.
  * Severity grading and feature markers (e.g., opacity, consolidation).

* **Explainability**

  * Saliency maps to visualize attention regions in the image.
  * Expandable model architecture and layer information.

* **Report Generation**

  * PDF report download with diagnosis summary and advice.

* **Performance**

  * CPU inference time < 2 seconds per image.
  * Tested on 5,800+ samples with accuracy > 92%.

## Technologies Used

* **Frontend:** Streamlit, HTML/CSS (Bootstrap-inspired)
* **Backend:** Python, TensorFlow/Keras, OpenCV, PIL
* **Reporting:** fpdf (PDF generation)
* **Modeling:** CNN, XGBoost (optional comparative mode)
* **Database:** SQLite (for logging/testing)

## Installation and Setup

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/Avjot-Chawla/Chest-Xray-Pneumonia-Detection.git
   cd Chest-Xray-Pneumonia-Detection
   ```

2. **Install Requirements:**

   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Application:**

   ```bash
   streamlit run app.py
   ```

> ⚠️ If using a pretrained model, place the `.h5` model file in the `models/` directory and update the model loading code accordingly.

## Dataset and Preprocessing

* **Source:** [Kaggle - Chest X-ray Images (Pneumonia)](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia)
* **Split:** 70% Train, 15% Validation, 15% Test
* **Preprocessing:**

  * Resize to `180x180` pixels
  * CLAHE contrast enhancement
  * Normalization to \[0,1] pixel range
  * Augmentation: random flip, zoom, rotation

## Usage

1. **Upload an Image:**

   * Supported formats: `.jpg`, `.jpeg`, `.png`

2. **Review Results:**

   * View prediction results for both CNN and XGBoost models (if enabled).
   * View saliency maps and architecture details.

3. **Download Report:**

   * Click **“Download Full Report”** to save PDF containing analysis and recommendations.

## Project Structure

```
Chest-Xray-Pneumonia-Detection/
├── models/                 # Trained model files
├── app.py                 # Streamlit main app
├── utils/                 # Preprocessing, prediction, and PDF functions
├── assets/                # Sample images or UI resources
├── requirements.txt       # Required Python libraries
└── README.md              # This file
```

## Contributing

Contributions are welcome! If you'd like to contribute a new feature or fix a bug:

1. Fork the repo.
2. Create a new branch.
3. Submit a pull request with clear documentation.

## Authors and Acknowledgements

* **Tanushree Borase** – RA2211003010575
* **Avjot Singh Chawla** – RA2211003010584
* **Project Supervisor:** Dr. Thamizhamuthu R
  *Department of Computing Technologies, SRM Institute of Science and Technology*
