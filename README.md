# PillTrack: MLOps Producer Hub

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?style=for-the-badge)
![DataOps](https://img.shields.io/badge/DataOps-Enabled-darkslategray?style=for-the-badge)
![AWS S3](https://img.shields.io/badge/Storage-AWS%20S3-orange?style=for-the-badge)

---

## Overview

**PillTrack Producer Hub** is a streamlined **MLOps production pipeline** for medicine pack identification.  
The system manages the **end-to-end lifecycle** of dataset ingestion, feature extraction, artifact versioning, and synchronization with production environments via **AWS S3**.

This hub is designed for **AI operators and MLOps engineers** who need reliability, traceability, and rapid deployment for vision-based healthcare systems.

---

## Architecture Overview

The system follows a **modular design** to ensure scalability, maintainability, and clean separation of responsibilities.

### Core Modules

- **`engine.py`**  
  Core AI logic using:
  - **YOLOv8 Segmentation** for object localization  
  - **DINOv2** for high-dimensional feature extraction  
  - Generates **4-directional rotation-invariant vectors** (0°, 90°, 180°, 270°)

- **`db_manager.py`**  
  - Local vector database management (`.pkl`)
  - Metadata handling
  - Automated **Activity Logging (Audit Trail)**

- **`cloud_manager.py`**  
  - Secure artifact synchronization with **AWS S3**
  - Handles model, vector DB, and metadata versioning

- **`app.py`**  
  - Centralized **Streamlit Dashboard**
  - Coordinates ingestion, processing, validation, and deployment

---

## Key Features

- **Automated Feature Extraction**  
  Rotation-invariant vector embeddings for robust medicine pack identification

- **Production Synchronization**  
  One-click upload of:
  - Model weights  
  - Vector databases  
  - Metadata and logs  
  to S3 production buckets

- **Operational Logging**  
  JSON-based activity logs covering:
  - Data ingestion  
  - Feature extraction  
  - Deployment events  

- **Raw Data Archiving**  
  Original images are preserved for:
  - Re-training  
  - Auditing  
  - Debugging  

---

## Project Structure

```bash
.
├── app.py                # Main Streamlit UI Dashboard
├── engine.py             # AI Logic (YOLOv8 + DINOv2)
├── db_manager.py         # Vector DB Registry & Logging
├── cloud_manager.py      # AWS S3 Integration
├── config.yaml           # System Configuration
├── database/             # Vector DB & Metadata
│   ├── db_packs_dino.pkl
│   └── activity_log.json
├── database_images/      # Archived Raw Image Dataset
└── models/               # Pre-trained Model Weights (.pt)

Getting Started
1️⃣ Prerequisites

Python 3.9+

AWS CLI configured with S3 access

2️⃣ Setup
pip install -r requirements.txt


Create a .env file:

AWS_ACCESS_KEY_ID=your_access_key
AWS_SECRET_ACCESS_KEY=your_secret_key
S3_BUCKET_NAME=your_bucket_name

3️⃣ Execution
streamlit run app.py


Author

Sitta S.
AI Engineer Intern @ AI SmartTech