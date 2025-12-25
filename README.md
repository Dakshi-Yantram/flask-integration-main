# Flask Integration for Cervical Cancer Analysis

This repository contains a Flask API for cervical cancer image analysis, classification, and visualization.

## Endpoints
- `/analyze` – Auto-detect and analyze images  
- `/analyze/colposcopy` – Analyze colposcopy images  
- `/analyze/pap_smear` – Analyze pap smear images  
- `/classify` – Classify image type  
- `/quality` – Check image quality  
- `/visualize` – Return visualization  

## How to Run Locally
```bash
pip install -r requirements.txt
python app.py
