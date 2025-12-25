# Flask Integration for Cervical Cancer Analysis

This repository contains a Flask API for cervical cancer image analysis, classification, and visualization.

## Endpoints
- `/analyze` – Auto-detect and analyze images  
- `/analyze/colposcopy` – Analyze colposcopy images  
- `/analyze/pap_smear` – Analyze pap smear images  
- `/classify` – Classify image type  
- `/quality` – Check image quality  
- `/visualize` – Return visualization  

## Changes made

Standardized API responses
– All endpoints now return { success, data, error } (MERN-friendly).

Fixed JSON serialization
– Converted NumPy types (bool/float/int) to JSON-safe values to prevent errors.

Consistent error handling
– Same error structure across all endpoints.

CORS configured for MERN
– Allowed localhost:3000 and localhost:4000.

## Result: 
Flask API now returns clean JSON and works smoothly with the MERN stack.

## How to Run Locally
```bash
pip install -r requirements.txt
python app.py
