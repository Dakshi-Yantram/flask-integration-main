from unittest import result
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import cv2
import numpy as np
import json
import tempfile
import os
from datetime import datetime
from code1 import (
    analyze_image,
    analyze_colposcopy_single_image,
    analyze_pap_smear,
    classify_image_type,
    check_image_quality,
    plot_results,
    Config
)

app = Flask(__name__)
# CORS(app)
CORS(app, resources={
    r"/*": {
        "origins": ["http://localhost:3000", "http://localhost:4000"]
    }
})

# api_response is new
def api_response(success=True, data=None, error=None):
    return jsonify({
        "success": success,
        "data": data,
        "error": error
    })
def make_json_safe(obj):
    if isinstance(obj, dict):
        return {k: make_json_safe(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_json_safe(i) for i in obj]
    elif isinstance(obj, (np.bool_,)):
        return bool(obj)
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    else:
        return obj


@app.route('/analyze', methods=['POST'])
def analyze_endpoint():
    try:
        if 'image' not in request.files:
            return api_response(False, None, "No image provided"), 400

        file = request.files['image']
        img_array = np.frombuffer(file.read(), np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        if img is None:
            return api_response(False, None, "Could not decode image"), 400

        metadata = None
        if 'metadata' in request.form:
            try:
                metadata = json.loads(request.form['metadata'])
            except:
                metadata = None

        result = analyze_image(img, None, metadata)
        safe_result = make_json_safe(result)
        return api_response(True, safe_result)

    except Exception as e:
        return api_response(False, None, str(e)), 500

@app.route('/analyze/colposcopy', methods=['POST'])
def analyze_colposcopy_endpoint():
    """
    Endpoint specifically for colposcopy images
    """
    try:
        if 'image' not in request.files:
            # return jsonify({'error': 'No image provided'}), 400
            return api_response(False, None, "No image provided"), 400
        
        file = request.files['image']
        img_array = np.frombuffer(file.read(), np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        
        if img is None:
            # return jsonify({'error': 'Could not decode image'}), 400
            # return api_response(True, {"image_type": image_type}) 
            return api_response(False, None, "Could not decode image"), 400


        
        # Analyze as colposcopy
        result = analyze_colposcopy_single_image(img)
        
        # return jsonify(result)
        return api_response(True, result)        
    except Exception as e:
        return api_response(False, None, str(e)), 500


@app.route('/analyze/pap_smear', methods=['POST'])
def analyze_pap_smear_endpoint():
    """
    Endpoint specifically for pap smear images
    """
    try:
        if 'image' not in request.files:
            # return jsonify({'error': 'No image provided'}), 400
            return api_response(False, None, "No image provided"), 400
        
        file = request.files['image']
        img_array = np.frombuffer(file.read(), np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        
        if img is None:
            # return jsonify({'error': 'Could not decode image'}), 400
            # return api_response(True, {"image_type": image_type}) 
            return api_response(False, None, "Could not decode image"), 400


        
        # Analyze as pap smear
        result = analyze_pap_smear(img)
        
        # return jsonify(result)
        return api_response(True, result)        
    except Exception as e:
        return api_response(False, None, str(e)), 500


@app.route('/classify', methods=['POST'])
def classify_image():
    """
    Endpoint just to classify image type without full analysis
    """
    try:
        if 'image' not in request.files:
            # return jsonify({'error': 'No image provided'}), 400
            return api_response(False, None, "No image provided"), 400
        
        file = request.files['image']
        img_array = np.frombuffer(file.read(), np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        
        if img is None:
            # return jsonify({'error': 'Could not decode image'}), 400 
            # return api_response(True, {"image_type": image_type}) 
            return api_response(False, None, "Could not decode image"), 400


        
        # Classify image type
        image_type = classify_image_type(img)
        
        # return jsonify({'image_type': image_type}) 
        return api_response(True, {"image_type": image_type})

        
    except Exception as e:
        return api_response(False, None, str(e)), 500


@app.route('/quality', methods=['POST'])
def check_quality():
    """
    Endpoint to check image quality only
    """
    try:
        if 'image' not in request.files:
            # return jsonify({'error': 'No image provided'}), 400
            return api_response(False, None, "No image provided"), 400

        
        file = request.files['image']
        img_array = np.frombuffer(file.read(), np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        
        if img is None:
            # return jsonify({'error': 'Could not decode image'}), 400 
            # return api_response(True, {"image_type": image_type}) 
            return api_response(False, None, "Could not decode image"), 400


        
        # Check image quality
        image_type = classify_image_type(img)
        quality_result = check_image_quality(img, image_type)
        
        # return jsonify(quality_result) 
        return api_response(True, quality_result)

        
    except Exception as e:
        return api_response(False, None, str(e)), 500


@app.route('/visualize', methods=['POST'])
def visualize_results():
    """
    Endpoint to generate and return visualization
    """
    try:
        if 'image' not in request.files:
            # return jsonify({'error': 'No image provided'}), 400

            return api_response(False, None, "No image provided"), 400        
        file = request.files['image']
        img_array = np.frombuffer(file.read(), np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        
        if img is None:
            # return jsonify({'error': 'Could not decode image'}), 400 
            # return api_response(True, {"image_type": image_type}) 
            return api_response(False, None, "Could not decode image"), 400


        
        # Run analysis
        result = analyze_image(img, None, None)
        
        # Create visualization
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            plot_results(img, result, save_path=tmp.name)
            
            # Return image
            return send_file(tmp.name, mimetype='image/png')
        
    except Exception as e:
        return api_response(False, None, str(e)), 500


@app.route('/health', methods=['GET'])
def health_check():
    """
    Health check endpoint
    """
    return jsonify({
        'status': 'healthy', 
        'message': 'Cervical cancer analysis server is running',
        'timestamp': datetime.now().isoformat()
    })

@app.route('/config', methods=['GET'])
def get_config():
    """
    Endpoint to get current configuration
    """
    config_dict = {}
    for attr in dir(Config):
        if not attr.startswith('_'):
            config_dict[attr] = getattr(Config, attr)
    
    return jsonify(config_dict)

@app.route('/')
def home():
    return '''
    <h1>Cervical Cancer Analysis API</h1>
    <p>Server is running successfully!</p>
    <p>Available endpoints:</p>
    <ul>
        <li><b>GET /health</b> - Health check</li>
        <li><b>GET /config</b> - Get configuration</li>
        <li><b>POST /analyze</b> - Auto-detect and analyze image</li>
        <li><b>POST /analyze/colposcopy</b> - Analyze colposcopy image</li>
        <li><b>POST /analyze/pap_smear</b> - Analyze pap smear image</li>
        <li><b>POST /classify</b> - Classify image type only</li>
        <li><b>POST /quality</b> - Check image quality only</li>
        <li><b>POST /visualize</b> - Get visualization image</li>
    </ul>
    <p>Use Postman, curl, or the test form below to send requests.</p>
    
    <h3>Test Form:</h3>
    <form action="/analyze" method="post" enctype="multipart/form-data">
        <input type="file" name="image" accept="image/*" required>
        <input type="submit" value="Analyze Image">
    </form>
    '''
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
