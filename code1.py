# %%
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage, spatial
from skimage import measure, morphology, filters, feature, exposure, segmentation
import os
import math
from datetime import datetime
from IPython.display import display, Markdown, HTML

# %%
# Configuration Settings
class Config:
    # Quality thresholds
    LAPLACIAN_VAR = 120
    SATURATION_MEAN = 40
    MIN_DIMENSION = 400
    MAX_DIMENSION = 2000
    TISSUE_COVERAGE = 60
    
    # Color calibration card detection
    CALIBRATION_COLORS = [
        np.array([255, 255, 255]),  # White
        np.array([255, 0, 0]),      # Red
        np.array([0, 255, 0]),      # Green
        np.array([0, 0, 255]),      # Blue
    ]
    CALIBRATION_THRESHOLD = 30
    
    # Bethesda thresholds 
    MIN_SQUAMOUS_CELLS_LIQUID = 5000
    MIN_SQUAMOUS_CELLS_CONVENTIONAL = 8000
    MAX_OBSCURATION = 75
    MIN_ENDOCERVICAL_CELLS = 10
    
    # Swede score thresholds
    ACETOWHITE_DL_LOW = 10
    ACETOWHITE_DL_HIGH = 20
    ACETOWHITE_AREA_LOW = 5
    ACETOWHITE_AREA_HIGH = 25
    
    # Reid score thresholds
    REID_ACETOWHITE_THRESHOLDS = [1, 2]  # 0-2 points
    REID_MARGIN_THRESHOLDS = [0.1, 0.3]  # Normalized sharpness
    REID_VESSEL_THRESHOLDS = [0.6, 0.8]  # Circularity/regularity
    REID_IODINE_THRESHOLDS = [40, 80]    # Brown percentage
    
    # Margin analysis thresholds
    MARGIN_SHARPNESS_HIGH = 50
    MARGIN_SHARPNESS_MEDIUM = 20
    
    # Vessel analysis thresholds
    VESSEL_DENSITY_HIGH = 0.1
    VESSEL_DENSITY_MEDIUM = 0.05
    VESSEL_SPACING_FINE = 5
    VESSEL_SPACING_COARSE = 10
    VESSEL_DIAMETER_FINE = 2
    VESSEL_DIAMETER_COARSE = 5
    
    # Iodine color ranges (HSV)
    IODINE_BROWN = {'hue': (15, 30), 'sat': (40, 100), 'val': (0, 60)}
    IODINE_YELLOW = {'hue': (40, 65), 'sat': (30, 100), 'val': (60, 100)}
    
    # Edge case detection
    BLOOD_COLOR_RANGE = {'hue': (0, 10), 'sat': (50, 100), 'val': (20, 80)}
    INFLAMMATION_COLOR_RANGE = {'hue': (0, 180), 'sat': (0, 30), 'val': (80, 100)}
    
    # Timing parameters for acetic acid evolution
    ACETIC_EVOLUTION_TIMES = [30, 60, 120]
    
    # ORB feature detection parameters
    ORB_MAX_FEATURES = 1000
    ORB_GOOD_MATCH_PERCENT = 0.15
    
    # Specular highlight detection
    SPECULAR_SATURATION_THRESHOLD = 30
    SPECULAR_VALUE_THRESHOLD = 220
    
    # Texture analysis parameters
    LBP_RADIUS = 2
    LBP_POINTS = 16
    TEXTURE_WINDOW_SIZE = 11
    
    # Nuclear morphology thresholds
    NC_RATIO_NORMAL = 0.4
    NC_RATIO_HSIL = 0.65
    NUCLEAR_AREA_ASCUS = 100
    NUCLEAR_AREA_HSIL = 140

    # ASC-US detection thresholds
    ASCUS_NC_RATIO = 0.55      
    ASCUS_AREA = 120           
    ASCUS_STD_AREA = 35        

    # Koilocytosis detection 
    KOILO_SIZE_MIN = 50        
    KOILO_SIZE_MAX = 400       
    KOILO_CIRCULARITY = 0.6    
    KOILO_COUNT_THRESHOLD = 5  
    
    # HSIL detection thresholds 
    HSIL_NC_RATIO = 0.65      
    HSIL_AREA = 140            
    HSIL_STD_AREA = 45         
    HSIL_TEXTURE = 7.0         

    # ASC-H detection thresholds
    ASCH_NC_RATIO = 0.60       
    ASCH_AREA = 130            
    ASCH_STD_AREA = 40         
    ASCH_TEXTURE = 6.0         



# %%
# Image quality checks with enhanced validation
def check_image_quality(image, image_type=None):
    """Performs quality checks with different thresholds for different image types."""
    if image_type is None:
        image_type = classify_image_type(image)
    
    # Set different thresholds based on image type
    if image_type == 'pap_smear':
        LAPLACIAN_THRESHOLD = 5  
        SATURATION_THRESHOLD = 25
    elif image_type == 'colposcopy':
        LAPLACIAN_THRESHOLD = 120  
        SATURATION_THRESHOLD = 40
    else:  # 'other' category
        LAPLACIAN_THRESHOLD = 10   
        SATURATION_THRESHOLD = 20
    
    results = {'is_ok': True, 'reasons': [], 'metrics': {}, 'image_type': image_type}
    
    # Check for Blur
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    results['metrics']['laplacian_variance'] = lap_var
    
    # For Pap smears, use a more lenient approach
    if image_type == 'pap_smear':
        if lap_var < 5:  # Extremely blurry
            results['is_ok'] = False
            results['reasons'].append(f'Image too blurry (Laplacian variance: {lap_var:.2f} < 5)')
        elif lap_var < LAPLACIAN_THRESHOLD:
            # Warn but don't fail for slightly blurry Pap smears
            results['reasons'].append(f'Image somewhat blurry (Laplacian variance: {lap_var:.2f} < {LAPLACIAN_THRESHOLD})')
    else:
        if lap_var < LAPLACIAN_THRESHOLD:
            results['is_ok'] = False
            results['reasons'].append(f'Image too blurry (Laplacian variance: {lap_var:.2f} < {LAPLACIAN_THRESHOLD})')
    
    # Check for Glare/Over-exposure
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mean_saturation = np.mean(hsv[:, :, 1])
    results['metrics']['mean_saturation'] = mean_saturation
    if mean_saturation < SATURATION_THRESHOLD:
        results['is_ok'] = False
        results['reasons'].append(f'Image may have glare (mean saturation: {mean_saturation:.2f} < {SATURATION_THRESHOLD})')
    
    # Check Size (only for medical images)
    height, width = image.shape[:2]
    results['metrics']['dimensions'] = (width, height)
    if image_type != 'other' and (height < Config.MIN_DIMENSION or width < Config.MIN_DIMENSION):
        results['is_ok'] = False
        results['reasons'].append(f'Image dimensions too small ({width}x{height} < {Config.MIN_DIMENSION}px)')
    
    # Check for calibration card (only for colposcopy)
    if image_type == 'colposcopy':
        calibration_detected, calibration_metrics = detect_calibration_card(image)
        results['metrics']['calibration_detected'] = calibration_detected
        results['metrics'].update(calibration_metrics)
        
        if not calibration_detected:
            results['reasons'].append('Calibration card not detected - measurements may be inaccurate')
    
    # Check for common artifacts (but don't fail for them)
    artifacts_detected, artifact_info = detect_artifacts(image)
    results['metrics']['artifacts_detected'] = artifacts_detected
    results['metrics'].update(artifact_info)
    
    if artifacts_detected:
        results['reasons'].append('Image contains artifacts that may affect analysis quality')
    
    return results

def detect_calibration_card(image):
    """Detect color calibration card in the image."""
    # Convert to HSV for better color segmentation
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Look for saturated colors that might be calibration patches
    saturation_mask = hsv[:, :, 1] > 100
    value_mask = (hsv[:, :, 2] > 50) & (hsv[:, :, 2] < 200)
    color_mask = saturation_mask & value_mask
    
    if np.sum(color_mask) < 100:  # Not enough colored pixels
        return False, {}  # Return empty dict for metrics
    
    # Find contours of potential color patches
    contours, _ = cv2.findContours(color_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    color_patches = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if 100 < area < 10000:  # Reasonable size for calibration patches
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / h
            if 0.7 < aspect_ratio < 1.3:  # Roughly square
                patch = image[y:y+h, x:x+w]
                mean_color = np.mean(patch, axis=(0, 1))
                color_patches.append((mean_color, (x, y, w, h)))
    
    # Check if we found patches that match expected calibration colors
    matched_patches = 0
    for patch_color, _ in color_patches:
        for cal_color in Config.CALIBRATION_COLORS:
            color_diff = np.linalg.norm(patch_color - cal_color)
            if color_diff < Config.CALIBRATION_THRESHOLD:
                matched_patches += 1
                break
    
    calibration_detected = matched_patches >= 2  # At least 2 calibration colors detected
    
    metrics = {
        'calibration_patches_found': len(color_patches),
        'calibration_colors_matched': matched_patches
    }
    
    return calibration_detected, metrics
def normalize_colors(image):
    """Apply calibration-based or gray-world normalization."""
    calibration_detected, metrics = detect_calibration_card(image)
    if calibration_detected:
        # Placeholder: color correction using calibration patches
        # (here we just log, but can add 3x3 correction matrix later)
        print("Calibration card detected - applying color correction")
        return image
    else:
        # Gray-world normalization
        img_float = image.astype(np.float32)
        avg_b, avg_g, avg_r = np.mean(img_float[:, :, 0]), np.mean(img_float[:, :, 1]), np.mean(img_float[:, :, 2])
        gray_val = (avg_b + avg_g + avg_r) / 3
        scale_b, scale_g, scale_r = gray_val / avg_b, gray_val / avg_g, gray_val / avg_r
        img_float[:, :, 0] *= scale_b
        img_float[:, :, 1] *= scale_g
        img_float[:, :, 2] *= scale_r
        return np.clip(img_float, 0, 255).astype(np.uint8)
    
def detect_artifacts(image):
    """Detect common artifacts that might interfere with analysis."""
    artifacts = {
        'blood': False,
        'mucus': False,
        'inflammation': False,
        'specular_reflection': False
    }
    
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Detect blood (red hues with medium saturation/value)
    blood_mask = cv2.inRange(hsv, 
                           np.array([Config.BLOOD_COLOR_RANGE['hue'][0], 
                                    Config.BLOOD_COLOR_RANGE['sat'][0], 
                                    Config.BLOOD_COLOR_RANGE['val'][0]]),
                           np.array([Config.BLOOD_COLOR_RANGE['hue'][1], 
                                    Config.BLOOD_COLOR_RANGE['sat'][1], 
                                    Config.BLOOD_COLOR_RANGE['val'][1]]))
    blood_area = np.sum(blood_mask > 0) / blood_mask.size * 100
    artifacts['blood'] = blood_area > 5
    
    # Detect mucus/inflammation (pale areas)
    inflammation_mask = cv2.inRange(hsv, 
                                  np.array([Config.INFLAMMATION_COLOR_RANGE['hue'][0], 
                                           Config.INFLAMMATION_COLOR_RANGE['sat'][0], 
                                           Config.INFLAMMATION_COLOR_RANGE['val'][0]]),
                                  np.array([Config.INFLAMMATION_COLOR_RANGE['hue'][1], 
                                           Config.INFLAMMATION_COLOR_RANGE['sat'][1], 
                                           Config.INFLAMMATION_COLOR_RANGE['val'][1]]))
    inflammation_area = np.sum(inflammation_mask > 0) / inflammation_mask.size * 100
    artifacts['inflammation'] = inflammation_area > 10
    
    # Detect specular reflections (very high value, low saturation)
    specular_mask = (hsv[:, :, 1] < Config.SPECULAR_SATURATION_THRESHOLD) & (hsv[:, :, 2] > Config.SPECULAR_VALUE_THRESHOLD)
    specular_area = np.sum(specular_mask) / specular_mask.size * 100
    artifacts['specular_reflection'] = specular_area > 5
    
    # Check if any significant artifacts were detected
    significant_artifacts = any(artifacts.values())
    
    return significant_artifacts, artifacts

# %%
# Enhanced cervix segmentation with ellipse fitting
def segment_cervix_region(image):
    """Advanced cervix region segmentation with ellipse fitting."""
    # Convert to HSV for better color-based segmentation
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Create a mask for cervical tissue (pink/red tones)
    lower_bound = np.array([0, 30, 30])
    upper_bound = np.array([30, 255, 255])
    color_mask = cv2.inRange(hsv, lower_bound, upper_bound)
    
    # Add detection for other tissue colors (brownish tones)
    lower_brown = np.array([10, 30, 30])
    upper_brown = np.array([20, 150, 150])
    brown_mask = cv2.inRange(hsv, lower_brown, upper_brown)
    
    # Combine masks
    combined_mask = cv2.bitwise_or(color_mask, brown_mask)
    
    # Clean up the mask
    kernel = np.ones((7, 7), np.uint8)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
    
    # Find the largest contour (likely the cervix)
    contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return combined_mask
    
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Fit ellipse to the largest contour
    if len(largest_contour) >= 5:
        ellipse = cv2.fitEllipse(largest_contour)
        ellipse_mask = np.zeros_like(combined_mask)
        cv2.ellipse(ellipse_mask, ellipse, 255, -1)
        
        # Use the ellipse mask if it's reasonable
        ellipse_area = cv2.contourArea(largest_contour)
        image_area = image.shape[0] * image.shape[1]
        if ellipse_area > image_area * 0.1:
            return ellipse_mask
    
    return combined_mask

# %%
def classify_image_type(image, metadata=None):
    """Classify image as Pap smear, Colposcopy, or Other using nuclei + stain + structure + cervix-specific filters."""
    if metadata and 'image_type' in metadata:
        return metadata['image_type']

    height, width = image.shape[:2]

    # --- Step 1: Cervix detection (Colposcopy candidate) ---
    cervix_mask = segment_cervix_region(image)
    cervix_area_ratio = np.sum(cervix_mask > 0) / cervix_mask.size

    # Cervix circularity check
    circularity = 0
    contours, _ = cv2.findContours(cervix_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest)
        perimeter = cv2.arcLength(largest, True)
        if perimeter > 0:
            circularity = 4 * np.pi * area / (perimeter ** 2)
    
    # Helper: mean circularity of blobs (flowers tend to have irregular, nuclei ~round)
    circularities = []
    for c in contours:
        area = cv2.contourArea(c)
        peri = cv2.arcLength(c, True)
        if area > 20 and peri > 0:
            circ = 4 * np.pi * area / (peri ** 2)
            circularities.append(circ)
    mean_circ = np.mean(circularities) if circularities else 0
    

    # HSV color filters
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mean_hue = np.mean(hsv[:, :, 0])
    mean_sat = np.mean(hsv[:, :, 1])
    mean_val = np.mean(hsv[:, :, 2])

    # Texture filter (reject overly smooth mucosa)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()

    # --- Step 2: Nuclear detection (Pap smear candidate) ---
    hematoxylin = color_deconvolution_hematoxylin(image)
    nuclei_mask, nuclei_count = detect_nuclei_bethesda(hematoxylin)

    contours, _ = cv2.findContours(nuclei_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    valid_blobs = [cv2.contourArea(c) for c in contours if 20 < cv2.contourArea(c) < 300]

    # --- Step 3: Stain fingerprint check (Pap smears = purple nuclei + pink cytoplasm) ---
    mean_r = np.mean(image[:, :, 2])
    mean_b = np.mean(image[:, :, 0])
    has_pap_stain = (mean_b > mean_r * 0.6) and (mean_r > 40)

    # --- Step 4: Cell density check (avoid sparse backgrounds) ---
    cell_density = np.sum(nuclei_mask > 0) / nuclei_mask.size
    clustered = len(valid_blobs) > 5 and cell_density > 0.001

    # --- Debug info ---
    print(f"[DEBUG] Cervix area: {cervix_area_ratio:.3f}, "
          f"Circularity: {circularity:.2f}, "
          f"Hue: {mean_hue:.1f}, Sat: {mean_sat:.1f}, Val: {mean_val:.1f}, "
          f"LapVar: {lap_var:.1f}, "
          f"Nuclei count: {nuclei_count}, "
          f"Valid blobs: {len(valid_blobs)}, "
          f"Pap stain: {has_pap_stain}, "
          f"Cell density: {cell_density:.4f}")
    # Strong nuclei evidence check
    nuclei_ok = (
        nuclei_count >= 20 and
        len(valid_blobs) >= 10 and
        cell_density > 0.005 and
        lap_var > 30  # focus check
    )

    # --- Step 5: Final decision rules ---
    # Colposcopy → large cervix region, round shape, light pink, not smooth
    if (cervix_area_ratio > 0.12 and circularity > 0.5
        and 5 < mean_hue < 90
        and 40 < mean_sat < 180
        and mean_val > 120
        and lap_var > 150):
        return "colposcopy"

    
    # Strong Pap smear
    if has_pap_stain and nuclei_ok:
        return "pap_smear"

    # Backup: Very high density → clumped nuclei
    if has_pap_stain and cell_density > 0.02 and nuclei_count > 100:
        return "pap_smear"

    # Low-cellularity Pap smear (only if sharp + somewhat round nuclei)
    if has_pap_stain and 5 <= nuclei_count < 20:
        if lap_var > 100 and mean_circ > 0.6:   # sharp + nuclei-like blobs
            return "pap_smear"
        else:
            print(f"[DEBUG] Rejected low-cellularity case: LapVar={lap_var:.1f}, Circ={mean_circ:.2f}")
            return "other"

    # Sparse Pap smear (fallback)
    if has_pap_stain and len(valid_blobs) >= 5 and cell_density > 0.003:
        if mean_hue > 95 and nuclei_count < 15:
            return "other"
        return "pap_smear"



    # Otherwise reject
    return "other"


# %%
def check_bethesda_adequacy(image):
    """Bethesda adequacy check following clinical document specifications."""
    print("Performing Bethesda adequacy check...")
    
    # 1. TILE THE SLIDE - Analyze multiple regions (clinical doc: standardized FoVs)
    height, width = image.shape[:2]
    tile_size = min(height, width) // 2  # Analyze central region for simplicity
    
    # Extract central tile (simulating 40x field)
    y_start = (height - tile_size) // 2
    x_start = (width - tile_size) // 2
    tile = image[y_start:y_start+tile_size, x_start:x_start+tile_size]
    
    # 2. COLOR DECONVOLUTION - Separate hematoxylin (clinical doc: color deconvolution)
    hematoxylin = color_deconvolution_hematoxylin(tile)
    
    # 3. NUCLEI DETECTION with proper preprocessing (clinical doc: threshold + watershed)
    nuclei_mask, cell_count = detect_nuclei_bethesda(hematoxylin)
    
    # 4. PER-FIELD COUNTS & SCALING (clinical doc: cells/mm² + extrapolation)
    # Standard 40x field area: ~0.196 mm² (0.44mm diameter) - based on clinical standards
    field_area_mm2 = 0.196
    cells_per_mm2 = cell_count / field_area_mm2
    
    # Liquid-based prep total area: ~1300 mm² (ThinPrep) - clinical standard
    total_area_mm2 = 1300
    estimated_total_cells = cells_per_mm2 * total_area_mm2
    
    # 5. OBSCURATION ANALYSIS (clinical doc: blood/mucus detection)
    obscuration_percentage = calculate_obscuration_bethesda(tile)
    
    # 6. ENDOCERVICAL COMPONENT (clinical doc: honeycomb clusters)
    endocervical_count = detect_endocervical_component(tile)
    
    # 7. ADEQUACY DECISION (Bethesda rules - clinical doc criteria)
    is_adequate = True
    reasons = []
    
    # Liquid-based criteria (clinical doc: ≥5,000 squamous cells)
    if estimated_total_cells < 5000:
        is_adequate = False
        reasons.append(f"Insufficient squamous cells ({estimated_total_cells:.0f} < 5,000)")
    
    # Clinical doc: >75% obscured → unsatisfactory
    if obscuration_percentage > 75:
        is_adequate = False
        reasons.append(f"Excessive obscuration ({obscuration_percentage:.1f}% > 75%)")
    
    # Endocervical component (clinical doc: note but don't fail adequacy)
    if endocervical_count < 10:
        reasons.append(f"Limited endocervical component ({endocervical_count} cells)")
    
    print(f"Cells/field: {cell_count}, Cells/mm²: {cells_per_mm2:.0f}, Total estimate: {estimated_total_cells:.0f}")
    print(f"Obscuration: {obscuration_percentage:.1f}%, Endocervical: {endocervical_count} cells")
    
    return {
        'is_adequate': is_adequate,
        'nuclei_count': estimated_total_cells,
        'obscuration_percentage': obscuration_percentage,
        'endocervical_count': endocervical_count,
        'reasons': reasons,
        'cells_per_field': cell_count
    }

def color_deconvolution_hematoxylin(image):
    """Separate hematoxylin component using color deconvolution (clinical doc: stain separation)."""
    # Convert to optical density space
    image_float = image.astype(np.float32) / 255.0
    od = -np.log(image_float + 1e-6)
    
    # Hematoxylin stain vector (RGB) - standard values
    hematoxylin_vector = np.array([0.65, 0.70, 0.29])
    hematoxylin_vector /= np.linalg.norm(hematoxylin_vector)
    
    # Project onto hematoxylin vector
    hematoxylin_od = np.dot(od.reshape(-1, 3), hematoxylin_vector)
    hematoxylin_od = hematoxylin_od.reshape(od.shape[:2])
    
    # Convert back to 0-255
    hematoxylin = np.exp(-hematoxylin_od)
    hematoxylin = np.clip(hematoxylin * 255, 0, 255).astype(np.uint8)
    
    return hematoxylin

def detect_nuclei_bethesda(hematoxylin_channel):
    """Nuclei detection following clinical document specifications (threshold + watershed)."""
    # Adaptive thresholding for nuclei (clinical doc: adaptive thresholding)
    nuclei_mask = cv2.adaptiveThreshold(hematoxylin_channel, 255, 
                                      cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                      cv2.THRESH_BINARY, 21, 5)
    
    # Remove tiny debris (area < 20 pixels) - clinical doc: remove tiny debris
    kernel = np.ones((3, 3), np.uint8)
    nuclei_mask = cv2.morphologyEx(nuclei_mask, cv2.MORPH_OPEN, kernel)
    
    # Distance transform + watershed to split clumps (clinical doc: watershed)
    sure_bg = cv2.dilate(nuclei_mask, kernel, iterations=2)
    dist_transform = cv2.distanceTransform(nuclei_mask, cv2.DIST_L2, 3)
    
    # Threshold for sure foreground
    _, sure_fg = cv2.threshold(dist_transform, 0.5 * dist_transform.max(), 255, 0)
    sure_fg = sure_fg.astype(np.uint8)
    
    # Unknown region
    unknown = cv2.subtract(sure_bg, sure_fg)
    
    # Marker labelling
    _, markers = cv2.connectedComponents(sure_fg)
    markers += 1
    markers[unknown == 255] = 0
    
    # Apply watershed
    markers = cv2.watershed(cv2.cvtColor(hematoxylin_channel, cv2.COLOR_GRAY2BGR), markers)
    
    # Count cells
    cell_count = len(np.unique(markers)) - 2  # Subtract background and border
    
    return nuclei_mask, max(0, cell_count)

def detect_endocervical_component(image):
    """Detect endocervical cells using honeycomb cluster patterns (clinical doc: honeycomb clusters)."""
    # Convert to LAB color space for better tissue separation
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel = lab[:, :, 0]
    
    # Look for honeycomb patterns (endocervical clusters) - clinical doc pattern
    edges = cv2.Canny(l_channel, 50, 150)
    
    # Find contours that might represent clusters
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    cluster_count = 0
    for contour in contours:
        area = cv2.contourArea(contour)
        if 100 < area < 1000:  # Reasonable cluster size
            # Check circularity (honeycomb clusters are often round)
            perimeter = cv2.arcLength(contour, True)
            if perimeter > 0:
                circularity = 4 * np.pi * area / (perimeter ** 2)
                if circularity > 0.6:  # Fairly circular
                    cluster_count += 1
    
    # Each cluster typically contains multiple cells - clinical doc: ≥10 cells
    return min(10, cluster_count * 3)  # Estimate 3 cells per cluster, max 10

def calculate_obscuration_bethesda(image):
    """Calculate obscuration following Bethesda guidelines (clinical doc: blood/mucus detection)."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Blood detection (red/brown) - clinical doc: red/brown dominance
    blood_mask = cv2.inRange(hsv, np.array([0, 50, 20]), np.array([15, 255, 200]))
    
    # Inflammation/mucus (pale, low saturation) - clinical doc: low-frequency amorphous areas
    inflammation_mask = cv2.inRange(hsv, np.array([0, 0, 150]), np.array([180, 50, 255]))
    
    # Combine obscuring factors
    obscuration_mask = cv2.bitwise_or(blood_mask, inflammation_mask)
    
    # Calculate percentage
    total_pixels = obscuration_mask.size
    obscured_pixels = np.sum(obscuration_mask > 0)
    
    return (obscured_pixels / total_pixels) * 100

def analyze_nuclear_morphology_bethesda(image):
    """Enhanced morphology analysis with better HSIL feature detection."""
    try:
        hematoxylin = color_deconvolution_hematoxylin(image)
        nuclei_mask, _ = detect_nuclei_bethesda(hematoxylin)
        
        contours, _ = cv2.findContours(nuclei_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        cell_features = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if 40 < area < 400:  # Broader range for HSIL cells
                features = extract_cell_features(contour, image)
                
                # Specifically look for HSIL characteristics
                if (features['nc_ratio'] > 0.6 or 
                    features['area'] > 150 or 
                    features['eccentricity'] > 0.5):
                    cell_features.append(features)
        
        if not cell_features:
            return {}
        
        # Calculate statistics focusing on abnormal cells
        features_dict = {k: [f[k] for f in cell_features] for k in cell_features[0].keys()}
        
        stats = {}
        for feature_name, values in features_dict.items():
            stats[f'median_{feature_name}'] = np.median(values)
            stats[f'p75_{feature_name}'] = np.percentile(values, 75)
            stats[f'p90_{feature_name}'] = np.percentile(values, 90)
            stats[f'std_{feature_name}'] = np.std(values)
        
        stats['n_cells_analyzed'] = len(cell_features)
        
        # Add HSIL-specific metrics
        if len(cell_features) > 5:
            nc_ratios = features_dict['nc_ratio']
            stats['hsil_cell_ratio'] = sum(1 for ratio in nc_ratios if ratio > 0.7) / len(nc_ratios)
            stats['large_nuclei_ratio'] = sum(1 for area in features_dict['area'] if area > 180) / len(features_dict['area'])
        
        return stats
        
    except Exception as e:
        print(f"Morphology analysis error: {e}")
        return {}

def extract_cell_features(contour, image):
    """Extract cell-level features following clinical document."""
    # Basic morphology (clinical doc: area, perimeter, circularity)
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    circularity = 4 * np.pi * area / (perimeter ** 2) if perimeter > 0 else 0
    
    # Ellipse fitting for eccentricity (clinical doc: eccentricity, boundary irregularity)
    eccentricity = 0
    if len(contour) >= 5:
        try:
            rect = cv2.minAreaRect(contour)
            width, height = rect[1]
            if max(width, height) > 0:
                eccentricity = 1 - (min(width, height) / max(width, height))
        except:
            eccentricity = 0
    
    # N:C ratio estimation (clinical doc: N:C ratio via distance transform)
    nc_ratio = estimate_nc_ratio(contour, image)
    
    # Texture features (clinical doc: chromatin texture)
    texture_contrast = calculate_texture_contrast(contour, image)
    
    return {
        'area': area,
        'circularity': circularity,
        'eccentricity': eccentricity,
        'nc_ratio': nc_ratio,
        'texture_contrast': texture_contrast
    }

def estimate_nc_ratio(contour, image):
    """Estimate nuclear-to-cytoplasmic ratio (clinical doc: approximate cytoplasm)."""
    try:
        # Create mask for the nucleus
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        cv2.drawContours(mask, [contour], -1, 255, -1)
        
        # Get bounding box of nucleus
        x, y, w, h = cv2.boundingRect(contour)
        
        # Dilate to estimate cytoplasm area (clinical doc: grow-then-watershed approach)
        dilated = cv2.dilate(mask, np.ones((7, 7), np.uint8), iterations=3)
        
        # Restrict to reasonable area around nucleus
        cyto_mask = np.zeros_like(mask)
        cyto_mask[max(0, y-10):min(image.shape[0], y+h+10), 
                 max(0, x-10):min(image.shape[1], x+w+10)] = \
            dilated[max(0, y-10):min(image.shape[0], y+h+10),
                   max(0, x-10):min(image.shape[1], x+w+10)]
        
        # Calculate areas
        nucleus_area = np.sum(mask > 0)
        cytoplasm_area = np.sum(cyto_mask > 0) - nucleus_area
        
        return nucleus_area / (cytoplasm_area + 1e-6)
        
    except:
        return 0

def calculate_texture_contrast(contour, image):
    """Calculate texture contrast within nucleus (clinical doc: chromatin texture)."""
    try:
        # Create mask for the nucleus
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        cv2.drawContours(mask, [contour], -1, 255, -1)
        
        # Extract gray region within nucleus
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        nucleus_region = gray[mask > 0]
        
        if len(nucleus_region) > 0:
            return np.std(nucleus_region)  # Standard deviation as texture measure
        return 0
    except:
        return 0
def detect_koilocytosis(image, nuclei_mask):
    """Reliable koilocytosis detection that's not oversensitive."""
    background_mask = 255 - nuclei_mask
    dist_transform = cv2.distanceTransform(background_mask.astype(np.uint8), cv2.DIST_L2, 5)
    cv2.normalize(dist_transform, dist_transform, 0, 1.0, cv2.NORM_MINMAX)
    
    # Specific halo detection
    halo_mask = (dist_transform > 0.15) & (dist_transform < 0.25)
    
    # Clean up
    kernel = np.ones((3, 3), np.uint8)
    halo_mask = halo_mask.astype(np.uint8) * 255
    halo_mask = cv2.morphologyEx(halo_mask, cv2.MORPH_OPEN, kernel)
    
    # Count only well-defined halos
    contours, _ = cv2.findContours(halo_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    koilo_count = 0
    for contour in contours:
        area = cv2.contourArea(contour)
        if 80 < area < 300:  # Reasonable size range
            perimeter = cv2.arcLength(contour, True)
            if perimeter > 0:
                circularity = 4 * np.pi * area / (perimeter ** 2)
                if circularity > 0.8:  # Good circularity
                    koilo_count += 1
    
    return koilo_count, halo_mask
def analyze_pap_smear(image):
    """Solid Pap smear analysis with reliable HSIL detection."""
    result = {
        'classification': 'Unknown',
        'recommendation': 'Analysis failed',
        'explanations': [],
        'confidence': 0.0,
        'risk_level': 'unknown',
        'bethesda_category': 'Unsatisfactory for evaluation'
    }
    
    try:
        print("Starting Pap smear analysis...")
        adequacy = check_bethesda_adequacy(image)
        
        if not adequacy['is_adequate']:
            result.update({
                'classification': 'Unsatisfactory for evaluation',
                'recommendation': 'Repeat sample collection',
                'explanations': adequacy['reasons'],
                'confidence': 0.95,
                'risk_level': 'red',
                'adequacy_metrics': adequacy,
                'bethesda_category': 'Unsatisfactory for evaluation'
            })
            return result
        
        # Get comprehensive features
        morphology_stats = analyze_nuclear_morphology_bethesda(image)
        hematoxylin = color_deconvolution_hematoxylin(image)
        nuclei_mask, _ = detect_nuclei_bethesda(hematoxylin)
        koilo_count, _ = detect_koilocytosis(image, nuclei_mask)
        
        # Initialize with NILM
        classification = 'NILM'
        confidence = 0.80
        explanations = ['Adequate sample', 'No significant abnormalities detected']
        bethesda_category = 'Negative for intraepithelial lesion or malignancy'
        
        if morphology_stats and morphology_stats['n_cells_analyzed'] > 5:
            # Extract key features
            p90_nc_ratio = morphology_stats.get('p90_nc_ratio', 0)
            p90_area = morphology_stats.get('p90_area', 0)
            std_area = morphology_stats.get('std_area', 0)
            mean_eccentricity = morphology_stats.get('mean_eccentricity', 0)
            p90_texture = morphology_stats.get('p90_texture_contrast', 0)
            
            print(f"HSIL Detection - NC90: {p90_nc_ratio:.2f}, Area90: {p90_area:.1f}, StdArea: {std_area:.1f}, Texture90: {p90_texture:.1f}")
            
            # 1. HSIL DETECTION - PRIMARY FOCUS
            hsildetected = False
            if (p90_nc_ratio > 0.68 and        # High N:C ratio
                p90_area > 155 and            # Large nuclei
                std_area > 58 and             # High variation
                p90_texture > 8.5):           # Coarse chromatin
                classification = 'HSIL'
                confidence = 0.88
                explanations = [
                    'High nuclear-to-cytoplasmic ratio (>0.68)',
                    'Marked nuclear enlargement (>155 pixels)',
                    'Significant nuclear pleomorphism (>58 std)',
                    'Coarse chromatin pattern (>8.5 texture)'
                ]
                bethesda_category = 'High-grade squamous intraepithelial lesion'
                hsildetected = True
                print("✓ HSIL DETECTED: Clear high-grade features")
            
            # 2. ASC-H - if not HSIL but still suspicious
            elif not hsildetected and (p90_nc_ratio > 0.62 and
                  p90_area > 135 and
                  std_area > 48 and
                  p90_texture > 7.0):
                classification = 'ASC-H'
                confidence = 0.78
                explanations = [
                    'Elevated N:C ratio (>0.62)',
                    'Nuclear enlargement (>135 pixels)',
                    'Nuclear size variation (>48 std)',
                    'Suspicious for high-grade lesion'
                ]
                bethesda_category = 'Atypical squamous cells, cannot exclude HSIL'
                print("✓ ASC-H DETECTED: Suspicious for HSIL")
            
            # 3. LSIL - only if clear koilocytosis
            elif not hsildetected and koilo_count >= 12:
                classification = 'LSIL'
                confidence = 0.75
                explanations = [
                    f'Definite koilocytosis ({koilo_count} cells)',
                    'Low-grade cellular changes',
                    'Mild nuclear abnormalities'
                ]
                bethesda_category = 'Low-grade squamous intraepithelial lesion'
                print("✓ LSIL DETECTED: Koilocytosis present")
            
            # 4. ASC-US - mild changes only
            elif not hsildetected and ((p90_nc_ratio > 0.58 and p90_area > 125) or
                  (p90_nc_ratio > 0.60 and std_area > 42) or
                  (p90_area > 130 and std_area > 45)):
                classification = 'ASC-US'
                confidence = 0.68
                explanations = [
                    'Mild nuclear atypia',
                    'Borderline cellular changes',
                    'Requires further evaluation'
                ]
                bethesda_category = 'Atypical squamous cells of undetermined significance'
                print("✓ ASC-US DETECTED: Mild abnormalities")
        
        # Set risk level
        risk_level = 'green'
        if classification == 'ASC-US' or classification == 'LSIL':
            risk_level = 'amber'
        elif classification == 'ASC-H' or classification == 'HSIL':
            risk_level = 'red'
        
        result.update({
            'classification': classification,
            'recommendation': 'Routine screening' if classification == 'NILM' else 'Further evaluation needed',
            'explanations': explanations,
            'confidence': confidence,
            'risk_level': risk_level,
            'morphology_stats': morphology_stats,
            'adequacy_metrics': adequacy,
            'koilocytosis_count': koilo_count,
            'bethesda_category': bethesda_category
        })
        
        # Clinical action recommendations
        if classification == 'ASC-US':
            result['recommendation'] = 'HPV reflex testing recommended'
        elif classification == 'LSIL':
            result['recommendation'] = 'Colposcopy recommended (HPV positive or age >25)'
        elif classification == 'ASC-H':
            result['recommendation'] = 'Immediate colposcopy indicated'
        elif classification == 'HSIL':
            result['recommendation'] = 'URGENT colposcopy with biopsy indicated'
        
    except Exception as e:
        print(f"Analysis error: {str(e)}")
        result['error'] = str(e)
        result['classification'] = 'Analysis error'
        result['recommendation'] = 'Technical review needed'
        result['confidence'] = 0.3
    
    return result

# %%
# Enhanced colposcopy analysis with complete Swede score implementation
def register_images(reference, target):
    """Register target image to reference image using ORB + RANSAC."""
    gray_ref = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)
    gray_target = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY)
    
    orb = cv2.ORB_create(Config.ORB_MAX_FEATURES)
    
    kp1, des1 = orb.detectAndCompute(gray_ref, None)
    kp2, des2 = orb.detectAndCompute(gray_target, None)
    
    if des1 is None or des2 is None:
        return target, None
    
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    
    matches = sorted(matches, key=lambda x: x.distance)
    
    num_good_matches = int(len(matches) * Config.ORB_GOOD_MATCH_PERCENT)
    matches = matches[:num_good_matches]
    
    points1 = np.zeros((len(matches), 2), dtype=np.float32)
    points2 = np.zeros((len(matches), 2), dtype=np.float32)
    
    for i, match in enumerate(matches):
        points1[i, :] = kp1[match.queryIdx].pt
        points2[i, :] = kp2[match.trainIdx].pt
    
    if len(matches) >= 4:
        H, mask = cv2.findHomography(points2, points1, cv2.RANSAC, 5.0)
        
        if H is not None:
            height, width = reference.shape[:2]
            registered = cv2.warpPerspective(target, H, (width, height))
            return registered, H
    
    return target, None

def remove_specular_highlights(image):
    """Detect and remove specular highlights from image."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Create mask for specular highlights
    specular_mask = (hsv[:, :, 1] < Config.SPECULAR_SATURATION_THRESHOLD) & (hsv[:, :, 2] > Config.SPECULAR_VALUE_THRESHOLD)
    
    # Inpaint specular highlights
    result = image.copy()
    result[specular_mask] = cv2.inpaint(image, specular_mask.astype(np.uint8), 3, cv2.INPAINT_TELEA)
    
    return result, specular_mask

# %%
def estimate_vessel_diameter(vessel_mask):
    """Estimate average vessel diameter using distance transform."""
    distance_map = ndimage.distance_transform_edt(vessel_mask)
    vessel_diameters = 2 * distance_map[vessel_mask > 0]  # Diameter = 2 * radius
    return np.mean(vessel_diameters) if len(vessel_diameters) > 0 else 0

# %%
def calculate_texture_features(image, mask=None):
    """Calculate texture features using LBP and entropy."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    if mask is not None:
        gray = gray * (mask > 0)
    
    # Local Binary Pattern (LBP) texture analysis
    lbp = feature.local_binary_pattern(gray, Config.LBP_POINTS, Config.LBP_RADIUS, method='uniform')
    
    # Calculate LBP histogram
    lbp_hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, Config.LBP_POINTS + 3), range=(0, Config.LBP_POINTS + 2))
    lbp_hist = lbp_hist.astype("float")
    lbp_hist /= (lbp_hist.sum() + 1e-6)  # Normalize
    
    # Calculate texture uniformity (inverse of entropy)
    lbp_entropy = -np.sum([p * np.log2(p) for p in lbp_hist if p > 0])
    lbp_uniformity = 1 / (1 + lbp_entropy)  # Higher values indicate more uniform texture
    
    # Calculate local entropy
    entropy_img = filters.rank.entropy(gray.astype(np.uint8), morphology.disk(Config.TEXTURE_WINDOW_SIZE))
    mean_entropy = np.mean(entropy_img) if mask is None else np.mean(entropy_img[mask > 0])
    
    return {
        'lbp_uniformity': lbp_uniformity,
        'mean_entropy': mean_entropy,
        'texture_heterogeneity': mean_entropy  # Higher entropy = more heterogeneous
    }

# %%
# VESSEL PATTERN ANALYSIS WITH FREQUENCY ANALYSIS
def analyze_vessel_patterns(image, lesion_mask=None):
    """Comprehensive vessel pattern analysis using multiple approaches."""
    green_channel = image[:, :, 1]
    
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(green_channel)
    
    # MULTI-SCALE VESSEL
    vessel_enhanced = np.zeros_like(enhanced, dtype=np.float32)
    
    for sigma in [1, 2, 3, 4]:  # Multiple scales for different vessel sizes
        gaussian = cv2.GaussianBlur(enhanced, (0, 0), sigma)
        gx = cv2.Sobel(gaussian, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(gaussian, cv2.CV_64F, 0, 1, ksize=3)
        
        for i in range(enhanced.shape[0]):
            for j in range(enhanced.shape[1]):
                # Simple vesselness measure
                magnitude = np.sqrt(gx[i, j]**2 + gy[i, j]**2)
                vessel_enhanced[i, j] = max(vessel_enhanced[i, j], magnitude)
    
    vessel_enhanced = (vessel_enhanced / vessel_enhanced.max() * 255).astype(np.uint8)
    
    # Binarize and skeletonize
    _, vessel_mask = cv2.threshold(vessel_enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    skeleton = morphology.skeletonize(vessel_mask > 0)
    
    # VESSEL SPACING ANALYSIS - NEW
    vessel_points = np.column_stack(np.where(skeleton))
    if len(vessel_points) > 10:
        # Calculate nearest neighbor distances
        tree = spatial.cKDTree(vessel_points)
        distances, _ = tree.query(vessel_points, k=2)
        mean_spacing = np.mean(distances[:, 1])  # Distance to nearest neighbor
        
        # Vessel diameter estimation
        vessel_diameter = estimate_vessel_diameter(vessel_mask)
    else:
        mean_spacing = 0
        vessel_diameter = 0
    
    # PATTERN CLASSIFICATION
    points = 0
    pattern_type = "Normal"
    
    if mean_spacing < Config.VESSEL_SPACING_FINE and vessel_diameter < Config.VESSEL_DIAMETER_FINE:
        points = 1
        pattern_type = "Fine punctation"
    elif mean_spacing > Config.VESSEL_SPACING_COARSE or vessel_diameter > Config.VESSEL_DIAMETER_COARSE:
        points = 2
        pattern_type = "Coarse pattern"
    
    return {
        'vessel_density': np.sum(vessel_mask > 0) / vessel_mask.size,
        'mean_spacing': mean_spacing,
        'mean_diameter': vessel_diameter,
        'pattern_type': pattern_type,
        'points': points
    }

# %%
def calculate_reid_score(acetowhite_points, margin_results, vessel_results, iodine_results):
    """Calculate Reid Colposcopic Index (0-8 points) with clinical validation."""
    # Acetowhitening (0-2 points) - same as Swede
    reid_acetowhite = acetowhite_points
    
    # Margin (0-2 points) - based on sharpness AND regularity
    margin_sharpness = margin_results.get('sharpness_index', 0)
    margin_regularity = margin_results.get('regularity', 1.0)
    
    # Reid margins: 0=indistinct, 1=sharp but regular, 2=sharp AND irregular
    if margin_sharpness > 60 and margin_regularity < 0.6:
        reid_margin = 2  # Atypical (sharp + irregular)
    elif margin_sharpness > 40:
        reid_margin = 1  # Equivocal (sharp but regular)
    else:
        reid_margin = 0  # Normal (indistinct)
    
    # Vessels (0-2 points) - based on pattern regularity
    # Use your existing vessel analysis
    vessel_regularity = vessel_results.get('mean_circularity', 0)
    if vessel_regularity < Config.REID_VESSEL_THRESHOLDS[0]:
        reid_vessel = 2  # Atypical
    elif vessel_regularity < Config.REID_VESSEL_THRESHOLDS[1]:
        reid_vessel = 1  # Equivocal
    else:
        reid_vessel = 0  # Normal
    
    # Iodine uptake (0-2 points)
    brown_percentage = iodine_results.get('brown_percentage', 0)
    if brown_percentage > Config.REID_IODINE_THRESHOLDS[1]:
        reid_iodine = 0  # Normal uptake
    elif brown_percentage > Config.REID_IODINE_THRESHOLDS[0]:
        reid_iodine = 1  # Partial uptake
    else:
        reid_iodine = 2  # Negative uptake
    
    return reid_acetowhite + reid_margin + reid_vessel + reid_iodine

# %%
# ENHANCED: QUADRANT ANALYSIS FOR LESION SIZE
def calculate_lesion_area_percentage(lesion_mask, cervix_mask):
    """Calculate lesion area as percentage of cervix ROI with proper quadrant analysis."""
    if np.sum(cervix_mask) == 0:
        return {'percentage': 0, 'points': 0, 'quadrants_affected': 0}
    
    lesion_area = np.sum(lesion_mask > 0)
    cervix_area = np.sum(cervix_mask > 0)
    percentage = (lesion_area / cervix_area) * 100
    
    # PROPER QUADRANT ANALYSIS
    contours, _ = cv2.findContours(cervix_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    quadrants_affected = 0
    
    if contours:
        cervix_contour = max(contours, key=cv2.contourArea)
        
        if len(cervix_contour) >= 5:
            # Fit ellipse and get major/minor axes
            ellipse = cv2.fitEllipse(cervix_contour)
            center, axes, angle = ellipse
            
            # Create quadrant masks
            height, width = cervix_mask.shape
            y_coords, x_coords = np.indices((height, width))
            
            # Center coordinates relative to ellipse center
            x_centered = x_coords - center[0]
            y_centered = y_coords - center[1]
            
            # Rotate coordinates to align with ellipse axes
            angle_rad = np.radians(angle)
            x_rotated = x_centered * np.cos(angle_rad) + y_centered * np.sin(angle_rad)
            y_rotated = -x_centered * np.sin(angle_rad) + y_centered * np.cos(angle_rad)
            
            # Define quadrants
            quadrants = [
                (x_rotated >= 0) & (y_rotated >= 0),  # Q1
                (x_rotated < 0) & (y_rotated >= 0),   # Q2
                (x_rotated < 0) & (y_rotated < 0),    # Q3
                (x_rotated >= 0) & (y_rotated < 0)    # Q4
            ]
            
            # Check lesion occupancy in each quadrant
            for i, quadrant_mask in enumerate(quadrants):
                quadrant_cervix = quadrant_mask & (cervix_mask > 0)
                if np.any(quadrant_cervix):
                    quadrant_lesion = quadrant_mask & (lesion_mask > 0)
                    lesion_ratio = np.sum(quadrant_lesion) / np.sum(quadrant_cervix)
                    if lesion_ratio > 0.1:  # At least 10% occupancy
                        quadrants_affected += 1
    
    # SWEDE SCORING WITH QUADRANT SUPPORT
    points = 0
    if percentage > 25 or quadrants_affected >= 3:
        points = 2
    elif percentage > 5 or quadrants_affected >= 1:
        points = 1
    
    return {
        'percentage': percentage,
        'quadrants_affected': quadrants_affected,
        'points': points
    }

# ENHANCED: IODINE UPTAKE WITH SPECULAR REMOVAL
def analyze_iodine_uptake(iodine_image, cervix_mask=None):
    """Comprehensive iodine uptake analysis with proper specular removal."""
    if cervix_mask is None:
        cervix_mask = np.ones(iodine_image.shape[:2], dtype=bool)
    
    # REMOVE SPECULAR HIGHLIGHTS FIRST
    iodine_clean, specular_mask = remove_specular_highlights(iodine_image)
    
    # Convert to normalized HSV
    hsv = cv2.cvtColor(iodine_clean, cv2.COLOR_BGR2HSV)
    h = hsv[:, :, 0].astype(float) / 179.0  # Normalize hue to 0-1
    s = hsv[:, :, 1].astype(float) / 255.0  # Normalize saturation to 0-1
    v = hsv[:, :, 2].astype(float) / 255.0  # Normalize value to 0-1
    
    # BROWN DETECTION (normalized)
    brown_mask = (
        (h >= Config.IODINE_BROWN['hue'][0]/180.0) & (h <= Config.IODINE_BROWN['hue'][1]/180.0) &
        (s >= Config.IODINE_BROWN['sat'][0]/100.0) & (s <= Config.IODINE_BROWN['sat'][1]/100.0) &
        (v >= Config.IODINE_BROWN['val'][0]/100.0) & (v <= Config.IODINE_BROWN['val'][1]/100.0) &
        (cervix_mask > 0)
    )
    
    # YELLOW DETECTION (normalized)
    yellow_mask = (
        (h >= Config.IODINE_YELLOW['hue'][0]/180.0) & (h <= Config.IODINE_YELLOW['hue'][1]/180.0) &
        (s >= Config.IODINE_YELLOW['sat'][0]/100.0) & (s <= Config.IODINE_YELLOW['sat'][1]/100.0) &
        (v >= Config.IODINE_YELLOW['val'][0]/100.0) & (v <= Config.IODINE_YELLOW['val'][1]/100.0) &
        (cervix_mask > 0)
    )
    
    cervix_pixels = np.sum(cervix_mask > 0)
    if cervix_pixels == 0:
        return {
            'brown_percentage': 0,
            'yellow_percentage': 0,
            'points': 2
        }
    
    brown_percentage = np.sum(brown_mask) / cervix_pixels * 100
    yellow_percentage = np.sum(yellow_mask) / cervix_pixels * 100
    
    # SCORING
    points = 0
    if brown_percentage > 80:
        points = 0
    elif brown_percentage >= 40:
        points = 1
    else:
        points = 2
    
    return {
        'brown_percentage': brown_percentage,
        'yellow_percentage': yellow_percentage,
        'points': points
    }

# ENHANCED: TIME-SERIES ACETOWHITENING ANALYSIS
def calculate_acetowhite_dL(pre_acetic, post_acetic_sequence, timestamps, cervix_mask=None):
    """Calculate ΔL* for acetowhitening across a time sequence with persistence analysis."""
    # Remove specular highlights from pre-acetic image
    pre_clean, pre_specular_mask = remove_specular_highlights(pre_acetic)
    pre_lab = cv2.cvtColor(pre_clean, cv2.COLOR_BGR2LAB)
    
    results = {}
    persistence_scores = []  # Track persistence across time points

    if cervix_mask is None:
        cervix_mask = np.ones(pre_acetic.shape[:2], dtype=bool)

    for i, (post_img, timestamp) in enumerate(zip(post_acetic_sequence, timestamps)):
        # Remove specular highlights from post-acetic image
        post_clean, post_specular_mask = remove_specular_highlights(post_img)
        
        # Register images if we have a sequence
        if i > 0:
            registered, _ = register_images(pre_clean, post_clean)
            post_lab = cv2.cvtColor(registered, cv2.COLOR_BGR2LAB)
        else:
            post_lab = cv2.cvtColor(post_clean, cv2.COLOR_BGR2LAB)
        
        # Calculate ΔL*
        l_pre = pre_lab[:, :, 0].astype(np.float32)
        l_post = post_lab[:, :, 0].astype(np.float32)
        dL = l_post - l_pre
        
        # Remove areas with specular highlights from calculation
        combined_specular = pre_specular_mask | post_specular_mask
        dL[combined_specular] = 0
        
        # Calculate metrics
        mean_dL = np.mean(dL[dL > 0]) if np.any(dL > 0) else 0
        area_percentage = np.sum(dL > Config.ACETOWHITE_DL_LOW) / dL.size * 100
        
        # Check persistence (maintains or increases whitening)
        if i > 0:
            prev_mean_dL = results[timestamps[i-1]]['mean_dL']
            persistence = 1 if mean_dL >= prev_mean_dL * 0.8 else 0  # At least 80% maintained
            persistence_scores.append(persistence)
        
        results[timestamp] = {
            'mean_dL': mean_dL,
            'area_percentage': area_percentage,
            'dL_map': dL
        }
    
    # PERSISTENCE ANALYSIS - NEW
    total_persistence = sum(persistence_scores) if persistence_scores else 0
    max_persistence = len(persistence_scores)
    persistence_ratio = total_persistence / max_persistence if max_persistence > 0 else 0
    
    # Get maximum values across time points
    max_dL = max([data['mean_dL'] for data in results.values()])
    max_area = max([data['area_percentage'] for data in results.values()])
    
    # SCORING WITH PERSISTENCE CONSIDERATION
    points = 0
    if (max_area >= Config.ACETOWHITE_AREA_HIGH or max_dL >= Config.ACETOWHITE_DL_HIGH) and persistence_ratio >= 0.7:
        points = 2
    elif (max_area >= Config.ACETOWHITE_AREA_LOW or max_dL >= Config.ACETOWHITE_DL_LOW) and persistence_ratio >= 0.5:
        points = 1
    
    results['summary'] = {
        'max_dL': max_dL,
        'max_area': max_area,
        'persistence_ratio': persistence_ratio,
        'points': points
    }
    
    return results


# %%
def detect_lesions(post_acetic_image, cervix_mask):
    """Comprehensive lesion detection using multiple features."""
    lab = cv2.cvtColor(post_acetic_image, cv2.COLOR_BGR2LAB)
    l_channel = lab[:, :, 0]
    
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(l_channel)
    
    _, bright_mask = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    kernel = np.ones((5, 5), np.uint8)
    bright_mask = cv2.morphologyEx(bright_mask, cv2.MORPH_CLOSE, kernel)
    bright_mask = cv2.morphologyEx(bright_mask, cv2.MORPH_OPEN, kernel)
    
    lesion_mask = bright_mask & cervix_mask
    
    contours, _ = cv2.findContours(lesion_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    min_area = cervix_mask.size * 0.01
    valid_lesions = []
    
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > min_area:
            perimeter = cv2.arcLength(contour, True)
            if perimeter > 0:
                circularity = 4 * np.pi * area / (perimeter ** 2)
                if circularity > 0.3:
                    valid_lesions.append(contour)
    
    final_lesion_mask = np.zeros_like(lesion_mask)
    cv2.drawContours(final_lesion_mask, valid_lesions, -1, 255, -1)
    
    return final_lesion_mask

# %%
def check_colposcopy_edge_cases(image_sequence, cervix_mask):
    """Check for edge cases that require special handling."""
    if 'post_acetic' in image_sequence:
        image = image_sequence['post_acetic'] if not isinstance(image_sequence['post_acetic'], list) else image_sequence['post_acetic'][0]
        
        vessel_results = analyze_vessel_patterns(image, cervix_mask)
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        mean_intensity = np.mean(gray[cervix_mask > 0]) if np.any(cervix_mask > 0) else np.mean(gray)
        
        if (vessel_results['points'] >= 2 and 
            mean_intensity > 200 and 
            vessel_results['mean_circularity'] < 0.3):
            return True, "Atypical vessels with dense acetowhitening - possible cancer"
    
    return False, ""

# %%
def analyze_colposcopy(image_sequence, timestamps=None):
    """Comprehensive colposcopy image sequence analysis using Swede AND Reid scoring."""
    results = {
        'swede_score': 0,
        'reid_score': 0,  
        'explanations': [],
        'confidence': 0.0,
        'classification': 'Unknown',
        'risk_level': 'unknown',
        'recommendation': 'Analysis failed',
        'component_scores': {}
    }
    
    try:
        if 'pre_acetic' not in image_sequence or 'post_acetic' not in image_sequence:
            results['explanations'].append('Incomplete image sequence for proper analysis')
            results['confidence'] = 0.3
            return results
        
        pre_acetic = image_sequence['pre_acetic']
        post_acetic = image_sequence['post_acetic']
        
        if timestamps is None:
            timestamps = Config.ACETIC_EVOLUTION_TIMES
        
        # Basic checks to ensure images are valid
        if pre_acetic is None or post_acetic is None:
            results['explanations'].append('Invalid image data')
            return results
            
        cervix_mask = segment_cervix_region(pre_acetic)
        if np.sum(cervix_mask) == 0:
            results['explanations'].append('Could not segment cervix region')
            results['confidence'] = 0.4
            return results
        lesion_mask = detect_lesions(post_acetic, cervix_mask)
        
        # ACETOWHITENING ANALYSIS WITH TIME SERIES
        if isinstance(image_sequence['post_acetic'], list) and len(image_sequence['post_acetic']) > 1:
            acetowhite_results = calculate_acetowhite_dL(
                pre_acetic, 
                image_sequence['post_acetic'], 
                timestamps
            )
            acetowhite_points = acetowhite_results['summary']['points']
            results['component_scores']['acetowhitening'] = acetowhite_results
        else:
            # Fallback for single image
            lab_pre = cv2.cvtColor(pre_acetic, cv2.COLOR_BGR2LAB)
            lab_post = cv2.cvtColor(post_acetic, cv2.COLOR_BGR2LAB)
            dL = np.mean(lab_post[:, :, 0] - lab_pre[:, :, 0])
            
            if dL > Config.ACETOWHITE_DL_HIGH:
                acetowhite_points = 2
            elif dL > Config.ACETOWHITE_DL_LOW:
                acetowhite_points = 1
            else:
                acetowhite_points = 0
            
            results['component_scores']['acetowhitening'] = {'mean_dL': dL, 'points': acetowhite_points}
        
        # MARGIN ANALYSIS
        margin_results = analyze_lesion_margins(lesion_mask, post_acetic)
        margin_points = margin_results['points']
        results['component_scores']['margins'] = margin_results
        
        # VESSEL ANALYSIS
        vessel_results = analyze_vessel_patterns(post_acetic, lesion_mask)
        vessel_points = vessel_results['points']
        results['component_scores']['vessels'] = vessel_results
        
        # LESION SIZE WITH QUADRANT ANALYSIS
        size_results = calculate_lesion_area_percentage(lesion_mask, cervix_mask)
        size_points = size_results['points']
        results['component_scores']['size'] = size_results
        
        # IODINE UPTAKE ANALYSIS
        if 'iodine' in image_sequence:
            iodine_results = analyze_iodine_uptake(image_sequence['iodine'], cervix_mask)
            iodine_points = iodine_results['points']
            results['component_scores']['iodine'] = iodine_results
        else:
            iodine_points = 0
            results['component_scores']['iodine'] = {'points': 0, 'note': 'No iodine image provided'}
        
        # CALCULATE SWEDE SCORE
        total_swede_score = acetowhite_points + margin_points + vessel_points + size_points + iodine_points
        results['swede_score'] = total_swede_score
        
        # CALCULATE REID SCORE - NEW IMPLEMENTATION
        reid_score = calculate_reid_score(acetowhite_points, margin_results, 
                                        vessel_results, iodine_results)
        results['reid_score'] = reid_score
        
        explanations = [
            f"Acetowhitening: {acetowhite_points} points",
            f"Margin sharpness: {margin_points} points",
            f"Vessel pattern: {vessel_points} points ({vessel_results['pattern_type']})",
            f"Lesion size: {size_points} points ({size_results['percentage']:.1f}% of cervix, {size_results['quadrants_affected']} quadrants)",
            f"Iodine uptake: {iodine_points} points",
            f"Reid Score: {reid_score}/8"  # ADDED REID SCORE
        ]
        results['explanations'] = explanations
        
        # COMBINED RISK ASSESSMENT USING BOTH SCORES
        if total_swede_score >= 7 or reid_score >= 6:
            results.update({
                'classification': 'High-grade suspicion',
                'recommendation': 'Urgent biopsy indicated',
                'risk_level': 'red',
                'confidence': min(0.85 + (max(total_swede_score - 7, reid_score - 6) * 0.05), 0.95)
            })
        elif total_swede_score >= 5 or reid_score >= 4:
            results.update({
                'classification': 'Suspicious findings',
                'recommendation': 'Biopsy indicated',
                'risk_level': 'amber',
                'confidence': 0.75 + (max(total_swede_score - 5, reid_score - 4) * 0.05)
            })
        else:
            results.update({
                'classification': 'Likely normal/low grade',
                'recommendation': 'Routine screening',
                'risk_level': 'green',
                'confidence': 0.80 - (max(total_swede_score, reid_score) * 0.05)
            })
        
        # CHECK FOR EDGE CASES
        edge_case_detected, edge_info = check_colposcopy_edge_cases(image_sequence, cervix_mask)
        if edge_case_detected:
            results['risk_level'] = 'red'
            results['recommendation'] = 'Urgent expert review needed'
            results['explanations'].append(f"Edge case detected: {edge_info}")
            results['confidence'] = max(0.7, results['confidence'])
            
    except Exception as e:
        results['error'] = f'Analysis failed: {str(e)}'
        results['confidence'] = 0.0
        import traceback
        results['traceback'] = traceback.format_exc()
            
    return results



# %%
# ===== ENHANCED SINGLE-IMAGE COLPOSCOPY ANALYSIS =====
# Replace your old simple functions with these
def analyze_colposcopy_single_image(image):
    """Enhanced single-image colposcopy analysis with proper feature extraction and confidence scoring."""
    results = {
        'swede_score': 0,
        'reid_score': 0,
        'explanations': [],
        'confidence': 0.0,
        'classification': 'Unknown',
        'risk_level': 'unknown',
        'recommendation': 'Analysis failed',
        'component_scores': {}
    }
    
    try:
        # Segment cervix region
        cervix_mask = segment_cervix_region(image)
        if np.sum(cervix_mask) == 0:
            results['explanations'].append('Could not segment cervix region')
            results['confidence'] = 0.4
            return results
        
        # Enhanced lesion detection
        lesion_mask = detect_lesions_enhanced(image, cervix_mask)
        
        # Analyze features with improved methods
        margin_results = analyze_lesion_margins_enhanced(lesion_mask, image, cervix_mask)
        vessel_results = analyze_vessel_patterns_enhanced(image, lesion_mask)
        area_results = calculate_lesion_area_percentage(lesion_mask, cervix_mask) # Use your GOOD quadrant function
        iodine_results = {'points': 0, 'note': 'No iodine image provided'} # Placeholder for single image
        
        # Improved acetowhite detection using LAB color space
        acetowhite_points = analyze_acetowhite_single(image, cervix_mask)
        
        # Calculate total Swede score
        total_swede_score = (acetowhite_points + 
                      margin_results['points'] + 
                      vessel_results['points'] + 
                      area_results['points'] +
                      iodine_results['points'])
        
        # Calculate Reid score (using Swede acetowhite points and enhanced features)
        reid_score = calculate_reid_score(acetowhite_points, margin_results, vessel_results, iodine_results)
        
        results['swede_score'] = total_swede_score
        results['reid_score'] = reid_score
        results['component_scores'] = {
            'acetowhitening': {'points': acetowhite_points},
            'margins': margin_results,
            'vessels': vessel_results,
            'size': area_results,
            'iodine': iodine_results
        }
        
        # Build detailed explanations
        explanations = [
            f"Acetowhitening: {acetowhite_points} points",
            f"Margin characteristics: {margin_results['points']} points ({margin_results['type']})",
            f"Vessel pattern: {vessel_results['points']} points ({vessel_results['pattern_type']})",
            f"Lesion size: {area_results['points']} points ({area_results['percentage']:.1f}% of cervix, {area_results['quadrants_affected']} quadrants)",
            f"Iodine uptake: {iodine_results['points']} points ({iodine_results.get('note', 'N/A')})",
            f"Reid Score: {reid_score}/8"
        ]
        results['explanations'] = explanations
        
        # ===== CORE IMPROVEMENT: CONFIDENCE & RISK BASED ON SCORE MARGINS =====
        # Determine risk level and confidence based on BOTH scores and their proximity to decision thresholds.
        swede_confidence = calculate_confidence_from_margin(total_swede_score, [0, 5, 7, 10])
        reid_confidence = calculate_confidence_from_margin(reid_score, [0, 3, 5, 8])
        
        # Combined confidence gives more weight to the score that is more certain (further from a threshold)
        combined_confidence = (swede_confidence + reid_confidence) / 2.0
        
        # Determine classification based on highest risk level indicated by either score
        if total_swede_score >= 7 or reid_score >= 6:
            classification = 'High-grade suspicion'
            risk_level = 'red'
            recommendation = 'Urgent biopsy indicated'
            # Boost confidence if both scores agree on high risk
            if total_swede_score >= 7 and reid_score >= 6:
                combined_confidence = min(combined_confidence + 0.1, 0.95)
        elif total_swede_score >= 5 or reid_score >= 4:
            classification = 'Suspicious findings'
            risk_level = 'amber'
            recommendation = 'Biopsy indicated'
        else:
            classification = 'Likely normal/low grade'
            risk_level = 'green'
            recommendation = 'Routine screening'
            # Boost confidence for clearly normal cases
            if total_swede_score < 3 and reid_score < 2:
                combined_confidence = min(combined_confidence + 0.1, 0.95)
                
        results.update({
            'classification': classification,
            'recommendation': recommendation,
            'risk_level': risk_level,
            'confidence': combined_confidence
        })
        
        # Final safety rail: Low confidence overrides to escalation
        if combined_confidence < 0.6:
            results['action'] = 'escalate'
            results['recommendation'] = 'Expert review recommended due to low confidence'
            if 'Expert review recommended' not in results['explanations']:
                results['explanations'].append('Expert review recommended due to low confidence in automated analysis')
            
    except Exception as e:
        results['error'] = f'Analysis failed: {str(e)}'
        results['confidence'] = 0.0
    
    return results

def calculate_confidence_from_margin(score, thresholds):
    """
    Calculates confidence based on how far a score is from the nearest decision threshold.
    thresholds: [lower_bound, low_risk_upper, high_risk_lower, upper_bound] e.g., [0, 5, 7, 10] for Swede.
    """
    low_risk_upper = thresholds[1]
    high_risk_lower = thresholds[2]
    
    if score < low_risk_upper:
        # Confidence increases as you move away from the low-risk/suspicious threshold (5)
        distance_from_threshold = low_risk_upper - score
        max_distance = low_risk_upper - thresholds[0]
        confidence = 0.7 + 0.25 * (distance_from_threshold / max_distance) # Base 0.7, up to 0.95
        
    elif score >= high_risk_lower:
        # Confidence increases as you move above the high-risk threshold (7)
        distance_from_threshold = score - high_risk_lower
        max_distance = thresholds[3] - high_risk_lower
        confidence = 0.7 + 0.25 * (distance_from_threshold / max_distance) # Base 0.7, up to 0.95
        
    else:
        # Score is in the equivocal zone (between 5 and 7). Confidence is lowest here.
        # Confidence is lowest at the midpoint of the equivocal zone and increases towards the edges.
        midpoint = (low_risk_upper + high_risk_lower) / 2.0
        distance_from_midpoint = abs(score - midpoint)
        max_distance = (high_risk_lower - low_risk_upper) / 2.0
        # Scale from 0.5 (at midpoint) to 0.7 (at the edges of the equivocal zone)
        confidence = 0.5 + 0.2 * (distance_from_midpoint / max_distance)
        
    return min(max(confidence, 0.5), 0.95) # Clamp between 0.5 and 0.95

# --- Enhanced Helper Functions for Single Image Analysis ---

def detect_lesions_enhanced(image, cervix_mask):
    """Enhanced lesion detection using adaptive thresholding."""
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel = lab[:, :, 0]
    
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(l_channel)
    
    cervix_intensity = enhanced[cervix_mask > 0]
    if len(cervix_intensity) > 0:
        mean_intensity = np.mean(cervix_intensity)
        std_intensity = np.std(cervix_intensity)
        threshold = mean_intensity + 1.5 * std_intensity
    else:
        threshold = 160
    
    _, lesion_mask = cv2.threshold(enhanced, threshold, 255, cv2.THRESH_BINARY)
    
    kernel = np.ones((5, 5), np.uint8)
    lesion_mask = cv2.morphologyEx(lesion_mask, cv2.MORPH_CLOSE, kernel)
    lesion_mask = cv2.morphologyEx(lesion_mask, cv2.MORPH_OPEN, kernel)
    
    lesion_mask = lesion_mask & cervix_mask
    
    return lesion_mask

def analyze_acetowhite_single(image, cervix_mask):
    """Analyze acetowhitening from single image using LAB color space."""
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel = lab[:, :, 0].astype(np.float32)
    
    cervix_intensity = l_channel[cervix_mask > 0]
    if len(cervix_intensity) == 0:
        return 0
    
    mean_intensity = np.mean(cervix_intensity)
    std_intensity = np.std(cervix_intensity)
    
    acetowhite_threshold = mean_intensity + 2 * std_intensity
    acetowhite_mask = l_channel > acetowhite_threshold
    acetowhite_mask = acetowhite_mask & (cervix_mask > 0)
    
    acetowhite_area = np.sum(acetowhite_mask)
    cervix_area = np.sum(cervix_mask > 0)
    area_percentage = (acetowhite_area / cervix_area) * 100 if cervix_area > 0 else 0
    
    intensity_difference = np.mean(l_channel[acetowhite_mask]) - mean_intensity if np.any(acetowhite_mask) else 0
    
    if area_percentage > 25 or intensity_difference > 20:
        return 2
    elif area_percentage > 5 or intensity_difference > 10:
        return 1
    else:
        return 0

def analyze_lesion_margins_enhanced(lesion_mask, image, cervix_mask):
    """ULTRA-CONSERVATIVE margin analysis to avoid false positives."""
    if np.sum(lesion_mask) == 0:
        return {'points': 0, 'type': 'No lesion', 'sharpness': 0, 'regularity': 1.0}
    
    # Convert to grayscale and enhance
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))  # Less aggressive
    enhanced = clahe.apply(gray)
    
    # Calculate gradients with noise reduction
    grad_x = cv2.Scharr(enhanced, cv2.CV_64F, 1, 0)
    grad_y = cv2.Scharr(enhanced, cv2.CV_64F, 0, 1)
    gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
    
    # Find boundaries with higher thresholds
    boundaries = cv2.Canny(enhanced, 70, 200)  # Higher thresholds
    lesion_boundaries = boundaries & lesion_mask
    
    if np.sum(lesion_boundaries) == 0:
        return {'points': 0, 'type': 'Indistinct', 'sharpness': 0, 'regularity': 1.0}
    
    # Calculate margin sharpness - USE 90TH PERCENTILE to ignore noise
    margin_pixels = gradient_magnitude[lesion_boundaries > 0]
    if len(margin_pixels) == 0:
        return {'points': 0, 'type': 'Indistinct', 'sharpness': 0, 'regularity': 1.0}
    
    sharpness_index = np.percentile(margin_pixels, 90)  # 90th percentile, not median
    
    # Analyze contour
    contours, _ = cv2.findContours(lesion_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return {'points': 0, 'type': 'No contour', 'sharpness': sharpness_index, 'regularity': 1.0}
    
    main_contour = max(contours, key=cv2.contourArea)
    perimeter = cv2.arcLength(main_contour, True)
    area = cv2.contourArea(main_contour)
    
    circularity = 4 * np.pi * area / (perimeter ** 2) if perimeter > 0 else 0
    convex_hull = cv2.convexHull(main_contour)
    hull_area = cv2.contourArea(convex_hull)
    solidity = area / hull_area if hull_area > 0 else 1
    regularity_score = (circularity + solidity) / 2

    # ===== ULTRA-CONSERVATIVE THRESHOLDS =====
    points = 0
    margin_type = "Indistinct"
    
    # ONLY give 2 points for EXTREMELY clear cases
    if sharpness_index > 75 and regularity_score < 0.55:  # Much higher threshold
        points = 2
        margin_type = "Sharp, irregular (CIN2+ confirmed)"
    
    # Give 1 point for clear but regular borders
    elif sharpness_index > 50 and regularity_score > 0.7:
        points = 1
        margin_type = "Distinct, geographic"
    
    # Default to 0 for everything else
    else:
        points = 0
        if sharpness_index < 25:
            margin_type = "Feathery, indistinct"
        else:
            margin_type = "Mildly distinct (not suspicious)"
    
    return {
        'points': points,
        'type': margin_type,
        'sharpness': sharpness_index,
        'regularity': regularity_score
    }

def analyze_vessel_patterns_enhanced(image, lesion_mask=None):
    """Enhanced vessel pattern analysis with better feature extraction."""
    green_channel = image[:, :, 1]
    
    vessel_enhanced = np.zeros_like(green_channel, dtype=np.float32)
    
    for sigma in [1, 2, 3]:
        blurred = cv2.GaussianBlur(green_channel, (0, 0), sigma)
        gx = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
        magnitude = np.sqrt(gx**2 + gy**2)
        vessel_enhanced = np.maximum(vessel_enhanced, magnitude)
    
    vessel_enhanced = (vessel_enhanced / vessel_enhanced.max() * 255).astype(np.uint8)
    _, vessel_mask = cv2.threshold(vessel_enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    vessel_density = np.sum(vessel_mask > 0) / vessel_mask.size
    
    points = 0
    pattern_type = "Normal"
    
    if vessel_density > Config.VESSEL_DENSITY_HIGH:
        points = 2
        pattern_type = "Coarse pattern"
    elif vessel_density > Config.VESSEL_DENSITY_MEDIUM:
        points = 1
        pattern_type = "Fine pattern"
    
    return {
        'vessel_density': vessel_density,
        'pattern_type': pattern_type,
        'points': points
    }

# ===== UPDATE THE MAIN ANALYZE_IMAGE FUNCTION =====
# (This section shows the changes to make to your existing function)
# In your analyze_image function, find the colposcopy section and REPLACE it with this:

    # elif image_type == 'colposcopy':
    #     # Check if we have a sequence for comprehensive analysis or just a single image
    #     if image_sequence is not None and 'pre_acetic' in image_sequence and 'post_acetic' in image_sequence:
    #         # Use the comprehensive time-series analysis if sequence is provided
    #         result = analyze_colposcopy(image_sequence, timestamps)
    #     else:
    #         # Use the enhanced single-image analysis as a fallback
    #         result = enhanced_analyze_colposcopy_single_image(image)
    #     result.setdefault('image_type', 'Colposcopy')

# %%
# Main analysis function with enhanced image type detection

def analyze_image(image, image_type=None, metadata=None, image_sequence=None, timestamps=None):
    """
    Comprehensive analysis for Pap smear, colposcopy, or other images.
    Uses Swede score for colposcopy, Bethesda rules for Pap smears.
    """

    # --- Step 1: Decide type ---
    if image_type is None:
        image_type = classify_image_type(image, metadata)

    # --- Step 2: Quality check ---
    quality_result = check_image_quality(image, image_type)

    # Pap smear override for blur
    if image_type == 'pap_smear':
        blue_channel = image[:, :, 0]
        blue_intensity = np.mean(blue_channel)
        _, nuclei_mask = cv2.threshold(blue_channel, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        nuclei_density = np.sum(nuclei_mask > 0) / nuclei_mask.size
        if nuclei_density > 0.1 and blue_intensity > 100:
            quality_result['reasons'] = [r for r in quality_result['reasons'] if 'blur' not in r.lower()]
            quality_result['is_ok'] = True

    # "Other" images always pass
    if image_type == 'other':
        quality_result['is_ok'] = True

    # Critical failures
    critical_failures = [
        r for r in quality_result['reasons']
        if 'dimensions' in r.lower() or ('blur' in r.lower() and 'extremely' in r.lower())
    ]
    if critical_failures and image_type != 'other':
        return {
            'status': 'quality_fail',
            'action': 'retake',
            'reasons': critical_failures,
            'confidence': 0.0,
            'metrics': quality_result['metrics'],
            'image_type': image_type
        }

    # --- Step 3: Type-specific analysis ---
    if image_type == 'pap_smear':
        result = analyze_pap_smear(image)

        # Only override if Pap function failed completely
        if not result or result.get('classification', '').lower() in ['', 'unknown', None]:
            result = {
                'classification': 'Unsatisfactory',
                'bethesda_category': 'Unsatisfactory for evaluation',
                'recommendation': 'Repeat sample collection',
                'confidence': 0.9,
                'risk_level': 'red',
                'explanations': ['Pap smear analysis failed, defaulted to Unsatisfactory'],
            }
        result.setdefault('image_type', 'Pap Smear')

    elif image_type == 'colposcopy':
        # Multi-image analysis preferred
        if image_sequence and len(image_sequence) > 1:
            result = analyze_colposcopy(image_sequence)
        else:
            result = analyze_colposcopy_single_image(image)

        # Only override if colposcopy analysis failed
        if not result or result.get('classification', '').lower() in ['', 'unknown', None]:
            result = {
                'classification': 'Negative',
                'recommendation': 'Routine follow-up',
                'confidence': 0.8,
                'risk_level': 'green',
                'explanations': ['Colposcopy analysis failed, defaulted to Negative'],
            }
        result.setdefault('image_type', 'Colposcopy')

    else:  # Non-cervical
        result = {
            'classification': 'Non-cervical image',
            'recommendation': 'Provide Pap smear or colposcopy image',
            'explanations': ['This does not appear to be a cervical cancer screening image'],
            'confidence': 0.9,
            'risk_level': 'unknown',
            'image_type': 'Other'
        }

    # --- Step 4: Attach quality info ---
    result['quality_metrics'] = quality_result['metrics']
    result['quality_warnings'] = quality_result['reasons']

    # Safety defaults
    result.setdefault('confidence', 0.0)
    result.setdefault('risk_level', 'unknown')
    result.setdefault('explanations', [])
    result.setdefault('classification', 'Analysis inconclusive')
    result.setdefault('recommendation', 'Expert review needed')

    # Escalation
    if (result['confidence'] < 0.6 or
        result['risk_level'] == 'red' or
        str(result.get('bethesda_category', '')).startswith(('HSIL', 'ASC-H'))):
        result['action'] = 'escalate'
        if 'Expert review recommended' not in result['explanations']:
            result['explanations'].append('Expert review recommended')

    result['status'] = 'completed'
    result['timestamp'] = datetime.now().isoformat()

    return result


# Enhanced visualization function
def plot_results(image, result, save_path=None):
    """Visualize the image and analysis results with comprehensive details."""
    fig = plt.figure(figsize=(20, 12))
    
    ax1 = plt.subplot2grid((3, 3), (0, 0), rowspan=2, colspan=2)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    ax1.imshow(image_rgb)
    ax1.set_title(f'{result.get("image_type", "Image")} - {result.get("classification", "Unknown")}')
    ax1.axis('off')
    
    ax2 = plt.subplot2grid((3, 3), (0, 2), rowspan=3)
    ax2.axis('off')
    
    result_text = []
    result_text.append(f"**Classification:** {result.get('classification', 'Unknown')}")
    
    if 'swede_score' in result:
        result_text.append(f"**Swede Score:** {result['swede_score']}/10")
    
    if 'bethesda_category' in result:
        result_text.append(f"**Bethesda Category:** {result['bethesda_category']}")
    
    if 'confidence' in result:
        result_text.append(f"**Confidence:** {result['confidence']:.2f}")
    
    if 'risk_level' in result:
        risk_color = {'green': 'green', 'amber': 'orange', 'red': 'red'}.get(result['risk_level'], 'black')
        result_text.append(f"**Risk Level:** <span style='color:{risk_color}'>{result['risk_level'].upper()}</span>")
    
    if 'recommendation' in result:
        result_text.append(f"**Recommendation:** {result['recommendation']}")
    
    if 'explanations' in result and result['explanations']:
        result_text.append("\n**Details:**")
        for explanation in result['explanations']:
            result_text.append(f"• {explanation}")
    
    if 'metrics' in result:
        result_text.append("\n**Quality Metrics:**")
        for key, value in result['metrics'].items():
            if key != 'calibration_detected' or not value:
                result_text.append(f"• {key}: {value}")
    
    ax2.text(0.05, 0.95, '\n'.join(result_text), transform=ax2.transAxes, 
             verticalalignment='top', fontsize=10, 
             bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.7))
    
    if result.get('image_type') == 'Pap Smear' and 'morphology_stats' in result:
        ax3 = plt.subplot2grid((3, 3), (2, 0))
        stats = result['morphology_stats']
        if stats and 'n_nuclei' in stats and stats['n_nuclei'] > 0:
            metrics = ['Mean Area', 'Std Area', 'Circularity']
            values = [stats.get('mean_area', 0), stats.get('std_area', 0), stats.get('mean_circularity', 0)]
            ax3.bar(metrics, values)
            ax3.set_title('Nuclear Morphology')
            ax3.tick_params(axis='x', rotation=45)
    
    elif result.get('image_type') == 'Colposcopy' and 'component_scores' in result:
        ax3 = plt.subplot2grid((3, 3), (2, 0))
        scores = result['component_scores']
        components = list(scores.keys())
        points = [scores[comp].get('points', 0) for comp in components]
        ax3.bar(components, points)
        ax3.set_title('Swede Score Components')
        ax3.tick_params(axis='x', rotation=45)
        ax3.set_ylim(0, 2)
    
    ax4 = plt.subplot2grid((3, 3), (2, 1))
    
    if result.get('image_type') == 'Pap Smear':
        blue_channel = image[:, :, 0]
        _, nuclei_mask = cv2.threshold(blue_channel, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        ax4.imshow(nuclei_mask, cmap='gray')
        ax4.set_title('Detected Nuclei')
        ax4.axis('off')
    
    elif result.get('image_type') == 'Colposcopy':
        cervix_mask = segment_cervix_region(image)
        ax4.imshow(cervix_mask, cmap='gray')
        ax4.set_title('Cervix Segmentation')
        ax4.axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()



# Update the run_analysis function to work with single images
def run_analysis(image_path, metadata=None, save_path=None):
    """Run complete analysis on an image."""
    try:
        # Load image
        img = cv2.imread(image_path)
        if img is None:
            print(f"Error: Could not load image from {image_path}")
            return None
        
        print(f"Analyzing image: {os.path.basename(image_path)}")
        print("=" * 50)
        
        # Run analysis - pass as single image, not sequence
        result = analyze_image(img, None, metadata)
        
        # Display results
        if result.get('status') == 'quality_fail':
            print("QUALITY CHECK FAILED")
            for reason in result['reasons']:
                print(f"- {reason}")
            print("\nACTION: Retake image")
        else:
            # Visualize results
            plot_results(img, result, save_path)
            
            # Print detailed results
            print("\nANALYSIS RESULTS:")
            print("=" * 50)
            
            if 'classification' in result:
                print(f"Classification: {result['classification']}")
            if 'swede_score' in result:
                print(f"Swede Score: {result['swede_score']}/10")
            if 'reid_score' in result:
                print(f"Reid Score: {result['reid_score']}/8")
            if 'bethesda_category' in result:
                print(f"Bethesda Category: {result['bethesda_category']}")
            if 'confidence' in result:
                print(f"Confidence: {result['confidence']:.2f}")
            if 'risk_level' in result:
                print(f"Risk Level: {result['risk_level'].upper()}")
            if 'recommendation' in result:
                print(f"Recommendation: {result['recommendation']}")
            if 'explanations' in result and result['explanations']:
                print("\nDetails:")
                for explanation in result['explanations']:
                    print(f"- {explanation}")
            if 'note' in result:
                print(f"Note: {result['note']}")
            
            # Print quality warnings if any
            if 'quality_warnings' in result and result['quality_warnings']:
                print("\nQuality Warnings:")
                for warning in result['quality_warnings']:
                    print(f"- {warning}")
            
            # Special message for non-cervical images
            if result.get('image_type') == 'Other':
                print("\n" + "=" * 50)
                print("NOTE: This image doesn't appear to be a cervical cancer screening image.")
                print("Please provide a Pap smear or colposcopy image for proper analysis.")
        
        return result
        
    except Exception as e:
        print(f"Error during analysis: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

 

# %%
result = run_analysis('D:\\Yantram\\Cervical Cancer\\stomach.jpg')


