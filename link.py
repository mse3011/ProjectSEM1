from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import os
import logging

import numpy as np
import tensorflow as tf
import cv2

import SimpleITK as sitk
from radiomics import featureextractor

import joblib
import pandas as pd

app = Flask(__name__)
CORS(app)

# -------------------------------------------------------------------
# 1. Logging Configuration
# -------------------------------------------------------------------
logging.basicConfig(level=logging.DEBUG)
app.logger.setLevel(logging.DEBUG)

# -------------------------------------------------------------------
# 2. Segmentation Model (segment.h5)
# -------------------------------------------------------------------
def dice_coef(y_true, y_pred):
    smooth = 1e-15
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    y_true = tf.keras.layers.Flatten()(y_true)
    y_pred = tf.keras.layers.Flatten()(y_pred)
    intersection = tf.reduce_sum(y_true * y_pred)
    return (2.0 * intersection + smooth) / (
        tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) + smooth
    )

def dice_loss(y_true, y_pred):
    return 1.0 - dice_coef(y_true, y_pred)

SEGMENT_MODEL_PATH = 'segment.h5'
segmentation_model = tf.keras.models.load_model(
    SEGMENT_MODEL_PATH,
    custom_objects={'dice_loss': dice_loss, 'dice_coef': dice_coef}
)
app.logger.info(f"Segmentation model loaded from {SEGMENT_MODEL_PATH}")

# -------------------------------------------------------------------
# 3. Radiomics Classifier (scikit-learn)
# -------------------------------------------------------------------
CLASSIFIER_PATH = 'radiomics_classifier.pkl'
FEATURE_COLS_PATH = 'feature_columns.pkl'

if os.path.exists(CLASSIFIER_PATH) and os.path.exists(FEATURE_COLS_PATH):
    try:
        clf = joblib.load(CLASSIFIER_PATH)  # random forest or other scikit-learn model
        feature_cols = joblib.load(FEATURE_COLS_PATH)  # columns used in training
        app.logger.info(f"Scikit-learn classifier loaded from {CLASSIFIER_PATH}")
    except Exception as e:
        clf = None
        feature_cols = None
        app.logger.error(f"Failed to load scikit-learn classifier: {e}")
else:
    clf = None
    feature_cols = None
    app.logger.warning("No radiomics_classifier.pkl or feature_columns.pkl found. Classification will be skipped.")

# -------------------------------------------------------------------
# 4. Radiomics Feature Extraction
# -------------------------------------------------------------------
extractor = featureextractor.RadiomicsFeatureExtractor()
extractor.settings['resampledPixelSpacing'] = None
extractor.settings['correctMask'] = True

def extract_radiomics_features(image_path, mask_path):
    image_sitk = sitk.ReadImage(image_path)
    mask_sitk = sitk.ReadImage(mask_path)

    # Convert color image to grayscale if needed
    if image_sitk.GetNumberOfComponentsPerPixel() > 1:
        image_sitk = sitk.VectorIndexSelectionCast(image_sitk, 0)

    image_sitk = sitk.Cast(image_sitk, sitk.sitkFloat32)

    if mask_sitk.GetNumberOfComponentsPerPixel() > 1:
        mask_sitk = sitk.VectorIndexSelectionCast(mask_sitk, 0)
    mask_sitk = sitk.Cast(mask_sitk, sitk.sitkUInt8)

    # Execute pyradiomics
    feature_vector = extractor.execute(image_sitk, mask_sitk)

    # Filter out 'diagnostics_' keys
    features = {
        k: float(v) for k, v in feature_vector.items()
        if not k.startswith('diagnostics_')
    }
    app.logger.info(f"Extracted Radiomics Features: {features}")
    return features

# -------------------------------------------------------------------
# 5. Classification via Radiomics (scikit-learn)
# -------------------------------------------------------------------
def classify_radiomics(feature_dict):
    if clf is None or feature_cols is None:
        print("DEBUG: Radiomics classifier not loaded.")
        return "unknown"

    row = pd.DataFrame([feature_dict])
    row = row.reindex(columns=feature_cols, fill_value=0)
    pred_label = clf.predict(row)[0]
    return pred_label

# -------------------------------------------------------------------
# 6. Parameter-by-Parameter Interpretation
#    (Copied/Adapted from your "interpret_value" approach)
# -------------------------------------------------------------------
def interpret_value(value, parameter_name):
    """
    Takes a numeric feature value (or classification label) and returns
    a textual interpretation based on the logic from your script.
    """
    # If it's not numeric, e.g. classification label, handle separately
    if parameter_name == "Label":
        return f"Tumor classification label is '{value}'."  # Minimal interpretation

    # Convert to float if possible
    try:
        val = float(value)
    except:
        return f"No numeric value provided for {parameter_name}."

    # The same if-else logic from your script:
    if parameter_name == 'Hypo-dense region intensity (minimum)':
        if val < 20:
            return ("This low intensity value suggests the presence of necrotic or cystic tissue, "
                    "indicative of reduced metabolic activity.")
        elif 20 <= val <= 50:
            return ("A moderate intensity value suggests a mixture of healthy and necrotic tissue, "
                    "indicating possible tumor heterogeneity.")
        else:
            return ("This higher intensity suggests a region of dense fibrous or calcified tissue, "
                    "potentially indicating structural stability within the tumor.")

    if parameter_name == 'Hyper-dense region intensity (maximum)':
        if val < 30:
            return ("Low intensity in this region suggests soft tissue, which could indicate a "
                    "predominance of non-calcified components.")
        elif 30 <= val <= 70:
            return ("This suggests a mix of soft tissue with possible calcification, "
                    "commonly seen in tumors with fibrous components.")
        else:
            return ("High intensity indicates significant calcification or fibrous tissue, "
                    "often indicative of a more organized region of the tumor.")

    if parameter_name == 'Total signal energy in the tumor':
        if val < 50:
            return ("Low energy suggests less metabolic activity, which may indicate a "
                    "less aggressive region.")
        elif 50 <= val <= 150:
            return ("Moderate energy suggests normal metabolic activity, typical of tumors "
                    "with varying aggressiveness.")
        else:
            return ("High signal energy reflects increased metabolic activity, often associated "
                    "with more active or aggressive regions.")

    if parameter_name == 'Tissue heterogeneity (entropy)':
        if val < 1:
            return ("A low entropy value suggests uniform tissue composition, "
                    "potentially a more organized structure.")
        elif 1 <= val <= 3:
            return ("Moderate entropy indicates some degree of heterogeneity, "
                    "found in mixed cell-type tumors.")
        else:
            return ("High entropy suggests significant heterogeneity in tumor tissue, "
                    "often associated with irregular tumor structures.")

    if parameter_name == 'Interquartile intensity range':
        if val < 10:
            return ("A narrow intensity range suggests relatively uniform tissue, "
                    "common in tumors with less structural variation.")
        elif 10 <= val <= 30:
            return ("A moderate range suggests a mix of more and less dense regions, "
                    "implying some heterogeneity.")
        else:
            return ("A wide intensity range indicates marked heterogeneity, "
                    "with areas of both high and low density.")

    if parameter_name == 'Tissue uniformity (kurtosis)':
        if val < 2:
            return ("A low kurtosis value indicates uneven tissue distribution, "
                    "often observed in mixed cell populations.")
        elif 2 <= val <= 5:
            return ("Moderate kurtosis suggests a mix of uniform and irregular areas, "
                    "seen in various tumor types.")
        else:
            return ("A high kurtosis value suggests more uniform tissue, "
                    "often observed in less irregular tumors.")

    if parameter_name == 'Average tissue density':
        if val < 30:
            return ("Low average tissue density suggests a soft or potentially cystic component "
                    "within the tumor.")
        elif 30 <= val <= 60:
            return ("Moderate density indicates a mixture of soft and firm structures.")
        else:
            return ("High density suggests a solid, fibrous, or calcified component, "
                    "indicating a more rigid tumor.")

    if parameter_name == 'Median tissue density':
        if val < 30:
            return ("A low median density indicates a tumor with predominantly less dense tissue.")
        elif 30 <= val <= 60:
            return ("A moderate median density reflects a balance of soft and firmer tissue types.")
        else:
            return ("A high median density suggests more fibrous or calcified regions, "
                    "often stable tumor structure.")

    if parameter_name == 'Density variance':
        if val < 10:
            return ("Low density variance suggests a more homogeneous structure with "
                    "less variation in tissue composition.")
        elif 10 <= val <= 30:
            return ("Moderate density variance implies some heterogeneity within the tumor, "
                    "with diverse tissue densities.")
        else:
            return ("High density variance indicates significant variation in tissue types, "
                    "often seen in irregular or aggressive tumors.")

    if parameter_name == 'Edge sharpness (contrast)':
        if val < 1:
            return ("Poor edge sharpness suggests poorly defined margins, possibly indicative "
                    "of invasive or irregular growth.")
        elif 1 <= val <= 3:
            return ("Moderate edge sharpness indicates moderately defined margins, "
                    "suggesting a less invasive pattern.")
        else:
            return ("High edge sharpness suggests well-defined margins, typically seen in "
                    "more organized or less aggressive tumors.")

    if parameter_name == 'Tissue texture correlation':
        if val < 0.5:
            return ("Low correlation suggests disorganized structure, often seen in "
                    "aggressive or irregular tumors.")
        elif 0.5 <= val <= 0.7:
            return ("Moderate correlation suggests some organization of structure, "
                    "a mix of regular and irregular patterns.")
        else:
            return ("High correlation indicates well-organized tissue, typically "
                    "associated with less aggressive tumors.")

    if parameter_name == 'Small uniform tissue regions':
        if val < 5:
            return ("Few small uniform regions suggest a more complex, potentially "
                    "more aggressive tumor.")
        elif 5 <= val <= 20:
            return ("A moderate number of uniform tissue regions indicates a mix of "
                    "organized and disorganized areas.")
        else:
            return ("A high number of small uniform regions suggests a more organized structure, "
                    "often benign or less aggressive.")

    if parameter_name == 'Tumor texture complexity':
        if val < 2:
            return ("Low texture complexity suggests a simple structure, often observed "
                    "in benign tumors.")
        elif 2 <= val <= 4:
            return ("Moderate complexity suggests some irregularity, common in "
                    "mixed-characteristic tumors.")
        else:
            return ("High complexity indicates a highly irregular structure, often "
                    "characteristic of malignant tumors.")

    return f"No interpretation rule for {parameter_name}."

# -------------------------------------------------------------------
# 7. Updated generate_medical_report
# -------------------------------------------------------------------
def generate_medical_report(features, tumor_type):
    """
    Combine radiomics features + classification label into a textual report,
    including per-parameter interpretation from 'interpret_value'.
    """
    # We'll rename selected Radiomics keys to the user-friendly names:
    display_map = {
        'original_firstorder_10Percentile': 'Hypo-dense region intensity (minimum)',
        'original_firstorder_90Percentile': 'Hyper-dense region intensity (maximum)',
        'original_firstorder_Energy': 'Total signal energy in the tumor',
        'original_firstorder_Entropy': 'Tissue heterogeneity (entropy)',
        'original_firstorder_InterquartileRange': 'Interquartile intensity range',
        'original_firstorder_Kurtosis': 'Tissue uniformity (kurtosis)',
        'original_firstorder_Mean': 'Average tissue density',
        'original_firstorder_Median': 'Median tissue density',
        'original_firstorder_Variance': 'Density variance',
        'original_glcm_Contrast': 'Edge sharpness (contrast)',
        'original_glcm_Correlation': 'Tissue texture correlation',
        'original_glszm_SmallAreaEmphasis': 'Small uniform tissue regions',
        'original_ngtdm_Busyness': 'Tumor texture complexity',
    }

    # We'll place the final classification label under "Label" for interpretation
    features["Label"] = tumor_type

    report = "Radiological Evaluation Report:\n\n"
    report += f"Tumor Classification: {tumor_type}\n\n"

    # For each recognized key in 'display_map', interpret the value
    # If it's absent, we skip it.
    for rad_key, display_name in display_map.items():
        if rad_key in features:
            val = features[rad_key]
            # Interpret
            meaning = interpret_value(val, display_name)
            # Add to report
            report += f"{display_name}: {val}\n{meaning}\n\n"

    # Now interpret classification label if we want more detail:
    class_meaning = interpret_value(tumor_type, "Label")
    report += f"Classification Interpretation:\n{class_meaning}\n\n"

    # Additional standard insights
    report += "Insights:\n"
    report += "- Tumor heterogeneity suggests varied structure/metabolic activity.\n"
    report += "- Dense/soft tissue regions may guide treatment planning.\n"
    report += "- Ongoing imaging recommended to track changes over time.\n\n"
    report += "These findings should be correlated with clinical context for an optimal management plan.\n"

    return report

# -------------------------------------------------------------------
# 8. Route: /predict -> Segmentation
# -------------------------------------------------------------------
@app.route('/predict', methods=['POST'])
def predict_segmentation():
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400

        file = request.files['image']
        os.makedirs('uploads', exist_ok=True)

        image_path = os.path.join('uploads', file.filename)
        file.save(image_path)

        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            return jsonify({'error': 'Invalid image file'}), 400

        original_size = image.shape[:2]
        image_resized = cv2.resize(image, (256, 256))
        image_resized = image_resized / 255.0
        image_resized = np.expand_dims(image_resized, axis=(0, -1))

        prediction = segmentation_model.predict(image_resized)
        mask = (prediction[0] > 0.5).astype(np.uint8)

        mask_resized = cv2.resize(mask, (original_size[1], original_size[0]))
        mask_path = os.path.join('uploads', 'predicted_mask.png')
        cv2.imwrite(mask_path, mask_resized * 255)

        return send_file(mask_path, mimetype='image/png')

    except Exception as e:
        app.logger.error(f"Error in /predict: {e}")
        return jsonify({'error': str(e)}), 500
import SimpleITK as sitk

def preprocess_mask(mask_path):
    """
    Ensures a mask image is binary by converting any non-zero pixel to 1.
    The updated mask is then saved to the same path.

    :param mask_path: Path to the mask image file (e.g., .nii, .png, .jpg).
    """
    # Read the mask image
    mask_img = sitk.ReadImage(mask_path)

    # Convert it into a NumPy array
    mask_arr = sitk.GetArrayFromImage(mask_img)

    # Set all non-zero pixels to 1
    mask_arr[mask_arr > 0] = 1

    # Convert back to a SimpleITK image
    processed_mask = sitk.GetImageFromArray(mask_arr)

    # Preserve the original metadata (spacing, origin, direction)
    processed_mask.CopyInformation(mask_img)

    # Overwrite the original file with the binary mask
    sitk.WriteImage(processed_mask, mask_path)

# -------------------------------------------------------------------
# 9. Route: /report -> Radiomics + Classification + Detailed Interpretation
# -------------------------------------------------------------------
@app.route('/report', methods=['POST'])
def generate_report_endpoint():
    try:
        if 'image' not in request.files or 'mask' not in request.files:
            return jsonify({'error': 'Both image and mask files are required'}), 400

        image_file = request.files['image']
        mask_file = request.files['mask']
        os.makedirs('uploads', exist_ok=True)

        image_path = os.path.join('uploads', image_file.filename)
        mask_path = os.path.join('uploads', mask_file.filename)
        image_file.save(image_path)
        mask_file.save(mask_path)

        # Ensure mask is binary
        preprocess_mask(mask_path)

        # Extract features
        features = extract_radiomics_features(image_path, mask_path)

        # Classify
        tumor_type = classify_radiomics(features)

        # Generate full text report
        final_report = generate_medical_report(features, tumor_type)
        return jsonify({'report': final_report}), 200

    except Exception as e:
        app.logger.error(f"Unexpected error in /report: {e}")
        return jsonify({'error': str(e)}), 500

# -------------------------------------------------------------------
# 10. Main
# -------------------------------------------------------------------
if __name__ == '__main__':
    os.makedirs('uploads', exist_ok=True)
    app.run(debug=True)
