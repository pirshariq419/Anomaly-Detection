"""
Flask Backend for LSTM Anomaly Detection Dashboard
Modern Web Application - 2025
"""

from flask import Flask, render_template, request, jsonify, send_file
from flask_cors import CORS
import pandas as pd
import numpy as np
import json
import os
import tempfile
from werkzeug.utils import secure_filename
import traceback
from typing import Any
import gc  

from lstm_solution import main, DataProcessor

app = Flask(__name__)
CORS(app)

app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max file size
app.secret_key = 'your-secret-key-here'

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)


def convert_numpy_types(obj: Any) -> Any:
    """
    Recursively convert numpy data types to native Python types for JSON serialization.
    """
    if isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(i) for i in obj]
    elif isinstance(obj, np.generic):
        return obj.item()
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.bool_):
        return bool(obj)
    else:
        return obj


@app.route('/')
def index():
    """Main dashboard page"""
    return render_template('dashboard.html')


@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload and data preview"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400

        file = request.files['file']

        if not file or file.filename == '':
            return jsonify({'error': 'No file selected'}), 400

        if not file.filename.lower().endswith('.csv'):
            return jsonify({'error': 'Please upload a CSV file'}), 400

        filename = secure_filename(file.filename)
        input_path = os.path.join(app.config['UPLOAD_FOLDER'], 'input.csv')

        # Save uploaded file
        file.save(input_path)

        

        processor = DataProcessor()
        

        df, feature_names = processor.load_and_preprocess(input_path)

        response_data = {
            'success': True,
            'filename': filename,
            'shape': df.shape,
            'columns': feature_names,
            'total_features': len(feature_names),
            'preview': df.head(5).to_dict('records'),
            'data_info': {
                'missing_values': int(df[feature_names].isnull().sum().sum()),
                'memory_usage': f"{df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB"
            }
        }

        return jsonify(convert_numpy_types(response_data))

    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': f'Upload failed: {str(e)}'}), 500


@app.route('/analyze', methods=['POST'])
def analyze():
    """Run LSTM anomaly detection analysis"""
    try:
        params = request.json or {}
        input_path = os.path.join(app.config['UPLOAD_FOLDER'], 'input.csv')
        output_path = os.path.join(app.config['UPLOAD_FOLDER'], 'output.csv')

        if not os.path.exists(input_path):
            return jsonify({'error': 'No data file found. Please upload a file first.'}), 400

        # Run LSTM analysis
        # Note: In your LSTM training code, reduce batch_size from default to something like 16 to save memory
        result_df = main(input_path, output_path)

        anomaly_scores = result_df['Abnormality_score'].values
        threshold = params.get('threshold', 50)

        high_anomalies = np.sum(anomaly_scores > threshold)
        normal_points = len(anomaly_scores) - high_anomalies

        feature_columns = [f'top_feature_{i}' for i in range(1, 8)]
        top_features_data = result_df[feature_columns].values

        feature_frequency = {}
        for features_row in top_features_data:
            for feature in features_row:
                if feature and feature.strip():
                    feature_frequency[feature] = feature_frequency.get(feature, 0) + 1

        top_contributors = sorted(feature_frequency.items(),
                                  key=lambda x: x[1], reverse=True)[:10]

        time_series_data = [{"time": int(i), "score": float(score), "is_anomaly": bool(score > threshold)}
                            for i, score in enumerate(anomaly_scores)]

        hist_data, bin_edges = np.histogram(anomaly_scores, bins=30)
        distribution_data = [{"bin_start": float(bin_edges[i]), "bin_end": float(bin_edges[i + 1]), "count": int(hist_data[i])}
                             for i in range(len(hist_data))]

        response_data = {
            'success': True,
            'anomaly_scores': anomaly_scores.tolist(),
            'time_series_data': time_series_data,
            'distribution_data': distribution_data,
            'statistics': {
                'mean': float(np.mean(anomaly_scores)),
                'max': float(np.max(anomaly_scores)),
                'min': float(np.min(anomaly_scores)),
                'std': float(np.std(anomaly_scores)),
                'median': float(np.median(anomaly_scores))
            },
            'anomaly_summary': {
                'high_anomalies': int(high_anomalies),
                'normal_points': int(normal_points),
                'total_points': int(len(anomaly_scores)),
                'anomaly_percentage': float(high_anomalies / len(anomaly_scores) * 100)
            },
            'top_contributors': top_contributors,
            'threshold_used': int(threshold)
        }

        # Clear memory explicitly (optional)
        gc.collect()

        return jsonify(convert_numpy_types(response_data))

    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': f'Analysis failed: {str(e)}'}), 500


@app.route('/download')
def download():
    """Download analysis results"""
    try:
        output_path = os.path.join(app.config['UPLOAD_FOLDER'], 'output.csv')
        if os.path.exists(output_path):
            absolute_path = os.path.abspath(output_path)
            return send_file(
                absolute_path,
                as_attachment=True,
                download_name='anomaly_detection_results.csv',
                mimetype='text/csv'
            )
        else:
            return jsonify({'error': 'No results file found. Please run analysis first.'}), 404

    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': f'Download failed: {str(e)}'}), 500


@app.route('/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'message': 'LSTM Anomaly Detection API is running'
    })


# Error handlers
@app.errorhandler(413)
def too_large(e):
    return jsonify({'error': 'File too large. Maximum size is 50MB.'}), 413


@app.errorhandler(404)
def not_found(e):
    return jsonify({'error': 'Endpoint not found'}), 404


@app.errorhandler(500)
def server_error(e):
    return jsonify({'error': 'Internal server error'}), 500


if __name__ == '__main__':
    import os

    debug = os.environ.get('DEBUG', 'False').lower() == 'true'
    port = int(os.environ.get('PORT', 5000))

    print("Starting Flask LSTM Anomaly Detection Dashboard...")
    if debug:
        print(f"Dashboard available at: http://localhost:{port}")
    else:
        print("Running in production mode")

    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
        print(f"Created upload directory: {app.config['UPLOAD_FOLDER']}")

    # Test write permissions
    test_file = os.path.join(app.config['UPLOAD_FOLDER'], 'test.txt')
    try:
        with open(test_file, 'w') as f:
            f.write('test')
        os.remove(test_file)
        print("✅ Upload directory has write permissions")
    except Exception as e:
        print(f"❌ Upload directory permission error: {e}")

    app.run(host='0.0.0.0', port=port, debug=debug)
