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
    
    This function handles:
    - numpy.bool_ -> bool
    - numpy.int64, numpy.int32, etc. -> int
    - numpy.float64, numpy.float32, etc. -> float
    - numpy.ndarray -> list
    - Nested dictionaries and lists
    """
    if isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(i) for i in obj]
    elif isinstance(obj, np.generic):
        # Convert numpy scalar types to native Python types
        return obj.item()
    elif isinstance(obj, np.ndarray):
        # Convert numpy arrays to Python lists
        return obj.tolist()
    elif isinstance(obj, np.bool_):
        # Explicit handling for numpy boolean
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
        print("Upload request received")
        print(f"Request files: {list(request.files.keys())}")
        print(f"Request method: {request.method}")
        print(f"Content type: {request.content_type}")
        
        # Check if file is in request
        if 'file' not in request.files:
            print("ERROR: No 'file' key in request.files")
            print(f"Available keys: {list(request.files.keys())}")
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        print(f"File object: {file}")
        print(f"Filename: {file.filename}")
        
        if not file or file.filename == '':
            print("ERROR: Empty file or filename")
            return jsonify({'error': 'No file selected'}), 400
        
        if not file.filename.lower().endswith('.csv'):
            print("ERROR: Not a CSV file")
            return jsonify({'error': 'Please upload a CSV file'}), 400
        
        # Save the file
        filename = secure_filename(file.filename)
        input_path = os.path.join(app.config['UPLOAD_FOLDER'], 'input.csv')
        
        print(f"Saving to: {input_path}")
        file.save(input_path)
        print("File saved successfully")
        
        # Verify file was saved
        if not os.path.exists(input_path):
            print("ERROR: File was not saved")
            return jsonify({'error': 'File save failed'}), 500
            
        file_size = os.path.getsize(input_path)
        print(f"Saved file size: {file_size} bytes")
        
        # Process the file
        try:
            processor = DataProcessor()
            df, feature_names = processor.load_and_preprocess(input_path)
            print(f"Data loaded: {df.shape}")
            
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
            
            # Convert numpy types before returning JSON
            return jsonify(convert_numpy_types(response_data))
            
        except Exception as data_error:
            print(f"Data processing error: {str(data_error)}")
            return jsonify({'error': f'Data processing failed: {str(data_error)}'}), 500
        
    except Exception as e:
        print(f"Upload error: {str(e)}")
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
        
        print(f"Starting analysis with parameters: {params}")
        
        # Run LSTM analysis
        result_df = main(input_path, output_path)
        
        # Extract results
        anomaly_scores = result_df['Abnormality_score'].values
        threshold = params.get('threshold', 50)
        
        # Calculate statistics
        high_anomalies = np.sum(anomaly_scores > threshold)
        normal_points = len(anomaly_scores) - high_anomalies
        
        # Get top features for analysis
        feature_columns = [f'top_feature_{i}' for i in range(1, 8)]
        top_features_data = result_df[feature_columns].values
        
        # Calculate feature frequency
        feature_names = [col for col in result_df.columns 
                        if col not in ['Abnormality_score'] + feature_columns]
        feature_frequency = {}
        
        for features_row in top_features_data:
            for feature in features_row:
                if feature and feature.strip():  # Skip empty strings
                    feature_frequency[feature] = feature_frequency.get(feature, 0) + 1
        
        # Get top contributing features overall
        top_contributors = sorted(feature_frequency.items(), 
                                key=lambda x: x[1], reverse=True)[:10]
        
        # Prepare time series data for visualization
        time_series_data = []
        for i, score in enumerate(anomaly_scores):
            time_series_data.append({
                'time': int(i),  # Explicit conversion to Python int
                'score': float(score),  # Explicit conversion to Python float
                'is_anomaly': bool(score > threshold)  # Explicit conversion to Python bool
            })
        
        # Calculate distribution data
        hist_data, bin_edges = np.histogram(anomaly_scores, bins=30)
        distribution_data = []
        for i in range(len(hist_data)):
            distribution_data.append({
                'bin_start': float(bin_edges[i]),  # Explicit conversion
                'bin_end': float(bin_edges[i + 1]),  # Explicit conversion
                'count': int(hist_data[i])  # Explicit conversion
            })
        
        response_data = {
            'success': True,
            'anomaly_scores': anomaly_scores.tolist(),  # Convert to Python list
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
                'high_anomalies': int(high_anomalies),  # Convert numpy int to Python int
                'normal_points': int(normal_points),
                'total_points': int(len(anomaly_scores)),
                'anomaly_percentage': float(high_anomalies / len(anomaly_scores) * 100)
            },
            'top_contributors': top_contributors,
            'threshold_used': int(threshold)  # Ensure threshold is Python int
        }
        
        print(f"Analysis completed successfully. Found {high_anomalies} anomalies out of {len(anomaly_scores)} points")
        
        # FIXED: Convert all numpy types to native Python types before jsonify
        return jsonify(convert_numpy_types(response_data))
        
    except Exception as e:
        print(f"Analysis error: {str(e)}")
        traceback.print_exc()
        return jsonify({'error': f'Analysis failed: {str(e)}'}), 500

@app.route('/download')
def download():
    """Download analysis results"""
    try:
        output_path = os.path.join(app.config['UPLOAD_FOLDER'], 'output.csv')
        
        # Debug logging
        print(f"Attempting to download file: {output_path}")
        print(f"File exists: {os.path.exists(output_path)}")
        
        if os.path.exists(output_path):
            # Get absolute path for better reliability
            absolute_path = os.path.abspath(output_path)
            print(f"Absolute path: {absolute_path}")
            
            return send_file(
                absolute_path, 
                as_attachment=True, 
                download_name='anomaly_detection_results.csv',
                mimetype='text/csv'
            )
        else:
            # List files in upload directory for debugging
            files_in_dir = os.listdir(app.config['UPLOAD_FOLDER']) if os.path.exists(app.config['UPLOAD_FOLDER']) else []
            print(f"Files in upload directory: {files_in_dir}")
            return jsonify({'error': 'No results file found. Please run analysis first.'}), 404
            
    except Exception as e:
        print(f"Download error: {str(e)}")
        traceback.print_exc()
        return jsonify({'error': f'Download failed: {str(e)}'}), 500

from datetime import datetime

@app.route('/generate-report')
def generate_report():
    """Generate comprehensive anomaly detection report with summary and detailed data."""
    try:
        output_csv = os.path.join(app.config['UPLOAD_FOLDER'], 'output.csv')
        
        if not os.path.exists(output_csv):
            return jsonify({'error': 'No analysis results found. Please run analysis first.'}), 404

        print("Generating comprehensive report...")
        
        # Read the analysis results
        df = pd.read_csv(output_csv)
        
        # Calculate comprehensive statistics
        anomaly_scores = df['Abnormality_score'].values
        total_points = len(anomaly_scores)
        high_anomalies = int(np.sum(anomaly_scores > 50))
        normal_points = total_points - high_anomalies
        anomaly_percentage = (high_anomalies / total_points) * 100
        
        # Get top contributing features
        feature_columns = [f'top_feature_{i}' for i in range(1, 8)]
        top_features_data = df[feature_columns].values
        
        # Count feature frequency
        feature_frequency = {}
        for features_row in top_features_data:
            for feature in features_row:
                if feature and feature.strip():
                    feature_frequency[feature] = feature_frequency.get(feature, 0) + 1
        
        top_contributors = sorted(feature_frequency.items(), key=lambda x: x[1], reverse=True)[:5]
        
        # Create comprehensive summary
        summary_data = {
            'Analysis Metric': [
                'Total Data Points',
                'High Anomalies (Score > 50)',
                'Normal Points (Score ≤ 50)',
                'Anomaly Percentage (%)',
                'Maximum Anomaly Score',
                'Mean Anomaly Score',
                'Standard Deviation',
                'Median Score',
                'Top Contributing Feature',
                'Analysis Timestamp'
            ],
            'Value': [
                total_points,
                high_anomalies,
                normal_points,
                f"{anomaly_percentage:.2f}%",
                f"{np.max(anomaly_scores):.3f}",
                f"{np.mean(anomaly_scores):.3f}",
                f"{np.std(anomaly_scores):.3f}",
                f"{np.median(anomaly_scores):.3f}",
                top_contributors[0][0] if top_contributors else "None",
                datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            ]
        }
        
        summary_df = pd.DataFrame(summary_data)
        
        # Create unique timestamped filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_filename = f"LSTM_Anomaly_Report_{timestamp}.csv"
        report_path = os.path.join(app.config['UPLOAD_FOLDER'], report_filename)
        
        # Generate comprehensive report file
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("LSTM ANOMALY DETECTION COMPREHENSIVE REPORT\n")
            f.write("="*60 + "\n\n")
            
            f.write("EXECUTIVE SUMMARY\n")
            f.write("-"*20 + "\n")
            summary_df.to_csv(f, index=False)
            
            f.write("\n\nTOP CONTRIBUTING FEATURES\n")
            f.write("-"*30 + "\n")
            f.write("Feature Name,Frequency,Percentage\n")
            total_occurrences = sum(count for _, count in top_contributors)
            for feature, count in top_contributors:
                percentage = (count / total_occurrences) * 100 if total_occurrences > 0 else 0
                f.write(f"{feature},{count},{percentage:.2f}%\n")
            
            f.write("\n\nDETAILED ANOMALY DETECTION RESULTS\n")
            f.write("-"*40 + "\n")
            df.to_csv(f, index=False)
            
            f.write(f"\n\nReport generated on: {datetime.now().strftime('%Y-%m-%d at %H:%M:%S')}\n")
            f.write("Generated by: LSTM Anomaly Detection Dashboard\n")
        
        print(f"Report generated: {report_filename}")
        
        # Convert numpy types to avoid serialization issues
        return send_file(
            convert_numpy_types(report_path),
            as_attachment=True,
            download_name=report_filename,
            mimetype='text/csv'
        )
        
    except Exception as e:
        print(f"Report generation error: {str(e)}")
        traceback.print_exc()
        return jsonify({'error': f'Report generation failed: {str(e)}'}), 500



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
    
    # Production environment configuration
    debug = os.environ.get('DEBUG', 'False').lower() == 'true'
    port = int(os.environ.get('PORT', 5000))
    
    print("Starting Flask LSTM Anomaly Detection Dashboard...")
    
    if debug:
        print(f"Dashboard will be available at: http://localhost:{port}")
    else:
        print("Running in production mode")
    
    # Production-friendly run configuration
    app.run(host='0.0.0.0', port=port, debug=debug)

    print(f"Upload folder: {os.path.abspath(app.config['UPLOAD_FOLDER'])}")
    print(f"Max file size: {app.config['MAX_CONTENT_LENGTH'] / 1024 / 1024}MB")
    
    # Create upload directory if it doesn't exist
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
    
    app.run(host='0.0.0.0', port=5000, debug=True)
