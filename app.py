import os
import io
import pandas as pd
from flask import Flask, render_template, request, jsonify, send_file
from credit_engine import (
    engine, CATEGORY_MAPS, NUMERIC_FIELDS, FEATURE_ORDER, FEATURE_LABELS, PRESET_PERSONAS,
)

app = Flask(__name__, static_folder='static', template_folder='templates')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/schema', methods=['GET'])
def get_schema():
    return jsonify({
        'categories': CATEGORY_MAPS,
        'numeric_fields': NUMERIC_FIELDS,
        'feature_order': FEATURE_ORDER,
        'feature_labels': FEATURE_LABELS,
        'policy': engine.model_info().get('policy', {}),
    })

@app.route('/api/presets', methods=['GET'])
def get_presets():
    return jsonify(PRESET_PERSONAS)

def _optional_threshold(raw):
    """Parse a caller-supplied threshold. None/'' /'auto' means use the fitted policy."""
    if raw is None or raw == '' or str(raw).lower() == 'auto':
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        req_data = request.get_json(force=True)
        applicant_data = req_data.get('applicant', {})
        # No threshold in the payload means "use the fitted policy", which is banded by
        # exposure. A value here is an explicit manual override by a risk officer.
        threshold = _optional_threshold(req_data.get('threshold'))

        result = engine.predict_single(applicant_data, threshold=threshold)
        return jsonify({'success': True, 'data': result})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400

@app.route('/api/predict-batch', methods=['POST'])
def predict_batch():
    try:
        if 'file' not in request.files:
            return jsonify({'success': False, 'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'success': False, 'error': 'No file selected'}), 400
        
        threshold = _optional_threshold(request.form.get('threshold'))
        
        # Read CSV
        df = pd.read_csv(file)
        if len(df) == 0:
            return jsonify({'success': False, 'error': 'Uploaded CSV is empty'}), 400
        
        df_result, summary = engine.predict_batch(df, threshold=threshold)
        
        # Convert first 100 rows to dict for preview
        preview_rows = df_result.head(100).to_dict(orient='records')
        
        # Store scored CSV in memory session or return payload
        csv_buffer = io.StringIO()
        df_result.to_csv(csv_buffer, index=False)
        csv_string = csv_buffer.getvalue()
        
        return jsonify({
            'success': True,
            'summary': summary,
            'preview': preview_rows,
            'columns': df_result.columns.tolist(),
            'csv_data': csv_string
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400

@app.route('/api/model-info', methods=['GET'])
def get_model_info():
    # Served from the artifact itself rather than hard-coded here, so the documentation
    # cannot drift away from the model that is actually loaded.
    return jsonify(engine.model_info())


if __name__ == '__main__':
    print("Starting Credit Scoring Frontend on http://127.0.0.1:5000")
    app.run(host='127.0.0.1', port=5000, debug=True)
