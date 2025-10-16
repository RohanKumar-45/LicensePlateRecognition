from flask import Flask, render_template, request, redirect, url_for, send_file, session
import os
import uuid
from werkzeug.utils import secure_filename

from config import UPLOAD_FOLDER, OUTPUT_FOLDER, ALLOWED_EXTENSIONS
from processors import process_image, process_video

app = Flask(__name__)
app.secret_key = os.urandom(24)

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def is_image_file(filename):
    """Check if file is an image"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'png', 'jpg', 'jpeg'}

def is_video_file(filename):
    """Check if file is a video"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'mp4', 'avi', 'mov'}

@app.route('/')
def index():
    """Render home page"""
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload"""
    if 'file' not in request.files:
        return redirect(request.url)
    
    file = request.files['file']
    
    if file.filename == '':
        return redirect(request.url)
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        unique_filename = f"{uuid.uuid4().hex}_{filename}"
        filepath = os.path.join(UPLOAD_FOLDER, unique_filename)
        
        file.save(filepath)
        
        session['filepath'] = filepath
        
        if is_image_file(filename):
            session['file_type'] = 'image'
            return redirect(url_for('process_uploaded_file'))
        elif is_video_file(filename):
            session['file_type'] = 'video'
            return redirect(url_for('process_uploaded_file'))
        else:
            return "Invalid file type"
    else:
        return "File type not allowed"

@app.route('/process')
def process_uploaded_file():
    """Process uploaded file"""
    if 'filepath' not in session:
        return redirect(url_for('index'))
    
    filepath = session['filepath']
    file_type = session['file_type']
    
    try:
        if file_type == 'image':
            results = process_image(filepath)
            
            if 'error' in results:
                return render_template('error.html', error=results['error'])
            
            # Clean up uploaded file
            if os.path.exists(filepath):
                os.remove(filepath)
            
            return render_template('image_results.html',
                                  results=results,
                                  output_image=results['output_image'],
                                  excel_file=results['excel_file'])
        
        elif file_type == 'video':
            results = process_video(filepath)
            
            if 'error' in results:
                return render_template('error.html', error=results['error'])
            
            # Clean up uploaded file
            if os.path.exists(filepath):
                os.remove(filepath)
            
            return render_template('video_results.html',
                                  results=results,
                                  output_video=results['output_video'],
                                  excel_file=results['excel_file'])
        
        else:
            return "Unknown file type"
    
    except Exception as e:
        print(f"Error processing file: {str(e)}")
        return render_template('error.html', error=str(e))

@app.route('/download/<filename>')
def download_file(filename):
    """Download processed file"""
    return send_file(os.path.join(OUTPUT_FOLDER, filename), as_attachment=True)

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=5000)