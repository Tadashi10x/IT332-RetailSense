"""
app.py
Flask entry point for the backend, using refactored modules.
"""

import os
import uuid
import threading
import logging
from flask import Flask, request, jsonify, session, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename
from job_manager import init_db, get_db_connection
from video_processing import validate_video_file
from object_tracking import detect_and_track
from heatmap_maker import blend_heatmap
from utils import hash_password, verify_password
import datetime
import cv2

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = b'supersecretkey'  # Replace with a secure key in production

# Configure CORS properly
CORS(app, resources={
    r"/api/*": {
        "origins": ["http://localhost:5173"],  # Frontend URL
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization"],
        "supports_credentials": True
    }
})

UPLOAD_FOLDER = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../project_uploads'))
RESULTS_FOLDER = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../project_results'))
ALLOWED_EXTENSIONS_VIDEO = {'mp4', 'avi', 'mov'}
ALLOWED_EXTENSIONS_IMAGE = {'png', 'jpg', 'jpeg'}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

jobs = {}

def allowed_file(filename, allowed_extensions):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_extensions

def process_video_job(job_id):
    """Process a video job in the background."""
    try:
        job = jobs[job_id]
        job['status'] = 'processing'
        job['message'] = 'Starting video processing...'

        # Validate video file
        video_path = job['input_files']['video']
        cap = validate_video_file(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        # Update status for YOLO detection
        job['message'] = 'Running YOLO detection (0%)'
        
        # Run object detection and tracking
        output_video_path, heatmap_path = detect_and_track(
            video_path,
            job['output_files_expected']['video'],
            progress_callback=lambda p: update_job_progress(job_id, 'YOLO detection', p)
        )

        # Update status for heatmap generation
        job['message'] = 'Processing completed successfully'
        
        # Mark job as completed
        job['status'] = 'completed'
        job['message'] = 'Processing completed successfully'
        
        # Update database
        conn = get_db_connection()
        conn.execute('''
            UPDATE jobs 
            SET status = ?, message = ?, updated_at = CURRENT_TIMESTAMP, output_heatmap_path = ?
            WHERE job_id = ?
        ''', (job['status'], job['message'], heatmap_path, job_id))
        conn.commit()
        conn.close()

    except Exception as e:
        # Handle any errors during processing
        job['status'] = 'error'
        job['message'] = f'Error during processing: {str(e)}'
        logger.error(f"Error processing job {job_id}: {str(e)}", exc_info=True)
        
        # Update database with error
        conn = get_db_connection()
        conn.execute('''
            UPDATE jobs 
            SET status = ?, message = ?, updated_at = CURRENT_TIMESTAMP
            WHERE job_id = ?
        ''', (job['status'], job['message'], job_id))
        conn.commit()
        conn.close()

def update_job_progress(job_id, stage, progress):
    """Update job progress in both memory and database."""
    job = jobs[job_id]
    job['message'] = f'{stage} ({int(progress * 100)}%)'
    
    # Update database
    conn = get_db_connection()
    conn.execute('''
        UPDATE jobs 
        SET message = ?, updated_at = CURRENT_TIMESTAMP
        WHERE job_id = ?
    ''', (job['message'], job_id))
    conn.commit()
    conn.close()

@app.route('/api/heatmap_jobs', methods=['POST'])
def create_heatmap_job():
    try:
        logger.debug("Received job creation request")
        logger.debug(f"Files in request: {request.files}")
        logger.debug(f"Form data: {request.form}")

        if 'videoFile' not in request.files or 'floorplanFile' not in request.files:
            logger.error("Missing required files")
            return jsonify({"error": "Missing videoFile or floorplanFile"}), 400
        
        points_data_str = request.form.get('pointsData')
        if not points_data_str:
            logger.error("Missing points data")
            return jsonify({"error": "Missing pointsData"}), 400

        video_file = request.files['videoFile']
        floorplan_file = request.files['floorplanFile']

        logger.debug(f"Video file: {video_file.filename}")
        logger.debug(f"Floorplan file: {floorplan_file.filename}")

        if not (video_file.filename and allowed_file(video_file.filename, ALLOWED_EXTENSIONS_VIDEO)):
            logger.error("Invalid video file type")
            return jsonify({"error": "Invalid video file type"}), 400
        if not (floorplan_file.filename and allowed_file(floorplan_file.filename, ALLOWED_EXTENSIONS_IMAGE)):
            logger.error("Invalid floorplan image type")
            return jsonify({"error": "Invalid floorplan image type"}), 400

        job_id = str(uuid.uuid4())
        logger.debug(f"Generated job ID: {job_id}")

        job_upload_folder = os.path.join(UPLOAD_FOLDER, job_id)
        job_results_folder = os.path.join(RESULTS_FOLDER, job_id)
        os.makedirs(job_upload_folder, exist_ok=True)
        os.makedirs(job_results_folder, exist_ok=True)

        video_filename = secure_filename(video_file.filename)
        floorplan_filename = secure_filename(floorplan_file.filename)
        points_filename = f"points_{job_id}.txt"

        input_video_path = os.path.join(job_upload_folder, video_filename)
        input_floorplan_path = os.path.join(job_upload_folder, floorplan_filename)
        input_points_path = os.path.join(job_upload_folder, points_filename)

        logger.debug(f"Saving files to: {job_upload_folder}")
        video_file.save(input_video_path)
        floorplan_file.save(input_floorplan_path)
        with open(input_points_path, 'w') as f:
            f.write(points_data_str)

        output_heatmap_image_path = os.path.join(job_results_folder, f"video_{job_id}_heatmap.jpg")
        output_processed_video_path = os.path.join(job_results_folder, f"video_{job_id}.mp4")

        # Create job entry
        jobs[job_id] = {
            'status': 'pending',
            'message': 'Job submitted, awaiting processing.',
            'input_files': {
                'video': input_video_path,
                'floorplan': input_floorplan_path,
                'points': input_points_path
            },
            'output_files_expected': {
                'image': output_heatmap_image_path,
                'video': output_processed_video_path
            }
        }

        # Get current user if logged in
        current_user = session.get('logged_in_user')
        logger.debug(f"Current user: {current_user}")

        # Create database entry
        conn = get_db_connection()
        try:
            logger.debug("Creating database entry")
            conn.execute('''
                INSERT INTO jobs (job_id, user, input_video_name, input_floorplan_name, status, message)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (job_id, current_user, video_filename, floorplan_filename, 'pending', 'Job submitted, awaiting processing.'))
            conn.commit()
            logger.debug("Database entry created successfully")
        except Exception as db_error:
            logger.error(f"Database error: {str(db_error)}")
            raise
        finally:
            conn.close()

        # Start processing in background thread
        processing_thread = threading.Thread(target=process_video_job, args=(job_id,))
        processing_thread.daemon = True
        processing_thread.start()

        return jsonify({"job_id": job_id, "status": "pending", "message": "Job submitted for processing."}), 202
    except Exception as e:
        logger.error(f"Error in create_heatmap_job: {str(e)}", exc_info=True)
        return jsonify({"error": f"Server error: {str(e)}"}), 500

@app.route('/api/heatmap_jobs/<job_id>/status', methods=['GET'])
def get_job_status(job_id):
    job = jobs.get(job_id)
    if job:
        return jsonify({"job_id": job_id, "status": job['status'], "message": job.get('message', '')})
    else:
        conn = get_db_connection()
        db_job = conn.execute("SELECT job_id, status, message FROM jobs WHERE job_id = ?", (job_id,)).fetchone()
        conn.close()
        if db_job:
            return jsonify({"job_id": db_job['job_id'], "status": db_job['status'], "message": db_job['message']})
        else:
            return jsonify({"error": "Job not found or not authorized"}), 404

@app.route('/api/heatmap_jobs/<job_id>/result/image', methods=['GET'])
def get_heatmap_image(job_id):
    conn = get_db_connection()
    job_row = conn.execute("SELECT * FROM jobs WHERE job_id = ?", (job_id,)).fetchone()
    conn.close()
    if not job_row or job_row['status'] != 'completed':
        return jsonify({"error": "Job not found or not completed"}), 404

    output_image_path = job_row['output_heatmap_path'] if 'output_heatmap_path' in job_row.keys() else None
    if not output_image_path or not os.path.exists(output_image_path):
        # Try .jpg if .png not found
        jpg_path = output_image_path.replace('.png', '.jpg') if output_image_path else None
        if jpg_path and os.path.exists(jpg_path):
            output_image_path = jpg_path
        else:
            return jsonify({"error": "Result image file not found on server"}), 404
    return send_from_directory(os.path.dirname(output_image_path), os.path.basename(output_image_path))

@app.route('/api/heatmap_jobs/<job_id>/result/video', methods=['GET'])
def get_processed_video(job_id):
    conn = get_db_connection()
    job_row = conn.execute("SELECT * FROM jobs WHERE job_id = ?", (job_id,)).fetchone()
    conn.close()
    if not job_row or job_row['status'] != 'completed':
        return jsonify({"error": "Job not found or not completed"}), 404

    output_video_path = job_row['output_video_path'] if 'output_video_path' in job_row.keys() else None
    if not output_video_path or not os.path.exists(output_video_path):
        return jsonify({"error": "Result video file not found on server"}), 404
    return send_from_directory(os.path.dirname(output_video_path), os.path.basename(output_video_path), as_attachment=True)

@app.route('/api/heatmap_jobs/history', methods=['GET'])
def get_job_history():
    conn = get_db_connection()
    history_jobs_cursor = conn.execute('''
        SELECT job_id, input_video_name, input_floorplan_name, status, message, created_at, updated_at
        FROM jobs ORDER BY created_at DESC
    ''')
    history_jobs = [dict(row) for row in history_jobs_cursor.fetchall()]
    conn.close()
    return jsonify(history_jobs)

@app.route('/api/register', methods=['POST'])
def register():
    data = request.get_json()
    username = data.get('username')
    password = data.get('password')
    email = data.get('email')
    if not all([username, password, email]):
        return jsonify({"error": "Missing required fields"}), 400
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT username FROM users WHERE username = ? OR email = ?", (username, email))
        if cursor.fetchone():
            return jsonify({"error": "Username or email already exists"}), 400
        password_hash = hash_password(password)
        cursor.execute(
            "INSERT INTO users (username, password_hash, email) VALUES (?, ?, ?)",
            (username, password_hash, email)
        )
        conn.commit()
        return jsonify({"success": True, "message": "Registration successful"}), 201
    except Exception as e:
        return jsonify({"error": f"Database error: {str(e)}"}), 500
    finally:
        conn.close()

@app.route('/api/login', methods=['POST'])
def login_api():
    data = request.get_json()
    username = data.get('username')
    password = data.get('password')
    if not username or not password:
        return jsonify({"error": "Missing username or password"}), 400
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT password_hash FROM users WHERE username = ?", (username,))
        user = cursor.fetchone()
        if user and verify_password(user['password_hash'], password):
            session['logged_in_user'] = username
            return jsonify({"success": True, "message": "Login successful"})
        return jsonify({"error": "Invalid credentials"}), 401
    except Exception as e:
        return jsonify({"error": f"Database error: {str(e)}"}), 500
    finally:
        conn.close()

@app.route('/api/logout', methods=['POST'])
def logout_api():
    session.pop('logged_in_user', None)
    return jsonify({"success": True, "message": "Logged out successfully"})

@app.route('/api/user', methods=['GET'])
def get_user_info():
    username = session.get('logged_in_user')
    if not username:
        return jsonify({"error": "Not logged in"}), 401
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT username, email, created_at FROM users WHERE username = ?", (username,))
        user = cursor.fetchone()
        if user:
            return jsonify({
                "username": user['username'],
                "email": user['email'],
                "created_at": user['created_at']
            })
        return jsonify({"error": "User not found"}), 404
    except Exception as e:
        return jsonify({"error": f"Database error: {str(e)}"}), 500
    finally:
        conn.close()

if __name__ == '__main__':
    init_db()
    app.run(host='0.0.0.0', port=5000, debug=True) 