# backend/app.py (or backend/api/app.py)
import os
import uuid # For generating unique job IDs
from flask import Flask, request, jsonify, send_from_directory, session, redirect, url_for, render_template
from functools import wraps
from werkzeug.utils import secure_filename
import threading # For simple background tasks (for production, use Celery)
from flask_cors import CORS # Import CORS
import sqlite3
import datetime

# Imports needed for generate_heatmap_and_video
import cv2
import numpy as np
from ultralytics import solutions

def generate_heatmap_and_video(
    input_video_path: str,
    input_floorplan_path: str,
    input_points_path: str,
    output_heatmap_image_path: str,
    output_processed_video_path: str,
    yolo_model_name: str = "yolov8n.pt",
    ultralytics_colormap=cv2.COLORMAP_PARULA,
    final_colormap=cv2.COLORMAP_JET,
    floorplan_blend_alpha: float = 0.7,
    heatmap_blend_alpha: float = 0.3,
    default_fps: int = 25,
    progress_callback=None # New argument for progress reporting
):
    """
    Generates a heatmap overlay on a floorplan from a video and corresponding points.
    Also produces a processed video with detections/heatmaps.
    """
    for f_path in [input_video_path, input_floorplan_path, input_points_path]:
        if not os.path.exists(f_path):
            if progress_callback:
                progress_callback(f"Error: Input file not found: {f_path}")
            raise FileNotFoundError(f"Input file not found: {f_path}")

    os.makedirs(os.path.dirname(output_heatmap_image_path), exist_ok=True)
    os.makedirs(os.path.dirname(output_processed_video_path), exist_ok=True)

    floorplan = cv2.imread(input_floorplan_path)
    if floorplan is None:
        if progress_callback:
            progress_callback(f"Error loading floorplan image from {input_floorplan_path}")
        raise ValueError(f"Error loading floorplan image from {input_floorplan_path}")

    if progress_callback:
        progress_callback("Loading video...")
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        if progress_callback:
            progress_callback(f"Error reading video file from {input_video_path}")
        raise ValueError(f"Error reading video file from {input_video_path}")

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    if fps <= 0:
        print(f"Warning: Video reported FPS of {fps}. Defaulting to {default_fps} FPS for output video.")
        fps = default_fps

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_processed_video_path, fourcc, fps, (w, h))
    if not video_writer.isOpened():
        cap.release()
        if progress_callback:
            progress_callback(f"Error creating video writer for {output_processed_video_path}")
        raise ValueError(f"Error creating video writer for {output_processed_video_path}")

    camera_points = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32)

    try:
        floorplan_points = np.loadtxt(input_points_path, dtype=np.float32, comments="#", usecols=(0, 1), delimiter=',')
        if floorplan_points.shape != (4, 2):
            raise ValueError("Floorplan points file should contain 4 points (x, y pairs).")
    except IOError:
        cap.release()
        if video_writer.isOpened(): video_writer.release()
        raise ValueError(f"Error loading floorplan points from {input_points_path}. Ensure it exists and is readable.")
    except ValueError as e:
        cap.release()
        if video_writer.isOpened(): video_writer.release()
        if progress_callback:
            progress_callback(f"Error with floorplan points data from {input_points_path}: {e}")
        raise ValueError(f"Error with floorplan points data from {input_points_path}: {e}")

    if progress_callback:
        progress_callback("Calculating homography...")
    homography_matrix, _ = cv2.findHomography(camera_points, floorplan_points)
    if homography_matrix is None:
        cap.release()
        if video_writer.isOpened(): video_writer.release()
        if progress_callback:
            progress_callback("Could not compute homography matrix. Check your points.")
        raise ValueError("Could not compute homography matrix. Check your points.")

    heatmap_obj = solutions.Heatmap(
        show=False,
        model=yolo_model_name,
        colormap=ultralytics_colormap,
    )

    floorplan_heatmap_acc = np.zeros_like(floorplan[:, :, 0], dtype=np.float32)
    frame_count = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) # Get total frames if available
    report_interval = max(1, total_frames // 20) if total_frames > 0 else 50 # Report progress roughly every 5% or 50 frames

    try:
        while cap.isOpened():
            success, im0 = cap.read()
            if not success:
                break

            frame_count += 1
            if progress_callback and frame_count % report_interval == 0:
                progress_message = f"Processing frame {frame_count}"
                if total_frames > 0:
                    percent_complete = int((frame_count / float(total_frames)) * 100)
                    progress_message += f" of approx {total_frames} ({percent_complete}%)"
                progress_callback(progress_message)
            results = heatmap_obj(im0) # Process with Ultralytics Heatmap
            if results.plot_im is not None: # results.plot_im is the frame with heatmap/detections
                video_writer.write(results.plot_im)
                frame_heatmap_gray = cv2.cvtColor(results.plot_im, cv2.COLOR_BGR2GRAY).astype(np.float32)
                warped_heatmap = cv2.warpPerspective(frame_heatmap_gray, homography_matrix, (floorplan.shape[1], floorplan.shape[0]))
                floorplan_heatmap_acc += warped_heatmap
    finally:
        cap.release()
        if video_writer.isOpened():
            video_writer.release()

    if frame_count == 0:
        if progress_callback:
            progress_callback("Error: No frames were processed from the video.")
        raise ValueError("No frames were processed from the video. Video might be empty or corrupted.")

    if progress_callback:
        progress_callback("Normalizing and blending final heatmap...")
    floorplan_heatmap_acc = cv2.normalize(floorplan_heatmap_acc, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    colored_heatmap = cv2.applyColorMap(floorplan_heatmap_acc, final_colormap)
    blended_output = cv2.addWeighted(floorplan, floorplan_blend_alpha, colored_heatmap, heatmap_blend_alpha, 0)
    if progress_callback:
        progress_callback("Saving final heatmap image...")
    cv2.imwrite(output_heatmap_image_path, blended_output)

    return {
        "status": "success",
        "message": f"Processed {frame_count} frames.",
        "output_heatmap_image": output_heatmap_image_path,
        "output_processed_video": output_processed_video_path,
    }

current_dir = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT_DIR = os.path.abspath(os.path.join(current_dir, "..", ".."))

# --- Database Setup ---
DATABASE_PATH = os.path.join(PROJECT_ROOT_DIR, 'heatmap_jobs.db')

def get_db_connection():
    conn = sqlite3.connect(DATABASE_PATH)
    conn.row_factory = sqlite3.Row # Access columns by name
    return conn

def init_db():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS jobs (
            job_id TEXT PRIMARY KEY,
            user TEXT NOT NULL,
            input_video_name TEXT,
            input_floorplan_name TEXT,
            output_heatmap_path TEXT,
            output_video_path TEXT,
            status TEXT NOT NULL,
            message TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()

app = Flask(__name__)
app.secret_key = os.urandom(24)
CORS(app)

# Configuration
UPLOAD_FOLDER = os.path.join(PROJECT_ROOT_DIR, 'project_uploads')
RESULTS_FOLDER = os.path.join(PROJECT_ROOT_DIR, 'project_results')
ALLOWED_EXTENSIONS_VIDEO = {'mp4', 'avi', 'mov'}
ALLOWED_EXTENSIONS_IMAGE = {'png', 'jpg', 'jpeg'}
ALLOWED_EXTENSIONS_POINTS = {'txt'}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

# In-memory job store for *active* jobs
jobs = {}

def allowed_file(filename, allowed_extensions):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in allowed_extensions

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'logged_in_user' not in session:
            if request.is_json:
                return jsonify(error="Authentication required"), 401
            return redirect(url_for('login_page', next=request.url))
        return f(*args, **kwargs)
    return decorated_function

def process_heatmap_task(job_id, input_video_path, input_floorplan_path, input_points_path,
                         output_heatmap_image_path, output_processed_video_path):
    
    conn_thread = None # Initialize to None

    def update_job_message_and_db(message):
        nonlocal conn_thread # To modify the outer scope conn_thread
        if job_id in jobs:
            jobs[job_id]['message'] = message
        
        # Update DB
        if not conn_thread: # Open connection if not already open for this thread
             conn_thread = get_db_connection()
        try:
            conn_thread.execute("UPDATE jobs SET message = ?, updated_at = ? WHERE job_id = ?",
                                (message, datetime.datetime.now(), job_id))
            conn_thread.commit()
        except sqlite3.Error as e:
            print(f"DB Error during progress update for job {job_id}: {e}")


    if not generate_heatmap_and_video:
        jobs[job_id]['status'] = 'error'
        db_message = 'Heatmap processing function not available.'
        jobs[job_id]['message'] = db_message
        conn_thread = get_db_connection()
        conn_thread.execute("UPDATE jobs SET status = ?, message = ?, updated_at = ? WHERE job_id = ?",
                     ('error', db_message, datetime.datetime.now(), job_id))
        conn_thread.commit()
        return

    try:
        print(f"Job {job_id}: Starting heatmap generation...")
        conn_thread = get_db_connection() # Open connection for this thread
        initial_message = "Initializing heatmap generation..."
        update_job_message_and_db(initial_message) # This will also update DB
        conn_thread.execute("UPDATE jobs SET status = ?, updated_at = ? WHERE job_id = ?",
                     ('processing', datetime.datetime.now(), job_id)) # Set status to processing
        conn_thread.commit()

        result = generate_heatmap_and_video(
            input_video_path=input_video_path,
            input_floorplan_path=input_floorplan_path,
            input_points_path=input_points_path,
            output_heatmap_image_path=output_heatmap_image_path,
            output_processed_video_path=output_processed_video_path,
            progress_callback=update_job_message_and_db
        )
        jobs[job_id]['status'] = 'completed'
        jobs[job_id]['result'] = result
        db_message_final = result.get('message', 'Processing completed successfully.')
        jobs[job_id]['message'] = db_message_final
        conn_thread.execute("UPDATE jobs SET status = ?, message = ?, output_heatmap_path = ?, output_video_path = ?, updated_at = ? WHERE job_id = ?",
                     ('completed', db_message_final, result['output_heatmap_image'], result['output_processed_video'], datetime.datetime.now(), job_id))
        conn_thread.commit()
        print(f"Job {job_id}: Completed successfully.")
    except Exception as e:
        jobs[job_id]['status'] = 'error'
        db_error_message = f"Error: {str(e)}"
        update_job_message_and_db(db_error_message)
        if conn_thread: # Ensure connection is open before trying to update status
            conn_thread.execute("UPDATE jobs SET status = ?, updated_at = ? WHERE job_id = ?",
                         ('error', datetime.datetime.now(), job_id))
            conn_thread.commit()
        print(f"Job {job_id}: Error during processing - {str(e)}")
    finally:
        if conn_thread:
            conn_thread.close()
        # Optionally remove from in-memory active jobs after a delay or based on status
        # if job_id in jobs and (jobs[job_id]['status'] == 'completed' or jobs[job_id]['status'] == 'error'):
        #     # Consider a delay before removing to allow final status polls
        #     pass


# --- Static Page Routes (for serving HTML) ---
@app.route('/')
def index_page():
    if 'logged_in_user' in session:
        return redirect(url_for('dashboard_page'))
    return redirect(url_for('login_page'))

@app.route('/login', methods=['GET'])
def login_page():
    frontend_dir = os.path.abspath(os.path.join(PROJECT_ROOT_DIR, "frontend"))
    return send_from_directory(frontend_dir, 'login.html')

@app.route('/dashboard', methods=['GET'])
@login_required
def dashboard_page():
    frontend_dir = os.path.abspath(os.path.join(PROJECT_ROOT_DIR, "frontend"))
    return send_from_directory(frontend_dir, 'dashboard.html')

# --- API Routes ---
@app.route('/api/login', methods=['POST'])
def login_api():
    data = request.get_json()
    username = data.get('username')
    password = data.get('password')
    if username == 'admin' and password == 'admin':
        session['logged_in_user'] = username
        return jsonify({"success": True, "message": "Login successful"})
    return jsonify({"success": False, "message": "Invalid credentials"}), 401

@app.route('/api/logout', methods=['POST'])
def logout_api():
    session.pop('logged_in_user', None)
    return jsonify({"success": True, "message": "Logged out successfully"})

@app.route('/api/heatmap_jobs', methods=['POST'])
@login_required
def create_heatmap_job():
    if 'videoFile' not in request.files or \
       'floorplanFile' not in request.files:
        return jsonify({"error": "Missing videoFile or floorplanFile"}), 400
    
    points_data_str = request.form.get('pointsData')
    if not points_data_str:
        return jsonify({"error": "Missing pointsData"}), 400

    video_file = request.files['videoFile']
    floorplan_file = request.files['floorplanFile']

    if not (video_file.filename and allowed_file(video_file.filename, ALLOWED_EXTENSIONS_VIDEO)):
        return jsonify({"error": "Invalid video file type"}), 400
    if not (floorplan_file.filename and allowed_file(floorplan_file.filename, ALLOWED_EXTENSIONS_IMAGE)):
        return jsonify({"error": "Invalid floorplan image type"}), 400
 
    job_id = str(uuid.uuid4())
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

    video_file.save(input_video_path)
    floorplan_file.save(input_floorplan_path)

    try:
        with open(input_points_path, 'w') as f:
            f.write(points_data_str)
    except IOError as e:
        return jsonify({"error": f"Could not write points file: {str(e)}"}), 500

    output_heatmap_image_path = os.path.join(job_results_folder, f"heatmap_{job_id}.png")
    output_processed_video_path = os.path.join(job_results_folder, f"video_{job_id}.mp4")

    conn = get_db_connection()
    try:
        conn.execute('''
            INSERT INTO jobs (job_id, user, input_video_name, input_floorplan_name, status, message, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (job_id, session['logged_in_user'], video_filename, floorplan_filename, 'pending', 'Job submitted for processing.', datetime.datetime.now(), datetime.datetime.now()))
        conn.commit()
    except sqlite3.Error as e:
        print(f"Database error on job insert: {e}")
        return jsonify({"error": "Failed to record job in database."}), 500
    finally:
        conn.close()

    jobs[job_id] = {
        'status': 'pending', # Will be updated by the thread
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

    thread = threading.Thread(target=process_heatmap_task, args=(
        job_id, input_video_path, input_floorplan_path, input_points_path,
        output_heatmap_image_path, output_processed_video_path
    ))
    thread.start()

    return jsonify({"job_id": job_id, "status": "pending", "message": "Job submitted for processing."}), 202

@app.route('/api/heatmap_jobs/<job_id>/status', methods=['GET'])
@login_required
def get_job_status(job_id):
    job = jobs.get(job_id) 
    if job:
        return jsonify({"job_id": job_id, "status": job['status'], "message": job.get('message', '')})
    else:
        conn = get_db_connection()
        db_job = conn.execute("SELECT job_id, status, message FROM jobs WHERE job_id = ? AND user = ?", (job_id, session['logged_in_user'])).fetchone()
        conn.close()
        if db_job:
            return jsonify({"job_id": db_job['job_id'], "status": db_job['status'], "message": db_job['message']})
        else:
            return jsonify({"error": "Job not found or not authorized"}), 404

@app.route('/api/heatmap_jobs/<job_id>/result/image', methods=['GET'])
@login_required
def get_heatmap_image(job_id):
    job = jobs.get(job_id)
    output_image_path = None
    if job and job['status'] == 'completed':
        output_image_path = job.get('result', {}).get('output_heatmap_image')
        if not output_image_path:
             output_image_path = job.get('output_files_expected', {}).get('image')
    
    if not output_image_path: # If not in active jobs or not completed, check DB
        conn = get_db_connection()
        db_job = conn.execute("SELECT output_heatmap_path, status FROM jobs WHERE job_id = ? AND user = ?", (job_id, session['logged_in_user'])).fetchone()
        conn.close()
        if db_job and db_job['status'] == 'completed' and db_job['output_heatmap_path']:
            output_image_path = db_job['output_heatmap_path']
        else:
            return jsonify({"error": "Job result image not found or job not completed"}), 404
    
    if not os.path.exists(output_image_path):
        return jsonify({"error": "Result image file not found on server"}), 404

    return send_from_directory(os.path.dirname(output_image_path),
                               os.path.basename(output_image_path))

@app.route('/api/heatmap_jobs/<job_id>/result/video', methods=['GET'])
@login_required
def get_processed_video(job_id):
    job = jobs.get(job_id)
    output_video_path = None
    if job and job['status'] == 'completed':
        output_video_path = job.get('result', {}).get('output_processed_video')
        if not output_video_path:
            output_video_path = job.get('output_files_expected', {}).get('video')

    if not output_video_path: # If not in active jobs or not completed, check DB
        conn = get_db_connection()
        db_job = conn.execute("SELECT output_video_path, status FROM jobs WHERE job_id = ? AND user = ?", (job_id, session['logged_in_user'])).fetchone()
        conn.close()
        if db_job and db_job['status'] == 'completed' and db_job['output_video_path']:
            output_video_path = db_job['output_video_path']
        else:
            return jsonify({"error": "Job result video not found or job not completed"}), 404
            
    if not os.path.exists(output_video_path):
        return jsonify({"error": "Result video file not found on server"}), 404
                                   
    return send_from_directory(os.path.dirname(output_video_path),
                               os.path.basename(output_video_path),
                               as_attachment=True)

@app.route('/api/heatmap_jobs/history', methods=['GET'])
@login_required
def get_job_history():
    conn = get_db_connection()
    history_jobs_cursor = conn.execute('''
        SELECT job_id, input_video_name, input_floorplan_name, status, message, created_at, updated_at
        FROM jobs WHERE user = ? ORDER BY created_at DESC
    ''', (session['logged_in_user'],))
    history_jobs = [dict(row) for row in history_jobs_cursor.fetchall()]
    conn.close()
    return jsonify(history_jobs)

if __name__ == '__main__':
    init_db() 
    if not generate_heatmap_and_video:
        print("CRITICAL: Heatmap processing function could not be loaded. The application might not work correctly.")
    app.run(debug=True, port=5000)
