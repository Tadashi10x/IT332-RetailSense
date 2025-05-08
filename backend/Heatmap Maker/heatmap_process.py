# c:\Projects\Capstone Project\IT332-RetailSense\backend\Heatmap Maker\heatmap_process.py
import os
import uuid # For generating unique job IDs
from flask import Flask, request, jsonify, send_from_directory, session, redirect, url_for, render_template
from functools import wraps
from werkzeug.utils import secure_filename
import threading # For simple background tasks (for production, use Celery)
from flask_cors import CORS # Import CORS
import sqlite3
import datetime
import json # For parsing zone data

# Imports needed for generate_heatmap_and_video
import cv2
import numpy as np
from ultralytics import YOLO # Import YOLO directly for tracking
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
    progress_callback=None,
    zones_data_str: str = None # For zone definitions
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
    tracking_model = YOLO(yolo_model_name)
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
    unique_person_ids = set() # To store unique tracked person IDs
    report_interval = max(1, total_frames // 20) if total_frames > 0 else 50 # Report progress roughly every 5% or 50 frames

    # --- Zone Analysis Setup ---
    parsed_zones = []
    zone_activity_counts = {} # To store counts of person-frames per zone
    if zones_data_str:
        try:
            parsed_zones = json.loads(zones_data_str)
            for zone in parsed_zones:
                zone_activity_counts[zone['name']] = 0
                # Convert zone points to NumPy array for cv2.pointPolygonTest
                zone['polygon_np'] = np.array([[p['originalX'], p['originalY']] for p in zone['points']], dtype=np.int32)
        except json.JSONDecodeError:
            if progress_callback:
                progress_callback("Warning: Could not parse zone definitions.")
            parsed_zones = [] # Reset if parsing fails

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

            tracks = tracking_model.track(im0, persist=True, verbose=False, classes=[0]) # classes=[0] for person
            
            # --- Zone Activity Calculation ---
            if tracks and tracks[0].boxes.id is not None:
                person_ids_in_frame = tracks[0].boxes.id.cpu().numpy().astype(int)
                for person_id in person_ids_in_frame:
                    unique_person_ids.add(person_id)

                # Get bottom-center of bounding boxes in camera coordinates
                boxes_xyxy = tracks[0].boxes.xyxy.cpu().numpy()
                person_points_camera_frame = []
                for box in boxes_xyxy:
                    x1, y1, x2, y2 = box
                    foot_x_camera = (x1 + x2) / 2
                    foot_y_camera = y2 
                    person_points_camera_frame.append([foot_x_camera, foot_y_camera])
                
                if person_points_camera_frame and homography_matrix is not None and len(parsed_zones) > 0:
                    person_points_camera_np = np.array([person_points_camera_frame], dtype=np.float32) # Reshape for perspectiveTransform
                    transformed_points_floorplan = cv2.perspectiveTransform(person_points_camera_np, homography_matrix)
                    
                    if transformed_points_floorplan is not None:
                        for point_on_floorplan in transformed_points_floorplan[0]: # Iterate through transformed points
                            for zone in parsed_zones:
                                if cv2.pointPolygonTest(zone['polygon_np'], (point_on_floorplan[0], point_on_floorplan[1]), False) >= 0:
                                    zone_activity_counts[zone['name']] += 1

            results = heatmap_obj(im0) 
            if results.plot_im is not None: 
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
        "people_counted": len(unique_person_ids),
        "zone_activity": zone_activity_counts # Add zone activity to results
    }

current_dir = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT_DIR = os.path.abspath(os.path.join(current_dir, "..", ".."))

# --- Database Setup ---
DATABASE_PATH = os.path.join(PROJECT_ROOT_DIR, 'heatmap_jobs.db')

def get_db_connection():
    conn = sqlite3.connect(DATABASE_PATH)
    conn.row_factory = sqlite3.Row 
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
            people_counted INTEGER DEFAULT 0,
            zone_definitions TEXT,
            zone_traffic_results TEXT,
            video_event_timestamp DATETIME,
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
                         output_heatmap_image_path, output_processed_video_path,
                         zones_data_str=None): 
    
    conn_thread = None 

    def update_job_message_and_db(message):
        nonlocal conn_thread 
        if job_id in jobs:
            jobs[job_id]['message'] = message
            print(f"Job {job_id} progress: {message}") 
        
        if not conn_thread: 
             conn_thread = get_db_connection()
        try:
            conn_thread.execute("UPDATE jobs SET message = ?, updated_at = ? WHERE job_id = ?",
                                (message, datetime.datetime.now(), job_id))
            conn_thread.commit()
        except sqlite3.Error as e:
            print(f"DB Error during progress update for job {job_id}: {e}")

    try:
        print(f"Job {job_id}: Background task started. zones_data_str: {'Present' if zones_data_str else 'Not Present'}")

        conn_thread = get_db_connection() 
        processing_message = "Processing video and generating heatmap..."
        if job_id in jobs:
            jobs[job_id]['status'] = 'processing'
            jobs[job_id]['message'] = processing_message
        
        conn_thread.execute("UPDATE jobs SET status = ?, updated_at = ? WHERE job_id = ?",
                     ('processing', datetime.datetime.now(), job_id)) 
        conn_thread.commit()
        print(f"Job {job_id}: Status updated to 'processing' in DB.")
        update_job_message_and_db(processing_message) 

        if not generate_heatmap_and_video:
            raise RuntimeError("Heatmap processing function (generate_heatmap_and_video) is not available.")

        result = generate_heatmap_and_video(
            input_video_path=input_video_path,
            input_floorplan_path=input_floorplan_path,
            input_points_path=input_points_path,
            output_heatmap_image_path=output_heatmap_image_path,
            output_processed_video_path=output_processed_video_path,
            progress_callback=update_job_message_and_db,
            zones_data_str=zones_data_str 
        ) 

        db_message_final = result.get('message', 'Processing completed successfully.')
        people_counted = result.get('people_counted', 0)

        if job_id in jobs:
            jobs[job_id]['status'] = 'completed'
            jobs[job_id]['result'] = result 
            jobs[job_id]['message'] = db_message_final

        zone_activity_results_json = json.dumps(result.get('zone_activity', {}))

        conn_thread.execute("UPDATE jobs SET status = ?, message = ?, output_heatmap_path = ?, output_video_path = ?, people_counted = ?, zone_traffic_results = ?, updated_at = ? WHERE job_id = ?",
                     ('completed', db_message_final, result['output_heatmap_image'], result['output_processed_video'], people_counted, zone_activity_results_json, datetime.datetime.now(), job_id))
        conn_thread.commit()
        print(f"Job {job_id}: Completed successfully.")
    except Exception as e:
        if job_id in jobs: # Check if job_id is in the in-memory store before trying to update it
            jobs[job_id]['status'] = 'error'
            jobs[job_id]['message'] = f"Error: {str(e)}"
        db_error_message = f"Error: {str(e)}"
        print(f"Job {job_id}: Error during processing - {db_error_message}") 
        # update_job_message_and_db(db_error_message) # This might fail if conn_thread is already closed or in error
        if conn_thread: 
            try:
                conn_thread.execute("UPDATE jobs SET status = ?, message = ?, updated_at = ? WHERE job_id = ?",
                             ('error', db_error_message, datetime.datetime.now(), job_id))
                conn_thread.commit()
            except sqlite3.Error as db_e:
                 print(f"DB Error during error status update for job {job_id}: {db_e}")
    finally:
        if conn_thread:
            conn_thread.close()

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

@app.route('/uploads/<job_id>/<filename>')
@login_required
def uploaded_file(job_id, filename):
    job_specific_upload_folder = os.path.join(UPLOAD_FOLDER, job_id)
    if not os.path.exists(os.path.join(job_specific_upload_folder, filename)):
        return jsonify({"error": "File not found"}), 404
    return send_from_directory(job_specific_upload_folder, filename)

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
    video_event_time_str = request.form.get('videoEventTime') 
    zones_data_str = request.form.get('zonesData') 
    
    if not points_data_str:
        return jsonify({"error": "Missing pointsData"}), 400

    if not video_event_time_str: 
        return jsonify({"error": "Video event time is required"}), 400
    try:
        video_event_dt = datetime.datetime.fromisoformat(video_event_time_str)
    except ValueError:
        return jsonify({"error": "Invalid video event time format. Use YYYY-MM-DDTHH:MM"}), 400

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
            INSERT INTO jobs (job_id, user, input_video_name, input_floorplan_name, status, message,
                              video_event_timestamp, people_counted, zone_definitions, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (job_id, session['logged_in_user'], video_filename, floorplan_filename, 
              'pending', 'Job submitted for processing.', video_event_dt,
              0, zones_data_str if zones_data_str else None, datetime.datetime.now(), datetime.datetime.now()))
        conn.commit()
    except sqlite3.Error as e:
        print(f"Database error on job insert: {e}")
        return jsonify({"error": "Failed to record job in database."}), 500
    finally:
        conn.close()

    jobs[job_id] = {
        'status': 'pending', 
        'message': 'Job submitted, awaiting processing.',
        'video_event_timestamp': video_event_dt.isoformat(), 
        'zone_definitions_raw': zones_data_str, 
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
        output_heatmap_image_path, output_processed_video_path,
        zones_data_str 
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
        db_job = conn.execute("SELECT job_id, status, message FROM jobs WHERE job_id = ? AND user = ?", 
                              (job_id, session['logged_in_user'])).fetchone()
        conn.close()
        if db_job:
            return jsonify({"job_id": db_job['job_id'], "status": db_job['status'], "message": db_job['message']})
        else:
            return jsonify({"error": "Job not found or not authorized"}), 404

@app.route('/api/heatmap_jobs/<job_id>/result/image', methods=['GET'])
@login_required
def get_heatmap_image(job_id):
    conn = get_db_connection()
    db_job = conn.execute("SELECT output_heatmap_path, status FROM jobs WHERE job_id = ? AND user = ?", 
                          (job_id, session['logged_in_user'])).fetchone()
    conn.close()

    output_image_path = None
    if db_job and db_job['status'] == 'completed' and db_job['output_heatmap_path']:
        output_image_path = db_job['output_heatmap_path']
    elif db_job and db_job['status'] != 'completed': 
         return jsonify({"error": f"Job {job_id} is not completed. Status: {db_job['status']}"}), 404
    elif not db_job: 
        job_mem = jobs.get(job_id)
        if job_mem and job_mem['status'] == 'completed':
            output_image_path = job_mem.get('result', {}).get('output_heatmap_image')
            if not output_image_path:
                output_image_path = job_mem.get('output_files_expected', {}).get('image')
        if not output_image_path:
            return jsonify({"error": "Job result image not found or job not completed"}), 404
    
    if not output_image_path or not os.path.exists(output_image_path):
        return jsonify({"error": "Result image file not found on server or path is invalid"}), 404

    return send_from_directory(os.path.dirname(output_image_path),
                               os.path.basename(output_image_path))

@app.route('/api/heatmap_jobs/<job_id>/result/video', methods=['GET'])
@login_required
def get_processed_video(job_id):
    conn = get_db_connection()
    db_job = conn.execute("SELECT output_video_path, status FROM jobs WHERE job_id = ? AND user = ?", 
                          (job_id, session['logged_in_user'])).fetchone()
    conn.close()

    output_video_path = None
    if db_job and db_job['status'] == 'completed' and db_job['output_video_path']:
        output_video_path = db_job['output_video_path']
    elif db_job and db_job['status'] != 'completed':
         return jsonify({"error": f"Job {job_id} is not completed. Status: {db_job['status']}"}), 404
    elif not db_job:
        job_mem = jobs.get(job_id)
        if job_mem and job_mem['status'] == 'completed':
            output_video_path = job_mem.get('result', {}).get('output_processed_video')
            if not output_video_path:
                output_video_path = job_mem.get('output_files_expected', {}).get('video')
        if not output_video_path:
            return jsonify({"error": "Job result video not found or job not completed"}), 404
            
    if not output_video_path or not os.path.exists(output_video_path):
        return jsonify({"error": "Result video file not found on server or path is invalid"}), 404
                                   
    return send_from_directory(os.path.dirname(output_video_path),
                               os.path.basename(output_video_path),
                               as_attachment=True) 

@app.route('/api/heatmap_jobs/history', methods=['GET'])
@login_required
def get_job_history():
    conn = get_db_connection()
    history_jobs_cursor = conn.execute('''
       SELECT job_id, input_video_name, input_floorplan_name, status, message,
              people_counted, video_event_timestamp, zone_definitions, created_at, updated_at
        FROM jobs WHERE user = ? ORDER BY created_at DESC
    ''', (session['logged_in_user'],))
    history_jobs = [dict(row) for row in history_jobs_cursor.fetchall()]
    conn.close()
    return jsonify(history_jobs)

@app.route('/api/analytics/overview', methods=['GET'])
@login_required
def get_analytics_overview():
    conn = get_db_connection()
    current_user = session['logged_in_user']
    # Get requested chart period from query parameters, default to daily_hourly
    chart_period_type = request.args.get('chart_period', 'daily_hourly') 

    print(f"Analytics Overview: Fetching data for user '{current_user}' for today. Chart period: {chart_period_type}")

    total_visitors_cursor = conn.execute('''
        SELECT SUM(people_counted) as total_visitors
        FROM jobs
        WHERE user = ? AND status = 'completed' AND video_event_timestamp IS NOT NULL
          AND DATE(video_event_timestamp) = DATE('now', 'localtime') 
    ''', (current_user,))
    total_visitors_row = total_visitors_cursor.fetchone()
    total_visitors_today = total_visitors_row['total_visitors'] if total_visitors_row and total_visitors_row['total_visitors'] is not None else 0
    print(f"Analytics Overview: Total visitors today from DB query: {total_visitors_row['total_visitors'] if total_visitors_row else 'None'}, Calculated: {total_visitors_today}")
    
    # Initialize chart-specific data
    chart_data_to_return = {}
    peak_hour_today_str = "N/A"
    peak_hour_visitor_count = 0

    if chart_period_type == 'daily_hourly':
        hourly_data_cursor = conn.execute('''
            SELECT strftime('%H', video_event_timestamp) as hour_of_day, 
                   SUM(people_counted) as count_in_hour
            FROM jobs
            WHERE user = ? AND status = 'completed' AND video_event_timestamp IS NOT NULL
              AND DATE(video_event_timestamp) = DATE('now', 'localtime') 
            GROUP BY hour_of_day
            ORDER BY hour_of_day ASC
        ''', (current_user,))
        db_hourly_rows = hourly_data_cursor.fetchall()
        print(f"Analytics Overview (Daily Hourly): Found {len(db_hourly_rows)} hourly data rows for today.")

        hourly_visitor_data_dict = {f"{h:02d}": 0 for h in range(9, 22)} 
        max_count_for_peak_hour = 0
        peak_hour_val = None

        for row in db_hourly_rows:
            hour_str = row['hour_of_day']
            count = row['count_in_hour'] if row['count_in_hour'] is not None else 0
            if hour_str and (9 <= int(hour_str) <= 21): 
                hourly_visitor_data_dict[hour_str] = count
                if count > max_count_for_peak_hour:
                    max_count_for_peak_hour = count
                    peak_hour_val = int(hour_str)
                elif count == max_count_for_peak_hour and peak_hour_val is not None and int(hour_str) < peak_hour_val:
                    peak_hour_val = int(hour_str)

        if peak_hour_val is not None:
            peak_hour_today_str = f"{peak_hour_val:02d}:00 - {peak_hour_val + 1:02d}:00"
            peak_hour_visitor_count = max_count_for_peak_hour
        
        chart_data_to_return['hourly_visitor_data'] = [{"hour": h_str, "count": hourly_visitor_data_dict[h_str]} 
                                                       for h_str in sorted(hourly_visitor_data_dict.keys())]
        chart_data_to_return['yearly_visitor_summary'] = []  # Ensure key exists
        print(f"Analytics Overview (Daily Hourly): Peak hour: {peak_hour_today_str}, Count: {peak_hour_visitor_count}")

    elif chart_period_type == 'yearly_monthly':
        monthly_visitor_data_cursor = conn.execute('''
            SELECT strftime('%m', video_event_timestamp) as month_number, SUM(people_counted) as total_visitors
            FROM jobs
            WHERE user = ? AND status = 'completed' AND video_event_timestamp IS NOT NULL
              AND strftime('%Y', video_event_timestamp) = strftime('%Y', 'now', 'localtime')
            GROUP BY month_number ORDER BY month_number ASC
        ''', (current_user,))
        db_monthly_rows = monthly_visitor_data_cursor.fetchall()
        print(f"Analytics Overview (Yearly Monthly): Found {len(db_monthly_rows)} monthly data rows for the current year.")
        month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        yearly_summary = [{"month_name": name, "count": 0} for name in month_names]
        for row in db_monthly_rows:
            month_idx = int(row['month_number']) - 1
            if 0 <= month_idx < 12:
                yearly_summary[month_idx]["count"] = row['total_visitors'] if row['total_visitors'] is not None else 0
        chart_data_to_return['hourly_visitor_data'] = [] # Ensure key exists
        chart_data_to_return['yearly_visitor_summary'] = yearly_summary
        print(f"Analytics Overview (Yearly Monthly): Summary: {yearly_summary}")
    else: 
        print(f"Warning: Unknown chart_period_type received: {chart_period_type}")
        chart_data_to_return['hourly_visitor_data'] = []
        chart_data_to_return['yearly_visitor_summary'] = []


    zone_traffic_summary = []
    zone_results_cursor = conn.execute('''
        SELECT zone_traffic_results
        FROM jobs
        WHERE user = ? AND status = 'completed' AND video_event_timestamp IS NOT NULL
          AND DATE(video_event_timestamp) = DATE('now', 'localtime')
          AND zone_traffic_results IS NOT NULL AND zone_traffic_results != ''
    ''', (current_user,))
    
    zone_results_rows = zone_results_cursor.fetchall() 
    print(f"Analytics Overview: Found {len(zone_results_rows)} jobs with zone traffic results for today.")
    if zone_results_rows:
        print(f"Analytics Overview: First zone_traffic_results from DB: {zone_results_rows[0]['zone_traffic_results']}")

    aggregated_zone_activity = {} 
    total_zone_activity_frames = 0

    for row in zone_results_rows:
        try:
            zone_data = json.loads(row['zone_traffic_results'])
            for zone_name, count in zone_data.items():
                aggregated_zone_activity[zone_name] = aggregated_zone_activity.get(zone_name, 0) + count
                total_zone_activity_frames += count
        except json.JSONDecodeError:
            print(f"Warning: Could not parse zone_traffic_results for a job: {row['zone_traffic_results']}")
        
    print(f"Analytics Overview: Aggregated zone activity: {aggregated_zone_activity}, Total zone frames: {total_zone_activity_frames}")

    if total_zone_activity_frames > 0:
        for zone_name, count in aggregated_zone_activity.items():
            percentage = (count / float(total_zone_activity_frames)) * 100
            classification = "High" if percentage > 30 else ("Medium" if percentage > 10 else "Low")
            zone_traffic_summary.append({"name": zone_name, "percentage": percentage, "count": count, "classification": classification})

    conn.close()
    print(f"Analytics Overview: Final chart_data_to_return before jsonify: {chart_data_to_return}")
    print(f"Analytics Overview: Final chart_period_type before jsonify: {chart_period_type}")
    print(f"Analytics Overview: Final zone_traffic_summary: {zone_traffic_summary}")
    return jsonify({
        "total_visitors_today": total_visitors_today,
        "peak_hour_today": peak_hour_today_str, 
        "peak_hour_visitor_count": peak_hour_visitor_count, 
        "chart_data": chart_data_to_return, 
        "chart_period_type": chart_period_type, 
        "zone_traffic_summary": zone_traffic_summary 
    })

@app.route('/api/latest_zone_map_data', methods=['GET'])
@login_required
def get_latest_zone_map_data():
    conn = get_db_connection() 
    latest_job_cursor = conn.execute('''
        SELECT job_id, input_video_name, input_floorplan_name, output_heatmap_path, zone_definitions, zone_traffic_results
        FROM jobs
        WHERE user = ? AND status = 'completed' AND zone_definitions IS NOT NULL AND zone_traffic_results IS NOT NULL AND zone_traffic_results != ''
        ORDER BY created_at DESC
        LIMIT 1
    ''', (session['logged_in_user'],))
    latest_job = latest_job_cursor.fetchone()
    conn.close() 

    if latest_job:
        job_upload_folder = os.path.join(UPLOAD_FOLDER, latest_job['job_id'])
        floorplan_filename_on_server = latest_job['input_floorplan_name'] 
        floorplan_path_on_server = os.path.join(job_upload_folder, floorplan_filename_on_server)

        if not os.path.exists(floorplan_path_on_server):
             print(f"Error: Floorplan file for latest job not found at {floorplan_path_on_server}")
             return jsonify({"error": "Floorplan file for the latest job not found on server."}), 404

        return jsonify({
            "floorplan_url": url_for('uploaded_file', job_id=latest_job['job_id'], filename=floorplan_filename_on_server, _external=True),
            "zone_definitions": json.loads(latest_job['zone_definitions']),
            "zone_traffic_results": json.loads(latest_job['zone_traffic_results']),
            "original_floorplan_name": latest_job['input_floorplan_name'] 
        })
    else:
        return jsonify({"message": "No completed jobs with zone data found."}), 404


if __name__ == '__main__':
    init_db() 
    app.run(debug=True, port=5000)
