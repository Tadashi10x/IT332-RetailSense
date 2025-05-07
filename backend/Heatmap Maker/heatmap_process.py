# backend/app.py (or backend/api/app.py)
import os
import uuid # For generating unique job IDs
from flask import Flask, request, jsonify, send_from_directory, session, redirect, url_for, render_template
from functools import wraps
from werkzeug.utils import secure_filename
import threading # For simple background tasks (for production, use Celery)
from flask_cors import CORS # Import CORS

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
    default_fps: int = 25
):
    """
    Generates a heatmap overlay on a floorplan from a video and corresponding points.
    Also produces a processed video with detections/heatmaps.
    """
    for f_path in [input_video_path, input_floorplan_path, input_points_path]:
        if not os.path.exists(f_path):
            raise FileNotFoundError(f"Input file not found: {f_path}")

    os.makedirs(os.path.dirname(output_heatmap_image_path), exist_ok=True)
    os.makedirs(os.path.dirname(output_processed_video_path), exist_ok=True)

    floorplan = cv2.imread(input_floorplan_path)
    if floorplan is None:
        raise ValueError(f"Error loading floorplan image from {input_floorplan_path}")

    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
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
        raise ValueError(f"Error with floorplan points data from {input_points_path}: {e}")

    homography_matrix, _ = cv2.findHomography(camera_points, floorplan_points)
    if homography_matrix is None:
        cap.release()
        if video_writer.isOpened(): video_writer.release()
        raise ValueError("Could not compute homography matrix. Check your points.")

    heatmap_obj = solutions.Heatmap(
        show=False,
        model=yolo_model_name,
        colormap=ultralytics_colormap,
    )

    floorplan_heatmap_acc = np.zeros_like(floorplan[:, :, 0], dtype=np.float32)
    frame_count = 0

    try:
        while cap.isOpened():
            success, im0 = cap.read()
            if not success:
                break

            results = heatmap_obj(im0) # Process with Ultralytics Heatmap
            if results.plot_im is not None: # results.plot_im is the frame with heatmap/detections
                video_writer.write(results.plot_im)
                # For accumulated heatmap, use the heatmap data directly if available,
                # or derive from plot_im if necessary.
                # Assuming results.heatmap is the raw heatmap data for accumulation:
                # If results.heatmap is not directly available, we might need to re-evaluate how to get
                # the non-colored heatmap data from the 'solutions.Heatmap' object.
                # For now, let's assume we can derive a grayscale representation for warping.
                frame_heatmap_gray = cv2.cvtColor(results.plot_im, cv2.COLOR_BGR2GRAY).astype(np.float32)
                warped_heatmap = cv2.warpPerspective(frame_heatmap_gray, homography_matrix, (floorplan.shape[1], floorplan.shape[0]))
                floorplan_heatmap_acc += warped_heatmap
            frame_count += 1
    finally:
        cap.release()
        if video_writer.isOpened():
            video_writer.release()

    if frame_count == 0:
        raise ValueError("No frames were processed from the video. Video might be empty or corrupted.")

    floorplan_heatmap_acc = cv2.normalize(floorplan_heatmap_acc, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    colored_heatmap = cv2.applyColorMap(floorplan_heatmap_acc, final_colormap)
    blended_output = cv2.addWeighted(floorplan, floorplan_blend_alpha, colored_heatmap, heatmap_blend_alpha, 0)
    cv2.imwrite(output_heatmap_image_path, blended_output)

    return {
        "status": "success",
        "message": f"Processed {frame_count} frames.",
        "output_heatmap_image": output_heatmap_image_path,
        "output_processed_video": output_processed_video_path,
    }

current_dir = os.path.dirname(os.path.abspath(__file__))
# Define the project root directory (two levels up from current_dir)
PROJECT_ROOT_DIR = os.path.abspath(os.path.join(current_dir, "..", ".."))



app = Flask(__name__)
app.secret_key = os.urandom(24) # Needed for session management; use a fixed, strong key in production

CORS(app) # Enable CORS for all routes and all origins by default

# Configuration
UPLOAD_FOLDER = os.path.join(PROJECT_ROOT_DIR, 'project_uploads') # Store uploaded files at project root
RESULTS_FOLDER = os.path.join(PROJECT_ROOT_DIR, 'project_results') # Store generated heatmaps/videos at project root
ALLOWED_EXTENSIONS_VIDEO = {'mp4', 'avi', 'mov'}
ALLOWED_EXTENSIONS_IMAGE = {'png', 'jpg', 'jpeg'}
ALLOWED_EXTENSIONS_POINTS = {'txt'}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

# In-memory job store (for simplicity; use a database for production)
jobs = {}

def allowed_file(filename, allowed_extensions):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in allowed_extensions

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'logged_in_user' not in session:
            if request.is_json: # For API calls
                return jsonify(error="Authentication required"), 401
            return redirect(url_for('login', next=request.url))
        return f(*args, **kwargs)
    return decorated_function

# --- Authentication Routes ---


def process_heatmap_task(job_id, input_video_path, input_floorplan_path, input_points_path,
                         output_heatmap_image_path, output_processed_video_path):
    """
    Worker function to run the heatmap generation.
    This should be run in a separate thread or a task queue.
    'generate_heatmap_and_video' should be available in the global scope
    if it's defined in this file.
    """
    # This check will now correctly use the globally defined function
    # or fail if it's truly not defined anywhere in this script.
    if not generate_heatmap_and_video:
        jobs[job_id]['status'] = 'error'
        jobs[job_id]['message'] = 'Heatmap processing function not available.'
        return

    try:
        print(f"Job {job_id}: Starting heatmap generation...")
        result = generate_heatmap_and_video(
            input_video_path=input_video_path,
            input_floorplan_path=input_floorplan_path,
            input_points_path=input_points_path,
            output_heatmap_image_path=output_heatmap_image_path,
            output_processed_video_path=output_processed_video_path,
            # You can pass other parameters from your function here if needed
        )
        jobs[job_id]['status'] = 'completed'
        jobs[job_id]['result'] = result
        jobs[job_id]['message'] = result.get('message', 'Processing successful.')
        print(f"Job {job_id}: Completed successfully.")
    except Exception as e:
        jobs[job_id]['status'] = 'error'
        jobs[job_id]['message'] = str(e)
        print(f"Job {job_id}: Error during processing - {str(e)}")


# --- Static Page Routes (for serving HTML) ---
@app.route('/')
def index_page():
    # This will serve as the entry point, redirecting to login or dashboard
    if 'logged_in_user' in session:
        return redirect(url_for('dashboard_page'))
    return redirect(url_for('login_page'))

@app.route('/login', methods=['GET'])
def login_page():
    # Serves the login.html page
    # Assuming login.html is in a 'templates' folder next to heatmap_process.py
    # Or adjust path to your frontend/login.html
    frontend_dir = os.path.abspath(os.path.join(PROJECT_ROOT_DIR, "frontend"))
    return send_from_directory(frontend_dir, 'login.html')

@app.route('/dashboard', methods=['GET'])
@login_required
def dashboard_page():
    # Serves the dashboard.html page
    frontend_dir = os.path.abspath(os.path.join(PROJECT_ROOT_DIR, "frontend"))
    return send_from_directory(frontend_dir, 'dashboard.html')

# --- API Routes ---

@app.route('/api/login', methods=['POST'])
def login_api():
    data = request.get_json()
    username = data.get('username')
    password = data.get('password')
    if username == 'admin' and password == 'admin': # Static credentials
        session['logged_in_user'] = username
        return jsonify({"success": True, "message": "Login successful"})
    return jsonify({"success": False, "message": "Invalid credentials"}), 401

@app.route('/api/logout', methods=['POST'])
def logout_api():
    session.pop('logged_in_user', None)
    return jsonify({"success": True, "message": "Logged out successfully"})

@app.route('/api/heatmap_jobs', methods=['POST'])
@login_required # Protect this route
def create_heatmap_job():
    if 'videoFile' not in request.files or \
       'floorplanFile' not in request.files:
        # This is the correct check for files
        return jsonify({"error": "Missing videoFile or floorplanFile"}), 400

    # Points will now come from 'pointsData' form field
    points_data_str = request.form.get('pointsData')
    if not points_data_str:
        return jsonify({"error": "Missing pointsData"}), 400

    video_file = request.files['videoFile']
    floorplan_file = request.files['floorplanFile']

    if not (video_file.filename and allowed_file(video_file.filename, ALLOWED_EXTENSIONS_VIDEO)):
        return jsonify({"error": "Invalid video file type"}), 400
    if not (floorplan_file.filename and allowed_file(floorplan_file.filename, ALLOWED_EXTENSIONS_IMAGE)):
        return jsonify({"error": "Invalid floorplan image file type"}), 400
 
    job_id = str(uuid.uuid4())
    job_upload_folder = os.path.join(UPLOAD_FOLDER, job_id)
    job_results_folder = os.path.join(RESULTS_FOLDER, job_id)
    os.makedirs(job_upload_folder, exist_ok=True)
    os.makedirs(job_results_folder, exist_ok=True)

    video_filename = secure_filename(video_file.filename)
    floorplan_filename = secure_filename(floorplan_file.filename)
    # We'll create a temporary points file on the server from pointsData
    points_filename = f"points_{job_id}.txt"

    input_video_path = os.path.join(job_upload_folder, video_filename)
    input_floorplan_path = os.path.join(job_upload_folder, floorplan_filename)
    input_points_path = os.path.join(job_upload_folder, points_filename)

    video_file.save(input_video_path)
    floorplan_file.save(input_floorplan_path)

    # Save the received pointsData to the temporary points file
    try:
        with open(input_points_path, 'w') as f:
            f.write(points_data_str)
    except IOError as e:
        return jsonify({"error": f"Could not write points file: {str(e)}"}), 500


    # Define output paths
    output_heatmap_image_path = os.path.join(job_results_folder, f"heatmap_{job_id}.png")
    output_processed_video_path = os.path.join(job_results_folder, f"video_{job_id}.mp4")

    jobs[job_id] = {
        'status': 'processing',
        'message': 'Job accepted and is being processed.',
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

    # For production, use Celery or RQ for background tasks
    thread = threading.Thread(target=process_heatmap_task, args=(
        job_id, input_video_path, input_floorplan_path, input_points_path,
        output_heatmap_image_path, output_processed_video_path
    ))
    thread.start()

    return jsonify({"job_id": job_id, "status": "processing", "message": "Job submitted for processing."}), 202

@app.route('/api/heatmap_jobs/<job_id>/status', methods=['GET'])
@login_required # Protect this route
def get_job_status(job_id):
    job = jobs.get(job_id)
    if not job:
        return jsonify({"error": "Job not found"}), 404
    return jsonify({"job_id": job_id, "status": job['status'], "message": job.get('message', '')})

@app.route('/api/heatmap_jobs/<job_id>/result/image', methods=['GET'])
@login_required # Protect this route
def get_heatmap_image(job_id):
    job = jobs.get(job_id)
    if not job or job['status'] != 'completed':
        return jsonify({"error": "Job not found or not completed"}), 404

    output_image_path = job.get('result', {}).get('output_heatmap_image')
    if not output_image_path or not os.path.exists(output_image_path):
         output_image_path = job.get('output_files_expected', {}).get('image') # Fallback
         if not output_image_path or not os.path.exists(output_image_path):
            return jsonify({"error": "Result image not found"}), 404

    return send_from_directory(os.path.dirname(output_image_path),
                               os.path.basename(output_image_path))

@app.route('/api/heatmap_jobs/<job_id>/result/video', methods=['GET'])
@login_required # Protect this route
def get_processed_video(job_id):
    job = jobs.get(job_id)
    if not job or job['status'] != 'completed':
        return jsonify({"error": "Job not found or not completed"}), 404

    output_video_path = job.get('result', {}).get('output_processed_video')
    if not output_video_path or not os.path.exists(output_video_path):
        output_video_path = job.get('output_files_expected', {}).get('video') # Fallback
        if not output_video_path or not os.path.exists(output_video_path):
            return jsonify({"error": "Result video not found"}), 404

    return send_from_directory(os.path.dirname(output_video_path),
                               os.path.basename(output_video_path),
                               as_attachment=True) # To prompt download

if __name__ == '__main__':
    # This check will now correctly use the globally defined function
    # or fail if it's truly not defined anywhere in this script.
    if not generate_heatmap_and_video:
        print("CRITICAL: Heatmap processing function could not be loaded. The application might not work correctly.")
    app.run(debug=True, port=5000) # Runs on http://127.0.0.1:5000/
