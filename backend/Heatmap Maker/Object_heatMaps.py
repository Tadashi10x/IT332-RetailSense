import cv2
import numpy as np
from ultralytics import solutions
import os

# --- Configuration ---
INPUT_VIDEO_FILENAME = "Mall.mp4"  # IMPORTANT: Change this to your actual video file name
INPUT_FLOORPLAN_FILENAME = "floorplan.png"
INPUT_POINTS_FILENAME = "floorplan_points.txt"
OUTPUT_HEATMAP_IMAGE_FILENAME = "final_floorplan_heatmap.png"
OUTPUT_PROCESSED_VIDEO_FILENAME = "processed_video_heatmap.mp4"

# Determine project root directory (assuming script is in backend/Heatmap Maker/)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT_DIR = os.path.join(SCRIPT_DIR, "..", "..")

INPUT_VIDEO_PATH = os.path.join(PROJECT_ROOT_DIR, "videos", "Sample Videos", INPUT_VIDEO_FILENAME)
INPUT_FLOORPLAN_PATH = os.path.join(PROJECT_ROOT_DIR, "Images", "Sample Images", INPUT_FLOORPLAN_FILENAME)
INPUT_POINTS_PATH = os.path.join(PROJECT_ROOT_DIR, "Points", INPUT_POINTS_FILENAME)

OUTPUT_VIDEO_DIR = os.path.join(PROJECT_ROOT_DIR, "videos", "Saved Videos")
OUTPUT_IMAGE_DIR = os.path.join(PROJECT_ROOT_DIR, "Images", "Saved Images")
FULL_OUTPUT_HEATMAP_IMAGE_PATH = os.path.join(OUTPUT_IMAGE_DIR, OUTPUT_HEATMAP_IMAGE_FILENAME)
FULL_OUTPUT_PROCESSED_VIDEO_PATH = os.path.join(OUTPUT_VIDEO_DIR, OUTPUT_PROCESSED_VIDEO_FILENAME)

# Ensure output directories exist
os.makedirs(OUTPUT_VIDEO_DIR, exist_ok=True)
os.makedirs(OUTPUT_IMAGE_DIR, exist_ok=True)

# Load the floorplan image
floorplan = cv2.imread(INPUT_FLOORPLAN_PATH)
assert floorplan is not None, f"Error loading floorplan image from {INPUT_FLOORPLAN_PATH}"

# Process video
cap = cv2.VideoCapture(INPUT_VIDEO_PATH)
assert cap.isOpened(), f"Error reading video file from {INPUT_VIDEO_PATH}"

# Get video frame dimensions
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
if fps <= 0: # Handle cases where FPS might not be read correctly
    print(f"Warning: Video reported FPS of {fps}. Defaulting to 25 FPS for output video.")
    fps = 25

# Initialize video writer for the processed output
fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Or use 'XVID'
video_writer = cv2.VideoWriter(FULL_OUTPUT_PROCESSED_VIDEO_PATH, fourcc, fps, (w, h))
assert video_writer.isOpened(), f"Error creating video writer for {FULL_OUTPUT_PROCESSED_VIDEO_PATH}"

# Points in the camera's view (source points - typically the corners of the video frame)
camera_points = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32)

# Load corresponding points on the floorplan (destination points)
try:
    floorplan_points = np.loadtxt(INPUT_POINTS_PATH, dtype=np.float32, comments="#", usecols=(0, 1))
    assert floorplan_points.shape == (4, 2), "Floorplan points file should contain 4 points (x, y pairs)."
except IOError:
    assert False, f"Error loading floorplan points from {INPUT_POINTS_PATH}. Please run pointMaker.py first."
except AssertionError as e:
    assert False, f"Error with floorplan points data from {INPUT_POINTS_PATH}: {e}"

# Compute the homography matrix
homography_matrix, _ = cv2.findHomography(camera_points, floorplan_points)

# Initialize heatmap object
heatmap = solutions.Heatmap(
    show=False,  # Disable real-time display for faster processing
    model="yolo11n.pt",  # Path to the YOLO11 model file
    colormap=cv2.COLORMAP_PARULA,  # Colormap of heatmap
)

# Initialize an accumulator for the floorplan heatmap
floorplan_heatmap = np.zeros_like(floorplan[:, :, 0], dtype=np.float32)

frame_count = 0
while cap.isOpened():
    success, im0 = cap.read()

    if not success:
        print("Video frame is empty or processing is complete.")
        break

    # Process the frame with the heatmap
    results = heatmap(im0)

    # Write the processed frame (with detections/heatmap from ultralytics solution) to the output video
    if results.plot_im is not None:
        video_writer.write(results.plot_im)

    # Extract the grayscale heatmap from the results
    frame_heatmap = cv2.cvtColor(results.plot_im, cv2.COLOR_BGR2GRAY).astype(np.float32)

    # Warp the heatmap to the floorplan
    warped_heatmap = cv2.warpPerspective(frame_heatmap, homography_matrix, (floorplan.shape[1], floorplan.shape[0]))

    # Accumulate the warped heatmap onto the floorplan heatmap
    floorplan_heatmap += warped_heatmap

    frame_count += 1

# Normalize the accumulated floorplan heatmap
floorplan_heatmap = cv2.normalize(floorplan_heatmap, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

# Apply a colormap to the floorplan heatmap
colored_heatmap = cv2.applyColorMap(floorplan_heatmap, cv2.COLORMAP_JET)

# Blend the heatmap with the floorplan
blended_output = cv2.addWeighted(floorplan, 0.7, colored_heatmap, 0.3, 0)

# Save the final heatmap over the floorplan
cv2.imwrite(FULL_OUTPUT_HEATMAP_IMAGE_PATH, blended_output)

# Release resources
cap.release()
cv2.destroyAllWindows()
if video_writer.isOpened():
    video_writer.release()

print(f"Processed {frame_count} frames.")
print(f"Heatmap over floorplan saved as '{FULL_OUTPUT_HEATMAP_IMAGE_PATH}'.")
print(f"Processed video saved as '{FULL_OUTPUT_PROCESSED_VIDEO_PATH}'.")