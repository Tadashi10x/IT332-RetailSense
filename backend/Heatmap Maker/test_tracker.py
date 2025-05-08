# test_tracker.py
from ultralytics import YOLO
import cv2
import os # Import os to help with path construction

# Construct the path to the video relative to the project root
# Assuming test_tracker.py is in backend/Heatmap Maker/
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT_DIR = os.path.join(SCRIPT_DIR, "..", "..") # Goes up two levels
VIDEO_FILENAME = "Mall.mp4" # Or your desired video
VIDEO_PATH = os.path.join(PROJECT_ROOT_DIR, "videos", "Sample Videos", VIDEO_FILENAME)
MODEL_NAME = "yolov8n.pt"

model = YOLO(MODEL_NAME)
cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print(f"Error: Could not open video {VIDEO_PATH}")
    exit()

frame_count = 0
unique_ids_test = set()

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    frame_count += 1
    results = model.track(frame, persist=True, verbose=True, classes=[0]) # verbose=True for more output

    if results and results[0].boxes.id is not None:
        current_ids = results[0].boxes.id.cpu().numpy().astype(int)
        print(f"Frame {frame_count}: Tracked Person IDs: {current_ids}")
        for tid in current_ids:
            unique_ids_test.add(tid)
    else:
        print(f"Frame {frame_count}: No persons tracked with IDs.")

    # Optional: Display the frame with tracks
    annotated_frame = results[0].plot() # This method draws the bounding boxes and track IDs
    cv2.imshow("YOLOv8 Tracking", annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord("q"): # Press 'q' to quit the video display
        break

cap.release()
cv2.destroyAllWindows() # Close all OpenCV windows when done

print(f"\n--- Test Complete ---")
print(f"Total frames processed: {frame_count}")
print(f"Total unique person IDs tracked: {len(unique_ids_test)}")
print(f"Unique IDs: {sorted(list(unique_ids_test))}")
