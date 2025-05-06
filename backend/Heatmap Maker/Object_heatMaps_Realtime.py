import cv2
from ultralytics import solutions
import time
import signal
import sys
import os

def signal_handler(sig, frame):
    print("\nSaving video and cleaning up...")
    if 'cap' in globals():
        cap.release()
    if 'video_writer' in globals():
        video_writer.release()
    cv2.destroyAllWindows()
    print("Video saved as 'camera_output.mp4'")
    sys.exit(0)

# Register the signal handler
signal.signal(signal.SIGINT, signal_handler)

print("Initializing camera...")

# Try different camera indices
for i in range(3):  # Try first 3 camera indices
    print(f"Trying camera index {i}...")
    cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)  # Try with DirectShow backend
    if cap.isOpened():
        print(f"Successfully opened camera {i}")
        break
    else:
        print(f"Failed to open camera {i}")
        cap.release()

if not cap.isOpened():
    print("Error: Could not open any camera. Please check your camera connection.")
    exit()

# Get camera properties
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# If FPS is 0 or invalid, set a default value
if fps <= 0:
    fps = 30  # Default to 30 FPS
    print(f"Camera reported invalid FPS ({fps}), using default value of 30 FPS")
else:
    print(f"Camera resolution: {w}x{h} at {fps} FPS")

# Initialize video writer with a more reliable codec and lower FPS for slower playback
output_file = "camera_output.mp4"
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use MP4V codec
output_fps = 10  # Save at 10   FPS for slower playback
video_writer = cv2.VideoWriter(output_file, fourcc, output_fps, (w, h))

if not video_writer.isOpened():
    print(f"Error: Could not create video writer for {output_file}")
    cap.release()
    exit()

# For object counting with heatmap, you can pass region points.
# region_points = [(20, 400), (1080, 400)]                                      # line points
# region_points = [(20, 400), (1080, 400), (1080, 360), (20, 360)]              # rectangle region
# region_points = [(20, 400), (1080, 400), (1080, 360), (20, 360), (20, 400)]   # polygon points

# Initialize heatmap object
print("Initializing heatmap...")
heatmap = solutions.Heatmap(
    show=True,  # display the output
    model="yolo11s.pt",  # path to the YOLO11 model file
    colormap=cv2.COLORMAP_PARULA,  # colormap of heatmap
    # region=region_points,  # object counting with heatmaps, you can pass region_points
    # classes=[0, 2],  # generate heatmap for specific classes i.e person and car.
)

print("Starting camera processing... Press 'q' to quit or Ctrl+C to save and exit")

frame_count = 0
try:
    # Process camera feed
    while cap.isOpened():
        success, im0 = cap.read()
        if not success:
            print("Error: Could not read from camera")
            break

        # Process frame with heatmap
        results = heatmap(im0)

        # Write frame to output
        if video_writer.isOpened():
            video_writer.write(results.plot_im)
            frame_count += 1
            if frame_count % 30 == 0:  # Print progress every 30 frames
                print(f"Processed {frame_count} frames...")

        # Break on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except Exception as e:
    print(f"An error occurred: {str(e)}")
finally:
    # Cleanup
    if 'cap' in globals():
        cap.release()
    if 'video_writer' in globals():
        video_writer.release()
    cv2.destroyAllWindows()
    
    # Verify the output file
    if os.path.exists(output_file):
        file_size = os.path.getsize(output_file)
        print(f"Video saved successfully as '{output_file}' ({file_size/1024/1024:.2f} MB)")
    else:
        print("Error: Video file was not created")