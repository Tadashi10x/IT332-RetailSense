import cv2
import numpy as np
from ultralytics import solutions

# Load the floorplan image
floorplan = cv2.imread("floorplan.png")
assert floorplan is not None, "Error loading floorplan image"

# Process video
cap = cv2.VideoCapture("videos/Mall.mp4") # CHANGE THIS ONCE YOU FIND A 2x2 Video I cant find :< So I just use the same video
assert cap.isOpened(), "Error reading video file"

# Get video frame dimensions
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define the grid layout (2x2 grid for 4 camera feeds)
grid_rows, grid_cols = 2, 2
feed_width = frame_width // grid_cols
feed_height = frame_height // grid_rows

# Define the mapping for each camera feed to the floorplan
camera_mappings = [
    # Camera 1 (Top-left)
    {
        "camera_points": np.array([
            [0, 0], [feed_width, 0], [feed_width, feed_height], [0, feed_height]
        ], dtype=np.float32),
        "floorplan_points": np.array([
            [684, 190], [794, 191], [792, 293], [687, 290]
        ], dtype=np.float32)
    },
    # Camera 2 (Top-right)
    {
        "camera_points": np.array([
            [feed_width, 0], [2 * feed_width, 0], [2 * feed_width, feed_height], [feed_width, feed_height]
        ], dtype=np.float32),
        "floorplan_points": np.array([
            [396, 119], [492, 119], [489, 224], [398, 224]
        ], dtype=np.float32)
    },
    # Camera 3 (Bottom-left)
    {
        "camera_points": np.array([
            [0, feed_height], [feed_width, feed_height], [feed_width, 2 * feed_height], [0, 2 * feed_height]
        ], dtype=np.float32),
        "floorplan_points": np.array([
            [391, 380], [490, 375], [489, 292], [387, 293]
        ], dtype=np.float32)
    },
    # Camera 4 (Bottom-right)
    {
        "camera_points": np.array([
            [feed_width, feed_height], [2 * feed_width, feed_height], [2 * feed_width, 2 * feed_height], [feed_width, 2 * feed_height]
        ], dtype=np.float32),
        "floorplan_points": np.array([
            [108, 170], [223, 173], [230, 276], [106, 273]
        ], dtype=np.float32)
    }
]

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
    success, frame = cap.read()

    if not success:
        print("Video frame is empty or processing is complete.")
        break

    # Process each sub-frame (camera feed) in the grid
    for i, mapping in enumerate(camera_mappings):
    # Extract the sub-frame for the current camera feed
        row, col = divmod(i, grid_cols)
        sub_frame = frame[row * feed_height:(row + 1) * feed_height, col * feed_width:(col + 1) * feed_width]

    # Ensure the sub-frame is contiguous in memory
        sub_frame = np.ascontiguousarray(sub_frame)

    # Process the sub-frame with the heatmap
        results = heatmap(sub_frame)
        # Extract the grayscale heatmap from the results
        frame_heatmap = cv2.cvtColor(results.plot_im, cv2.COLOR_BGR2GRAY).astype(np.float32)

        # Warp the heatmap to the floorplan
        homography_matrix, _ = cv2.findHomography(mapping["camera_points"], mapping["floorplan_points"])
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
cv2.imwrite("floorplan_heatmap2x2.png", blended_output)

# Release resources
cap.release()
cv2.destroyAllWindows()

print(f"Processed {frame_count} frames. Heatmap over floorplan saved as 'floorplan_heatmap2x2.png'.")