"""
object_tracking.py
Handles object/person detection and tracking logic for the backend.
"""

import cv2
import numpy as np
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import os
import logging

logger = logging.getLogger(__name__)

def detect_and_track(video_path, output_path, progress_callback=None):
    """
    Run person detection and tracking on a video.
    
    Args:
        video_path: Path to input video file
        output_path: Path to save the processed video
        progress_callback: Optional callback function(progress) to report progress
        
    Returns:
        Tuple of (output_video_path, heatmap_path)
    """
    # Load YOLO model
    model = YOLO('yolov8n.pt')
    
    # Initialize DeepSORT tracker
    tracker = DeepSort(max_age=30)
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise Exception("Error opening video file")
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Initialize heatmap
    heatmap = np.zeros((height, width), dtype=np.float32)
    
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # Run YOLO detection
        results = model(frame, classes=[0])  # class 0 is person
        
        # Process detections
        detections = []
        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                if conf > 0.5:  # Confidence threshold
                    detections.append(([x1, y1, x2, y2], conf, 0))  # 0 is class_id for person
        
        # Update tracker
        tracks = tracker.update_tracks(detections, frame=frame)
        
        # Update heatmap and draw tracks
        for track in tracks:
            if not track.is_confirmed():
                continue
                
            track_id = track.track_id
            ltrb = track.to_ltrb()
            
            # Update heatmap
            x1, y1, x2, y2 = map(int, ltrb)
            heatmap[y1:y2, x1:x2] += 1
            
            # Draw bounding box and ID with better contrast
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Add black background for text
            text = f"ID: {track_id}"
            (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
            cv2.rectangle(frame, (x1, y1-text_height-10), (x1+text_width, y1), (0, 0, 0), -1)
            cv2.putText(frame, text, (x1, y1-5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        # Write frame
        out.write(frame)
        
        # Update progress
        frame_count += 1
        if progress_callback and frame_count % 10 == 0:  # Update progress every 10 frames
            progress = frame_count / total_frames
            progress_callback(progress)
            logger.debug(f"Processing frame {frame_count}/{total_frames} ({progress*100:.1f}%)")
    
    # Release resources
    cap.release()
    out.release()
    
    # Normalize and convert heatmap to visualization
    heatmap = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX)
    heatmap = heatmap.astype(np.uint8)
    
    # Use a more contrasting colormap
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_HOT)
    
    # Add a colorbar
    colorbar_height = 30
    colorbar = np.zeros((colorbar_height, width, 3), dtype=np.uint8)
    for i in range(width):
        colorbar[:, i] = cv2.applyColorMap(np.array([[int(255 * i / width)]], dtype=np.uint8), cv2.COLORMAP_HOT)[0, 0]
    
    # Add text to colorbar
    cv2.putText(colorbar, "Low", (10, colorbar_height-5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(colorbar, "High", (width-60, colorbar_height-5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Combine heatmap and colorbar
    final_heatmap = np.vstack([heatmap, colorbar])
    
    # Save heatmap
    heatmap_path = output_path.rsplit('.', 1)[0] + '_heatmap.jpg'
    cv2.imwrite(heatmap_path, final_heatmap)
    
    return output_path, heatmap_path

# Add more tracking-related utilities as needed 