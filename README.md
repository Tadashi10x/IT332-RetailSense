# IT332-RetailSense

RetailSense is an AI-powered heatmap analysis system that helps retailers visualize foot traffic patterns using movement data from CCTV footage. The platform supports heatmap generation, cloud-based data storage, real-time analytics, and business insights.

## ✨ Features

-   **Heatmap Generation:** Visualize customer movement and high-traffic zones within retail spaces.
-   **CCTV Footage Analysis:** Processes existing CCTV video streams or recordings.
-   **Cloud-Based Data Storage:** Securely store and access heatmap data and analytics.
-   **Real-Time Analytics:** (If applicable) Provides up-to-the-minute insights on foot traffic.
-   **Business Insights:** Helps retailers optimize store layouts, product placements, and staffing based on data-driven patterns.

## 🛠️ Technologies Used

*Ultralytics YOLOv8 Object Detection*

-   **Programming Language:** Python
-   **Computer Vision:** OpenCV, Ultralytics (if using YOLO for object detection)
-   **Deep Learning Framework:** (e.g., TensorFlow, PyTorch - if applicable for custom models)
-   **Data Storage:** (e.g., AWS S3, Google Cloud Storage, Azure Blob Storage, a specific database like PostgreSQL with PostGIS)
-   **Backend Framework:** ()
-   **Frontend Framework:** ()
-   **Cloud Platform:** ()

## 🚀 Getting Started

### Prerequisites

*Python*
*pip install opencv-python numpy ultralytics*
*pip install -r requirements.txt*

### Installation

1.  Clone the repository:
    ```bash
    git clone https://github.com/Tadashi10x/IT332-RetailSense.git
    cd IT332-RetailSense
    ```
2.  Create and activate a virtual environment (recommended):
    *   **Create the environment:**
        ```bash
        python -m venv .venv
        ```
    *   **Activate the environment:**
        *   **Windows (PowerShell):**
            ```powershell
            .\.venv\Scripts\Activate.ps1
            ```
        *   **Windows (Command Prompt):**
            ```cmd
            .\.venv\Scripts\activate.bat
            ```
        *   **macOS / Linux (bash/zsh):**
            ```bash
            source .venv/bin/activate
            ```
3.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
4.  Deactive if you are done:
    ```bash
    deactivate
    ```

## Usage

This section guides you on how to run the main heatmap generation script (`Object_heatMaps.py`).

1.  **Navigate to the project root directory** (`IT332-RetailSense`) in your terminal if you are not already there.

2.  **Activate the virtual environment.** This step is crucial to ensure you are using the project-specific dependencies (if not already active from the installation steps).
    *   **Windows (PowerShell):**
        ```powershell
        .\.venv\Scripts\Activate.ps1
        ```
    *   **Windows (Command Prompt):**
        ```cmd
        .\.venv\Scripts\activate.bat
        ```
    *   **macOS / Linux (bash/zsh):**
        ```bash
        source .venv/bin/activate
        ```

3.  **Run the heatmap generation script:**
    Once the virtual environment is active (you should see `(.venv)` at the start of your prompt), run the script:
    ```bash
    python backend\Heatmap Maker\Object_heatMaps.py
    ```

**Additional Information:**

-   How to process a video file.
-   How to view generated heatmaps.
-   How to access the analytics dashboard (if applicable).
-   Ensure your input video (e.g., `Mall.mp4` as configured in `Object_heatMaps.py`) is in the `videos/Sample Videos/` directory.
-   Ensure your `floorplan.png` (as configured) is in the `Images/Sample Images/` directory.
-   Ensure `floorplan_points.txt` (as configured) is in the `Points/` directory. You may need to run `pointMaker.py` first to define the perspective points for your floorplan:
    ```bash
    python "backend\Floorplan Point Maker\pointMaker.py"
    ```

## 🤝 TEAM

Rigel L. Baltazar

## 📧 Contact

*rigelbaltazar2@gmail.com*

Project Link: https://github.com/Tadashi10x/IT332-RetailSense.git
