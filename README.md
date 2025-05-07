# RetailSense - Video Heatmap Generator

RetailSense is a full-stack web application designed to generate heatmaps from video footage by projecting activity onto a 2D floorplan. It allows users to upload videos and floorplans, interactively configure perspective points, view processing progress, and see a history of processed jobs.

## Features

*   **User Authentication:** Simple login page (currently static credentials: `admin`/`admin`).
*   **Interactive Dashboard:**
    *   **Configure Floorplan:**
        *   Upload floorplan images (PNG, JPG).
        *   Interactive point selection on a canvas to define 4 perspective points.
        *   Point configuration is saved in the browser's `localStorage` for persistence across browser refreshes.
    *   **Process Video:**
        *   Upload video files (MP4, AVI, MOV).
        *   Uses the currently configured floorplan points for heatmap generation.
        *   Displays text-based progress updates during video processing.
        *   Shows the generated heatmap image and provides a download link for the processed video.
    *   **History:**
        *   Displays a table of all previously processed jobs for the logged-in user.
        *   Shows job ID, input filenames, status, messages, and creation/update times.
        *   Provides links to view the result image and download the result video for completed jobs.
*   **Backend Processing:**
    *   Uses Flask as the web framework.
    *   Leverages OpenCV for image and video manipulation.
    *   Utilizes Ultralytics solutions for object detection and initial heatmap generation.
    *   Stores job history persistently in an SQLite database.
*   **Modular Design:** Core heatmap logic is separated for clarity and potential reuse.
*   **Client-Side Persistence:** Uses `localStorage` for saving the user's last floorplan point configuration across browser refreshes.

## Tech Stack

*   **Backend:** Python, Flask, OpenCV, NumPy, Ultralytics, SQLite3
*   **Frontend:** HTML, CSS, JavaScript
*   **Other:** `localStorage` for client-side configuration persistence.

## Project Structure (Simplified)

IT332-RetailSense/ ├── backend/ │ └── Heatmap Maker/ │ └── heatmap_process.py # Main Flask application and processing logic ├── frontend/ │ ├── login.html │ └── dashboard.html ├── project_uploads/ # Stores uploaded files for active jobs (created automatically) ├── project_results/ # Stores generated heatmap images and videos (created automatically) ├── heatmap_jobs.db # SQLite database for job history (created automatically) ├── .venv/ # Python virtual environment (recommended)


## Setup and Installation

### Prerequisites

*   Python 3.8+
*   `pip` (Python package installer)
*   Git (for cloning the repository, if applicable)

### Steps

1.  **Clone the Repository (if applicable):**
    ```bash
    git clone https://github.com/Tadashi10x/IT332-RetailSense.git
    cd IT332-RetailSense
    ```

2.  **Create and Activate a Virtual Environment (Recommended):**
    ```bash
    # Windows
    python -m venv .venv
    .venv\Scripts\activate

    # macOS/Linux
    python3 -m venv .venv
    source .venv/bin/activate
    ```

3.  **Install Dependencies:**
    Create a `requirements.txt` file in your project root with the following content:
    ```txt
    Flask
    Flask-CORS
    opencv-python
    numpy
    ultralytics
    werkzeug
    ```
    Then run:
    ```bash
    pip install -r requirements.txt
    ```

4.  **Run the Application:**
    *   **Start the Backend Server:**
        Open a terminal, ensure your virtual environment is activated, navigate to the `backend/Heatmap Maker/` directory, and run:
        ```bash
        python heatmap_process.py
        ```
        The backend server will typically start on `http://127.0.0.1:5000`. The SQLite database (`heatmap_jobs.db`) will be created in the project root if it doesn't exist.

    *   **Access the Frontend:**
        Open your web browser and navigate to `http://127.0.0.1:5000/`. You will be redirected to the login page.
        *(The Flask backend now serves the primary HTML pages for login and dashboard).*

## Usage

1.  **Login:** Access `http://127.0.0.1:5000/` and use the credentials:
    *   Username: `admin`
    *   Password: `admin`
2.  **Configure Floorplan:**
    *   Navigate to the "Configure Floorplan" page from the dashboard.
    *   **Step 1:** Upload your floorplan image. A preview will be shown.
    *   Click "Next: Configure Points".
    *   **Step 2:** Click on the displayed floorplan image to select 4 points in order: Top-Left, Top-Right, Bottom-Right, Bottom-Left of the area you want to map from the video.
    *   Click "Save This Point Configuration". This saves the points and the associated floorplan name/image data to your browser's local storage.
3.  **Process Video:**
    *   Navigate to the "Process Video" page.
    *   The "Loaded Point Configuration" section will show if a configuration is loaded from your browser's storage.
    *   Select the **Floorplan Image File** that corresponds to the point configuration you want to use (or the one you just configured).
    *   Select the **Video File** you want to process.
    *   Click "Generate Heatmap".
    *   The status section will show progress updates.
    *   Once completed, the resulting heatmap image will be displayed, and a link to download the processed video will be provided.
4.  **History:**
    *   Navigate to the "History" page to see a list of all your processed jobs.
    *   For completed jobs, you can view the result image or download the result video.

## Future Enhancements (Ideas)

*   More robust user management and database-backed credentials.
*   Ability to save and manage multiple named floorplan configurations.
*   Real-time heatmap generation from live camera feeds.
*   Enhanced visual progress indicators (e.g., progress bar).
*   Advanced analytics and reporting on heatmap data.
*   Option to delete jobs from history and clean up associated files.
*   Deployment to a cloud platform.

## 🤝 TEAM

Rigel L. Baltazar

## 📧 Contact

*rigelbaltazar2@gmail.com*

Project Link: https://github.com/Tadashi10x/IT332-RetailSense.git
