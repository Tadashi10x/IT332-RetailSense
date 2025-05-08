"use client";

import { useState, useEffect } from "react";
import { Upload, Play, AlertCircle, CheckCircle, Loader } from "lucide-react";
import toast from "react-hot-toast";
import { useNavigate } from "react-router-dom";
import { heatmapService } from "../../services/api";
import "../../styles/VideoProcessing.css";
import ProgressScreen from "../../components/ProgressScreen";

const getProgressPercent = (statusMessage) => {
  // Try to extract percentage from status message
  if (!statusMessage) return null;
  const match = statusMessage.match(/(\d+)%/);
  if (match) {
    return parseInt(match[1], 10);
  }
  return null;
};

const VideoProcessing = () => {
  const [file, setFile] = useState(null);
  const [floorplan, setFloorplan] = useState(null);
  const [pointsData, setPointsData] = useState("");
  const [isProcessing, setIsProcessing] = useState(false);
  const [processingStep, setProcessingStep] = useState(0);
  const [processingComplete, setProcessingComplete] = useState(false);
  const [jobId, setJobId] = useState(null);
  const [statusMessage, setStatusMessage] = useState("");
  const [backendError, setBackendError] = useState(null);
  const [progressPercent, setProgressPercent] = useState(null);
  const navigate = useNavigate();

  // Poll for job status if we have a jobId and are processing
  useEffect(() => {
    let intervalId;

    if (jobId && isProcessing) {
      intervalId = setInterval(async () => {
        try {
          const response = await heatmapService.getJobStatus(jobId);
          setStatusMessage(response.message || "Processing video...");

          // Update processing step based on message content
          if (response.message && response.message.includes("YOLO")) {
            setProcessingStep(1);
          } else if (response.message && response.message.includes("track")) {
            setProcessingStep(2);
          } else if (
            (response.message && response.message.includes("Normalizing")) ||
            (response.message && response.message.includes("Saving"))
          ) {
            setProcessingStep(3);
          }

          // Check if processing is complete
          if (response.status === "completed") {
            setIsProcessing(false);
            setProcessingComplete(true);
            clearInterval(intervalId);
            toast.success("Video processing complete");
          } else if (response.status === "error") {
            setIsProcessing(false);
            clearInterval(intervalId);
            toast.error(`Processing failed: ${response.message}`);
          }
        } catch (error) {
          console.error("Error checking job status:", error);
          // Don't stop polling on network errors, they might be temporary
          setStatusMessage("Waiting for server response...");
        }
      }, 2000); // Poll every 2 seconds
    }

    return () => {
      if (intervalId) clearInterval(intervalId);
    };
  }, [jobId, isProcessing]);

  useEffect(() => {
    // Update progress percent when statusMessage changes
    const percent = getProgressPercent(statusMessage);
    setProgressPercent(percent);
  }, [statusMessage]);

  useEffect(() => {
    if (processingComplete && jobId) {
      // Navigate to heatmap page after a short delay
      setTimeout(() => {
        navigate(`/heatmap-generation?jobId=${jobId}`);
      }, 1200);
    }
  }, [processingComplete, jobId, navigate]);

  const handleFileChange = (e) => {
    const selectedFile = e.target.files[0];
    if (!selectedFile) return;
    if (!selectedFile.type.includes("video/")) {
      toast.error("Please upload a valid video file");
      return;
    }
    setFile(selectedFile);
    setBackendError(null);
  };

  const handleFloorplanChange = (e) => {
    const selectedFile = e.target.files[0];
    if (!selectedFile) return;
    if (!selectedFile.type.match(/image\/(png|jpg|jpeg)/)) {
      toast.error("Please upload a valid floorplan image (PNG, JPG, JPEG)");
      return;
    }
    setFloorplan(selectedFile);
    setBackendError(null);
  };

  const handlePointsChange = (e) => {
    setPointsData(e.target.value);
  };

  const isReadyToProcess = file && floorplan && pointsData.trim().length > 0 && !isProcessing && !processingComplete && !backendError;

  const handleProcessVideo = async () => {
    if (!file || !floorplan || !pointsData.trim()) {
      toast.error("Please select a video, a floorplan image, and enter points data.");
      return;
    }
    setIsProcessing(true);
    setProcessingStep(0);
    setBackendError(null);
    try {
      const formData = new FormData();
      formData.append("videoFile", file);
      formData.append("floorplanFile", floorplan);
      formData.append("pointsData", pointsData);

      // Console log all FormData keys and values/files for debugging
      for (let pair of formData.entries()) {
        if (pair[1] instanceof File) {
          console.log(`FormData: ${pair[0]} = [File] name: ${pair[1].name}, size: ${pair[1].size}`);
        } else {
          console.log(`FormData: ${pair[0]} = ${pair[1]}`);
        }
      }

      const response = await heatmapService.createJob(formData);
      setJobId(response.job_id);
      setStatusMessage("Video uploaded and processing started");
      toast.success("Video uploaded and processing started");
    } catch (error) {
      console.error("Error processing video:", error);
      setIsProcessing(false);
      setBackendError(error.error || "Failed to process video");
      toast.error(error.error || "Failed to process video");
    }
  };

  const resetProcess = () => {
    setFile(null);
    setFloorplan(null);
    setPointsData("");
    setProcessingStep(0);
    setProcessingComplete(false);
    setJobId(null);
    setStatusMessage("");
    setBackendError(null);
  };

  const viewHeatmap = () => {
    if (jobId) {
      navigate(`/heatmap-generation?jobId=${jobId}`);
    }
  };

  const tryAgain = () => {
    setBackendError(null);
    setIsProcessing(false);
  };

  return (
    <div className="video-processing-container">
      <ProgressScreen
        show={isProcessing}
        percent={progressPercent || 0}
        label="Processing Video..."
        statusMessage={statusMessage}
      />
      <h1 className="page-title">Video Processing</h1>

      <div className="upload-card">
        <h2 className="section-title">Upload CCTV Footage</h2>

        <div className="upload-area">
          <label className="upload-label">
            {file ? (
              <div className="upload-preview">
                <video className="file-thumbnail-large" src={URL.createObjectURL(file)} controls />
                <div className="file-info-inside">
                  <p className="file-name">{file.name}</p>
                  <p className="file-size">{(file.size / (1024 * 1024)).toFixed(2)} MB</p>
                </div>
              </div>
            ) : (
              <div className="upload-content">
                <Upload className="upload-icon" />
                <p className="upload-text">
                  <span className="upload-text-bold">Click to upload</span> or drag and drop
                </p>
                <p className="upload-format">MP4, AVI, MOV (MAX. 500MB)</p>
              </div>
            )}
            <input
              type="file"
              className="upload-input"
              accept="video/*"
              onChange={handleFileChange}
              disabled={isProcessing}
            />
          </label>
        </div>

        <div className="upload-area">
          <label className="upload-label">
            {floorplan ? (
              <div className="upload-preview">
                <img className="file-thumbnail-large" src={URL.createObjectURL(floorplan)} alt="Floorplan Preview" />
                <div className="file-info-inside">
                  <p className="file-name">{floorplan.name}</p>
                  <p className="file-size">{(floorplan.size / (1024 * 1024)).toFixed(2)} MB</p>
                </div>
              </div>
            ) : (
              <div className="upload-content">
                <Upload className="upload-icon" />
                <p className="upload-text">
                  <span className="upload-text-bold">Click to upload</span> or drag and drop
                </p>
                <p className="upload-format">PNG, JPG, JPEG (MAX. 10MB)</p>
              </div>
            )}
            <input
              type="file"
              className="upload-input"
              accept="image/png, image/jpg, image/jpeg"
              onChange={handleFloorplanChange}
              disabled={isProcessing}
            />
          </label>
        </div>

        <div className="points-area">
          <label className="points-label">
            Points Data (format: x,y per line, 4 points required):
            <textarea
              className="points-input"
              rows={4}
              value={pointsData}
              onChange={handlePointsChange}
              placeholder={"e.g.\n0,0\n800,0\n800,600\n0,600"}
              disabled={isProcessing}
            />
          </label>
        </div>

        {backendError && (
          <div className="error-message">
            <AlertCircle className="error-icon" />
            <div className="error-content">
              <p className="error-title">Connection Error</p>
              <p className="error-description">{backendError}</p>
              <p className="error-help">
                Please make sure the backend server is running at
                http://localhost:5000
              </p>
              <button onClick={tryAgain} className="try-again-button">
                Try Again
              </button>
            </div>
          </div>
        )}

        <button
          onClick={handleProcessVideo}
          className="process-button"
          disabled={!isReadyToProcess}
        >
          <Play className="button-icon" /> Process Video
        </button>

        {isProcessing && (
          <div className="processing-section">
            <h3 className="processing-title">Processing Video</h3>
            {statusMessage && <p className="status-message">{statusMessage}</p>}

            <div className="processing-steps">
              <div className="processing-step">
                <div
                  className={`step-indicator ${
                    processingStep >= 1 ? "active" : ""
                  }`}
                >
                  {processingStep > 1 ? (
                    <CheckCircle className="step-icon" />
                  ) : processingStep === 1 ? (
                    <Loader className="spinner" />
                  ) : (
                    "1"
                  )}
                </div>
                <div className="step-info">
                  <p className="step-title">Person Detection with YOLO</p>
                  <p className="step-description">
                    Identifying individuals in video frames
                  </p>
                </div>
              </div>

              <div className="processing-step">
                <div
                  className={`step-indicator ${
                    processingStep >= 2 ? "active" : ""
                  }`}
                >
                  {processingStep > 2 ? (
                    <CheckCircle className="step-icon" />
                  ) : processingStep === 2 ? (
                    <Loader className="spinner" />
                  ) : (
                    "2"
                  )}
                </div>
                <div className="step-info">
                  <p className="step-title">Movement Tracking with Deep SORT</p>
                  <p className="step-description">
                    Tracking individuals across video frames
                  </p>
                </div>
              </div>

              <div className="processing-step">
                <div
                  className={`step-indicator ${
                    processingStep >= 3 ? "active" : ""
                  }`}
                >
                  {processingStep > 3 ? (
                    <CheckCircle className="step-icon" />
                  ) : processingStep === 3 ? (
                    <Loader className="spinner" />
                  ) : (
                    "3"
                  )}
                </div>
                <div className="step-info">
                  <p className="step-title">Generating Heatmap</p>
                  <p className="step-description">
                    Creating visualization of foot traffic
                  </p>
                </div>
              </div>
            </div>
          </div>
        )}

        {processingComplete && (
          <div className="completion-section">
            <div className="success-message">
              <CheckCircle className="success-icon" />
              <p>Video processing completed successfully!</p>
            </div>

            <div className="completion-actions">
              <button onClick={resetProcess} className="secondary-button">
                Process Another Video
              </button>
              <button onClick={viewHeatmap} className="primary-button">
                View Heatmap
              </button>
            </div>
          </div>
        )}
      </div>

      <div className="guidelines-card">
        <h2 className="section-title">Processing Guidelines</h2>

        <div className="guidelines-list">
          <div className="guideline-item">
            <AlertCircle className="guideline-icon" />
            <p className="guideline-text">
              <span className="guideline-highlight">Video Quality:</span> For
              best results, use footage with good lighting and minimal
              obstructions.
            </p>
          </div>

          <div className="guideline-item">
            <AlertCircle className="guideline-icon" />
            <p className="guideline-text">
              <span className="guideline-highlight">Processing Time:</span>{" "}
              Larger videos may take longer to process. Please be patient.
            </p>
          </div>

          <div className="guideline-item">
            <AlertCircle className="guideline-icon" />
            <p className="guideline-text">
              <span className="guideline-highlight">Backend Server:</span> Make
              sure the backend server is running at http://localhost:5000 before
              processing videos.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
};

export default VideoProcessing;
