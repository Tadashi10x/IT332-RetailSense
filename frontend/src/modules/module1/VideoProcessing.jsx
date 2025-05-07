import { useState } from "react";
import { Upload, Play, AlertCircle, CheckCircle, Loader } from "lucide-react";
import toast from "react-hot-toast";
import "./VideoProcessing.css";

const VideoProcessing = () => {
  const [file, setFile] = useState(null);
  const [isUploading, setIsUploading] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  const [processingStep, setProcessingStep] = useState(0);
  const [processingComplete, setProcessingComplete] = useState(false);

  const handleFileChange = (e) => {
    const selectedFile = e.target.files[0];

    if (!selectedFile) return;

    // Check if file is a video
    if (!selectedFile.type.includes("video/")) {
      toast.error("Please upload a valid video file");
      return;
    }

    setFile(selectedFile);
  };

  const handleUpload = async () => {
    if (!file) {
      toast.error("Please select a file first");
      return;
    }

    setIsUploading(true);

    // Simulate upload delay
    await new Promise((resolve) => setTimeout(resolve, 2000));

    setIsUploading(false);
    toast.success("Video uploaded successfully");
  };

  const handleProcessVideo = async () => {
    if (!file) {
      toast.error("Please upload a video first");
      return;
    }

    setIsProcessing(true);
    setProcessingStep(1);

    // Simulate YOLO processing
    await new Promise((resolve) => setTimeout(resolve, 3000));
    setProcessingStep(2);

    // Simulate Deep SORT tracking
    await new Promise((resolve) => setTimeout(resolve, 3000));
    setProcessingStep(3);

    // Simulate storing movement data
    await new Promise((resolve) => setTimeout(resolve, 2000));

    setIsProcessing(false);
    setProcessingComplete(true);
    toast.success("Video processing complete");
  };

  const resetProcess = () => {
    setFile(null);
    setProcessingStep(0);
    setProcessingComplete(false);
  };

  return (
    <div className="video-processing-container">
      <h1 className="page-title">Video Processing</h1>

      <div className="upload-card">
        <h2 className="section-title">Upload CCTV Footage</h2>

        <div className="upload-area">
          <label className="upload-label">
            <div className="upload-content">
              <Upload className="upload-icon" />
              <p className="upload-text">
                <span className="upload-text-bold">Click to upload</span> or
                drag and drop
              </p>
              <p className="upload-format">MP4, AVI, MOV (MAX. 500MB)</p>
            </div>
            <input
              type="file"
              className="upload-input"
              accept="video/*"
              onChange={handleFileChange}
              disabled={isUploading || isProcessing}
            />
          </label>
        </div>

        {file && (
          <div className="selected-file">
            <div className="file-preview">
              <video
                className="file-thumbnail"
                src={URL.createObjectURL(file)}
              />
            </div>
            <div className="file-info">
              <p className="file-name">{file.name}</p>
              <p className="file-size">
                {(file.size / (1024 * 1024)).toFixed(2)} MB
              </p>
            </div>
            {!isUploading && !isProcessing && !processingComplete && (
              <button onClick={handleUpload} className="upload-button">
                Upload
              </button>
            )}
            {isUploading && (
              <div className="uploading-indicator">
                <Loader className="spinner" />
                Uploading...
              </div>
            )}
          </div>
        )}

        {file && !isUploading && !isProcessing && !processingComplete && (
          <button onClick={handleProcessVideo} className="process-button">
            <Play className="button-icon" /> Process Video
          </button>
        )}

        {isProcessing && (
          <div className="processing-section">
            <h3 className="processing-title">Processing Video</h3>

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
                  <p className="step-title">Storing Movement Data</p>
                  <p className="step-description">
                    Saving coordinates and timestamps
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
              <button
                onClick={() => (window.location.href = "/heatmap-generation")}
                className="primary-button"
              >
                Generate Heatmap
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
              <span className="guideline-highlight">Privacy:</span> The system
              does not perform facial recognition or store personally
              identifiable information.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
};

export default VideoProcessing;
