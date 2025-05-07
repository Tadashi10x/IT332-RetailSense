import { useState, useEffect, useRef } from "react";
import { Calendar, Clock, Download, Filter, Map, Loader } from "lucide-react";
import toast from "react-hot-toast";
import "./HeatmapGeneration.css";

const HeatmapGeneration = () => {
  const [isGenerating, setIsGenerating] = useState(false);
  const [heatmapGenerated, setHeatmapGenerated] = useState(false);
  const [dateRange, setDateRange] = useState({ start: "", end: "" });
  const [timeRange, setTimeRange] = useState({ start: "09:00", end: "21:00" });
  const [selectedArea, setSelectedArea] = useState("all");
  const canvasRef = useRef(null);

  useEffect(() => {
    if (heatmapGenerated) {
      renderHeatmap();
    }
  }, [heatmapGenerated]);

  const handleGenerateHeatmap = async () => {
    if (!dateRange.start || !dateRange.end) {
      toast.error("Please select a date range");
      return;
    }

    setIsGenerating(true);

    // Simulate heatmap generation delay
    await new Promise((resolve) => setTimeout(resolve, 3000));

    setIsGenerating(false);
    setHeatmapGenerated(true);
    toast.success("Heatmap generated successfully");
  };

  const renderHeatmap = () => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext("2d");
    const width = canvas.width;
    const height = canvas.height;

    // Clear canvas
    ctx.clearRect(0, 0, width, height);

    // Draw store layout (simplified)
    ctx.strokeStyle = "#333";
    ctx.lineWidth = 2;
    ctx.strokeRect(50, 50, width - 100, height - 100);

    // Draw entrance
    ctx.fillStyle = "#333";
    ctx.fillRect(width / 2 - 25, height - 50, 50, 10);

    // Draw shelves
    ctx.fillStyle = "#777";
    // Left shelves
    ctx.fillRect(100, 100, 50, 200);
    ctx.fillRect(200, 100, 50, 200);
    // Right shelves
    ctx.fillRect(width - 150, 100, 50, 200);
    ctx.fillRect(width - 250, 100, 50, 200);
    // Center display
    ctx.fillRect(width / 2 - 50, height / 2 - 50, 100, 100);

    // Draw heatmap (simulated data)
    const drawHeatPoint = (x, y, intensity) => {
      const radius = 30;
      const gradient = ctx.createRadialGradient(x, y, 0, x, y, radius);
      gradient.addColorStop(0, `rgba(255, 0, 0, ${intensity})`);
      gradient.addColorStop(1, "rgba(255, 0, 0, 0)");

      ctx.fillStyle = gradient;
      ctx.beginPath();
      ctx.arc(x, y, radius, 0, Math.PI * 2);
      ctx.fill();
    };

    // Simulate foot traffic hotspots
    // Entrance area
    drawHeatPoint(width / 2, height - 60, 0.8);
    drawHeatPoint(width / 2 - 20, height - 70, 0.7);
    drawHeatPoint(width / 2 + 20, height - 70, 0.7);

    // Main pathways
    drawHeatPoint(width / 2, height - 120, 0.6);
    drawHeatPoint(width / 2, height - 180, 0.5);
    drawHeatPoint(width / 2, height - 240, 0.4);

    // Left side
    drawHeatPoint(150, 200, 0.7);
    drawHeatPoint(250, 200, 0.5);

    // Right side
    drawHeatPoint(width - 150, 150, 0.9);
    drawHeatPoint(width - 250, 150, 0.6);
    drawHeatPoint(width - 200, 250, 0.4);

    // Center display
    drawHeatPoint(width / 2, height / 2, 0.8);
  };

  const handleExport = (format) => {
    if (!heatmapGenerated) {
      toast.error("Please generate a heatmap first");
      return;
    }

    toast.success(`Heatmap exported as ${format.toUpperCase()}`);
  };

  return (
    <div className="heatmap-container">
      <h1 className="page-title">Heatmap Generation</h1>

      <div className="heatmap-grid">
        <div className="settings-card">
          <h2 className="section-title">Heatmap Settings</h2>

          <div className="settings-form">
            <div className="form-group">
              <label className="form-label">Date Range</label>
              <div className="input-group">
                <Calendar className="input-icon" />
                <input
                  type="date"
                  className="form-input"
                  value={dateRange.start}
                  onChange={(e) =>
                    setDateRange({ ...dateRange, start: e.target.value })
                  }
                />
                <span className="input-separator">to</span>
                <input
                  type="date"
                  className="form-input"
                  value={dateRange.end}
                  onChange={(e) =>
                    setDateRange({ ...dateRange, end: e.target.value })
                  }
                />
              </div>
            </div>

            <div className="form-group">
              <label className="form-label">Time Range</label>
              <div className="input-group">
                <Clock className="input-icon" />
                <input
                  type="time"
                  className="form-input"
                  value={timeRange.start}
                  onChange={(e) =>
                    setTimeRange({ ...timeRange, start: e.target.value })
                  }
                />
                <span className="input-separator">to</span>
                <input
                  type="time"
                  className="form-input"
                  value={timeRange.end}
                  onChange={(e) =>
                    setTimeRange({ ...timeRange, end: e.target.value })
                  }
                />
              </div>
            </div>

            <div className="form-group">
              <label className="form-label">Store Area</label>
              <div className="input-group">
                <Filter className="input-icon" />
                <select
                  className="form-select"
                  value={selectedArea}
                  onChange={(e) => setSelectedArea(e.target.value)}
                >
                  <option value="all">All Areas</option>
                  <option value="entrance">Entrance</option>
                  <option value="checkout">Checkout</option>
                  <option value="aisles">Product Aisles</option>
                  <option value="displays">Center Displays</option>
                </select>
              </div>
            </div>

            <button
              onClick={handleGenerateHeatmap}
              disabled={isGenerating}
              className="generate-button"
            >
              {isGenerating ? (
                <>
                  <Loader className="spinner" /> Generating...
                </>
              ) : (
                <>
                  <Map className="button-icon" /> Generate Heatmap
                </>
              )}
            </button>

            {heatmapGenerated && (
              <div className="export-buttons">
                <button
                  onClick={() => handleExport("csv")}
                  className="export-button"
                >
                  <Download className="export-icon" /> CSV
                </button>
                <button
                  onClick={() => handleExport("pdf")}
                  className="export-button"
                >
                  <Download className="export-icon" /> PDF
                </button>
                <button
                  onClick={() => handleExport("png")}
                  className="export-button"
                >
                  <Download className="export-icon" /> PNG
                </button>
              </div>
            )}
          </div>
        </div>

        <div className="visualization-card">
          <h2 className="section-title">Heatmap Visualization</h2>

          {!heatmapGenerated ? (
            <div className="empty-heatmap">
              <Map className="empty-icon" />
              <p className="empty-text">
                Configure settings and generate a heatmap to visualize foot
                traffic
              </p>
            </div>
          ) : (
            <div className="heatmap-visualization">
              <canvas
                ref={canvasRef}
                width={800}
                height={500}
                className="heatmap-canvas"
              />

              <div className="heatmap-legend">
                <div className="legend-labels">
                  <span className="legend-title">Traffic Density:</span>
                  <div className="legend-gradient"></div>
                </div>
                <div className="legend-values">
                  <span className="legend-value">Low</span>
                  <span className="legend-value">Medium</span>
                  <span className="legend-value">High</span>
                </div>
              </div>
            </div>
          )}
        </div>
      </div>

      {heatmapGenerated && (
        <div className="analysis-card">
          <h2 className="section-title">Heatmap Analysis</h2>

          <div className="analysis-grid">
            <div className="analysis-section high-traffic">
              <h3 className="analysis-title">High Traffic Areas</h3>
              <ul className="analysis-list">
                <li>Store entrance (78% density)</li>
                <li>Right side display (65% density)</li>
                <li>Center display (58% density)</li>
              </ul>
            </div>

            <div className="analysis-section medium-traffic">
              <h3 className="analysis-title">Medium Traffic Areas</h3>
              <ul className="analysis-list">
                <li>Main pathways (45% density)</li>
                <li>Left side shelves (42% density)</li>
                <li>Checkout area (38% density)</li>
              </ul>
            </div>

            <div className="analysis-section low-traffic">
              <h3 className="analysis-title">Low Traffic Areas</h3>
              <ul className="analysis-list">
                <li>Back corner shelves (15% density)</li>
                <li>Seasonal display (12% density)</li>
                <li>Promotional area (8% density)</li>
              </ul>
            </div>
          </div>

          <div className="recommendations">
            <h3 className="recommendations-title">Recommendations</h3>
            <ul className="recommendations-list">
              <li>
                Consider moving high-margin products to high-traffic areas
              </li>
              <li>
                Redesign low-traffic areas to improve visibility and customer
                flow
              </li>
              <li>
                Adjust staffing based on peak traffic hours identified in the
                heatmap
              </li>
              <li>
                Test different promotional placements in medium-traffic zones
              </li>
            </ul>
          </div>
        </div>
      )}
    </div>
  );
};

export default HeatmapGeneration;
