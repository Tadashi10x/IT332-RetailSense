import { useState, useEffect } from "react";
import { Link } from "react-router-dom";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from "recharts";
import { Video, Map, Users, Clock } from "lucide-react";
import "./Dashboard.css";

const Dashboard = () => {
  const [stats, setStats] = useState({
    totalVisitors: 0,
    peakHour: "14:00-15:00",
    processedVideos: 0,
    generatedHeatmaps: 0,
  });

  const [trafficData, setTrafficData] = useState([]);

  useEffect(() => {
    // Mock data - in a real app, you would fetch this from your backend
    setStats({
      totalVisitors: 1248,
      peakHour: "14:00-15:00",
      processedVideos: 12,
      generatedHeatmaps: 8,
    });

    setTrafficData([
      { hour: "9AM", visitors: 45 },
      { hour: "10AM", visitors: 78 },
      { hour: "11AM", visitors: 95 },
      { hour: "12PM", visitors: 132 },
      { hour: "1PM", visitors: 156 },
      { hour: "2PM", visitors: 178 },
      { hour: "3PM", visitors: 143 },
      { hour: "4PM", visitors: 120 },
      { hour: "5PM", visitors: 98 },
      { hour: "6PM", visitors: 87 },
      { hour: "7PM", visitors: 65 },
      { hour: "8PM", visitors: 51 },
    ]);
  }, []);

  return (
    <div className="dashboard-container">
      <h1 className="dashboard-title">Dashboard</h1>

      <div className="stats-grid">
        <div className="stat-card">
          <div className="stat-content">
            <div className="stat-icon-container users-icon">
              <Users className="stat-icon" />
            </div>
            <div className="stat-info">
              <p className="stat-label">Total Visitors</p>
              <p className="stat-value">{stats.totalVisitors}</p>
            </div>
          </div>
        </div>

        <div className="stat-card">
          <div className="stat-content">
            <div className="stat-icon-container clock-icon">
              <Clock className="stat-icon" />
            </div>
            <div className="stat-info">
              <p className="stat-label">Peak Hour</p>
              <p className="stat-value">{stats.peakHour}</p>
            </div>
          </div>
        </div>

        <div className="stat-card">
          <div className="stat-content">
            <div className="stat-icon-container video-icon">
              <Video className="stat-icon" />
            </div>
            <div className="stat-info">
              <p className="stat-label">Processed Videos</p>
              <p className="stat-value">{stats.processedVideos}</p>
            </div>
          </div>
        </div>

        <div className="stat-card">
          <div className="stat-content">
            <div className="stat-icon-container map-icon">
              <Map className="stat-icon" />
            </div>
            <div className="stat-info">
              <p className="stat-label">Generated Heatmaps</p>
              <p className="stat-value">{stats.generatedHeatmaps}</p>
            </div>
          </div>
        </div>
      </div>

      <div className="dashboard-grid">
        <div className="chart-card">
          <h2 className="chart-title">Hourly Foot Traffic</h2>
          <div className="chart-container">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={trafficData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="hour" />
                <YAxis />
                <Tooltip />
                <Bar dataKey="visitors" fill="#3f51b5" />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

        <div className="actions-card">
          <h2 className="actions-title">Quick Actions</h2>
          <div className="actions-buttons">
            <Link to="/video-processing" className="action-btn video-btn">
              <Video className="action-icon" /> Process New Video
            </Link>
            <Link to="/heatmap-generation" className="action-btn heatmap-btn">
              <Map className="action-icon" /> Generate Heatmap
            </Link>
          </div>

          <div className="recent-activity">
            <h3 className="activity-title">Recent Activity</h3>
            <div className="activity-list">
              <div className="activity-item">
                <div className="activity-indicator green"></div>
                <p className="activity-text">Processed "Store Front" video</p>
                <span className="activity-time">2h ago</span>
              </div>
              <div className="activity-item">
                <div className="activity-indicator blue"></div>
                <p className="activity-text">Generated weekly heatmap</p>
                <span className="activity-time">5h ago</span>
              </div>
              <div className="activity-item">
                <div className="activity-indicator purple"></div>
                <p className="activity-text">Exported traffic report</p>
                <span className="activity-time">1d ago</span>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Dashboard;
