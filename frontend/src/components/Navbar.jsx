"use client";

import { Link, useNavigate } from "react-router-dom";
import { useState } from "react";
import { Menu, X, BarChart2, Video, Map, LogOut } from "lucide-react";
import toast from "react-hot-toast";
import "../styles/Navbar.css";

const Navbar = ({ isAuthenticated, setIsAuthenticated }) => {
  const [isOpen, setIsOpen] = useState(false);
  const navigate = useNavigate();

  const handleLogout = () => {
    setIsAuthenticated(false);
    toast.success("Logged out successfully");
    navigate("/");
  };

  return (
    <nav className="navbar">
      <div className="navbar-container">
        <div className="navbar-logo">
          <Link to="/" className="logo-text">
            RetailSense
          </Link>
        </div>
        <div className="navbar-links-desktop">
          {isAuthenticated && (
            <div className="nav-links">
              <Link to="/dashboard" className="nav-link">
                <BarChart2 className="nav-icon" /> Dashboard
              </Link>
              <Link to="/video-processing" className="nav-link">
                <Video className="nav-icon" /> Video Processing
              </Link>
              <Link to="/heatmap-generation" className="nav-link">
                <Map className="nav-icon" /> Heatmap Generation
              </Link>
            </div>
          )}
        </div>
        <div className="navbar-actions">
          {isAuthenticated && (
            <button onClick={handleLogout} className="logout-btn">
              <LogOut className="nav-icon" /> Logout
            </button>
          )}
        </div>
        <div className="navbar-mobile-toggle">
          <button
            onClick={() => setIsOpen(!isOpen)}
            className="mobile-menu-btn"
          >
            <span className="sr-only">Open main menu</span>
            {isOpen ? (
              <X className="menu-icon" />
            ) : (
              <Menu className="menu-icon" />
            )}
          </button>
        </div>
      </div>

      {isOpen && (
        <div className="navbar-mobile-menu">
          {isAuthenticated ? (
            <div className="mobile-links">
              <Link
                to="/dashboard"
                className="mobile-link"
                onClick={() => setIsOpen(false)}
              >
                <BarChart2 className="nav-icon" /> Dashboard
              </Link>
              <Link
                to="/video-processing"
                className="mobile-link"
                onClick={() => setIsOpen(false)}
              >
                <Video className="nav-icon" /> Video Processing
              </Link>
              <Link
                to="/heatmap-generation"
                className="mobile-link"
                onClick={() => setIsOpen(false)}
              >
                <Map className="nav-icon" /> Heatmap Generation
              </Link>
              <button
                onClick={() => {
                  handleLogout();
                  setIsOpen(false);
                }}
                className="mobile-logout-btn"
              >
                <LogOut className="nav-icon" /> Logout
              </button>
            </div>
          ) : (
            <div className="mobile-links">
              <Link
                to="/"
                className="mobile-link"
                onClick={() => setIsOpen(false)}
              >
                Login
              </Link>
            </div>
          )}
        </div>
      )}
    </nav>
  );
};

export default Navbar;
