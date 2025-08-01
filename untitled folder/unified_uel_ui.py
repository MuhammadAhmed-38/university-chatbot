import streamlit as st
import pandas as pd
import numpy as np
import json
import threading
import logging
import time
import random
import base64
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from unified_uel_ai_system import (
    UnifiedUELAIAgent, initialize_session_state, get_ai_agent,
    format_currency, format_date, format_duration, 
    get_status_color, get_level_color, generate_sample_data,
    config, PLOTLY_AVAILABLE
)

# Try to import plotting libraries
if PLOTLY_AVAILABLE:
    import plotly.express as px
    import plotly.graph_objects as go

# =============================================================================
# ENHANCED STREAMLIT CONFIGURATION WITH IMPROVED UEL BRANDING
# =============================================================================

def get_uel_logo_base64():
    """Get UEL logo as base64 string with fallback paths"""
    try:
        # Try multiple paths for the image
        possible_paths = [
            Path("/Users/muhammadahmed/Desktop/uel-enhanced-ai-assistant/east-london.jpg"),
            Path("east-london.jpg"),
            Path("assets/east-london.jpg"),
            Path("images/east-london.jpg"),
            Path("data/east-london.jpg")
        ]
        
        for logo_path in possible_paths:
            if logo_path.exists():
                with open(logo_path, "rb") as img_file:
                    return base64.b64encode(img_file.read()).decode()
        
        # If no image found, return empty string
        logging.info("UEL logo not found in any expected location")
        return ""
        
    except Exception as e:
        logging.warning(f"Logo loading error: {e}")
        return ""

def setup_streamlit():
    """Configure Streamlit application with enhanced UEL styling and improved visibility"""
    st.set_page_config(
        page_title="🎓 UEL Enhanced AI Assistant",
        page_icon="🎓",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Enhanced CSS styling with improved colors and input visibility
    st.markdown("""
    <style>
        /* Import Google Fonts */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=Poppins:wght@300;400;500;600;700;800&display=swap');
        
        /* Enhanced root variables for better UEL brand colors */
        :root {
            --uel-primary: #00B8A9;           /* UEL Teal */
            --uel-primary-dark: #009688;      /* Darker teal */
            --uel-primary-light: #26C6DA;     /* Lighter teal */
            --uel-secondary: #1B4B5A;         /* UEL Dark Blue */
            --uel-secondary-dark: #0D2E3A;    /* Darker blue */
            --uel-accent: #FF6B6B;            /* Coral accent */
            --uel-accent-light: #FF8A80;      /* Light coral */
            --uel-success: #4ECDC4;           /* Light teal */
            --uel-warning: #FFB74D;           /* Enhanced orange */
            --uel-error: #E57373;             /* Softer red */
            --uel-info: #64B5F6;              /* Blue info */
            
            /* Enhanced gradients */
            --uel-gradient: linear-gradient(135deg, #00B8A9 0%, #1B4B5A 100%);
            --uel-gradient-reverse: linear-gradient(135deg, #1B4B5A 0%, #00B8A9 100%);
            --uel-light-gradient: linear-gradient(135deg, #E0F7FA 0%, #F0F8FF 100%);
            --uel-card-gradient: linear-gradient(145deg, #FFFFFF 0%, #F8FFFE 100%);
            
            /* Enhanced shadows */
            --shadow-soft: 0 4px 20px rgba(0, 184, 169, 0.15);
            --shadow-medium: 0 8px 30px rgba(0, 184, 169, 0.25);
            --shadow-strong: 0 12px 40px rgba(0, 184, 169, 0.35);
            --shadow-glow: 0 0 30px rgba(0, 184, 169, 0.3);
            
            /* Text colors */
            --text-primary: #263238;
            --text-secondary: #546E7A;
            --text-light: #78909C;
            --text-white: #FFFFFF;
            
            /* Background colors */
            --bg-primary: #FAFAFA;
            --bg-secondary: #F5F5F5;
            --bg-card: #FFFFFF;
            --bg-input: #FFFFFF;
            --bg-input-focus: #F0FDFC;
        }
        
        /* Global font styling with better hierarchy */
        html, body, [class*="css"] {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            color: var(--text-primary);
            line-height: 1.6;
        }
        
        h1, h2, h3, h4, h5, h6 {
            font-family: 'Poppins', sans-serif;
            font-weight: 600;
            color: var(--text-primary);
        }
        
        /* Enhanced main app styling with better background */
        .main {
            background: var(--uel-light-gradient);
            position: relative;
            overflow-x: hidden;
            min-height: 100vh;
        }
        
        .main::before {
            content: '';
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: 
                radial-gradient(circle at 25% 25%, rgba(0, 184, 169, 0.08) 0%, transparent 50%),
                radial-gradient(circle at 75% 75%, rgba(27, 75, 90, 0.08) 0%, transparent 50%),
                radial-gradient(circle at 50% 50%, rgba(78, 205, 196, 0.06) 0%, transparent 50%),
                linear-gradient(45deg, rgba(255, 255, 255, 0.02) 0%, rgba(0, 184, 169, 0.02) 100%);
            animation: backgroundShift 30s ease-in-out infinite;
            pointer-events: none;
            z-index: -1;
        }
        
        @keyframes backgroundShift {
            0%, 100% { transform: translate(0, 0) rotate(0deg) scale(1); }
            25% { transform: translate(-10px, -15px) rotate(0.5deg) scale(1.01); }
            50% { transform: translate(15px, -10px) rotate(-0.5deg) scale(0.99); }
            75% { transform: translate(-5px, 20px) rotate(0.3deg) scale(1.01); }
        }
        
        /* Enhanced sidebar styling with better colors */
        .css-1d391kg {
            background: var(--uel-gradient);
            backdrop-filter: blur(15px);
            border-right: 3px solid rgba(255, 255, 255, 0.1);
            box-shadow: var(--shadow-medium);
        }
        
        .css-1d391kg .css-17lntkn {
            color: var(--text-white);
        }
        
        /* Enhanced logo styling with better animations */
        .uel-logo {
            width: 80px;
            height: 80px;
            border-radius: 20px;
            background: rgba(255, 255, 255, 0.95);
            padding: 12px;
            box-shadow: var(--shadow-strong);
            animation: logoFloat 4s ease-in-out infinite;
            display: flex;
            align-items: center;
            justify-content: center;
            border: 3px solid rgba(255, 255, 255, 0.3);
            backdrop-filter: blur(10px);
        }
        
        .uel-logo img {
            width: 100%;
            height: 100%;
            object-fit: cover;
            border-radius: 12px;
        }
        
        @keyframes logoFloat {
            0%, 100% { transform: translateY(0px) scale(1); }
            50% { transform: translateY(-8px) scale(1.02); }
        }
        
        /* Enhanced page headers with better visibility */
        .page-header {
            display: flex;
            align-items: center;
            padding: 2.5rem;
            background: var(--uel-card-gradient);
            border-radius: 24px;
            margin-bottom: 2rem;
            box-shadow: var(--shadow-medium);
            position: relative;
            overflow: hidden;
            animation: slideInDown 0.8s ease-out;
            border: 1px solid rgba(0, 184, 169, 0.1);
        }
        
        .page-header::before {
            content: '';
            position: absolute;
            top: 0;
            left: -100%;
            width: 100%;
            height: 100%;
            background: linear-gradient(90deg, transparent, rgba(0, 184, 169, 0.15), transparent);
            transition: left 0.6s ease;
        }
        
        .page-header:hover::before {
            left: 100%;
        }
        
        .header-icon {
            background: var(--uel-gradient);
            color: var(--text-white);
            width: 80px;
            height: 80px;
            display: flex;
            align-items: center;
            justify-content: center;
            border-radius: 24px;
            margin-right: 24px;
            font-size: 32px;
            box-shadow: var(--shadow-strong);
            animation: bounceIn 1s ease-out;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        }
        
        .header-icon:hover {
            transform: scale(1.1) rotate(5deg);
            box-shadow: var(--shadow-glow);
        }
        
        /* ENHANCED INPUT STYLING - Much more visible */
        .stTextInput > div > div > input,
        .stTextArea > div > div > textarea,
        .stSelectbox > div > div > select,
        .stNumberInput > div > div > input,
        .stDateInput > div > div > input,
        .stTimeInput > div > div > input {
            background: var(--bg-input) !important;
            border: 3px solid var(--uel-primary-light) !important;
            border-radius: 16px !important;
            padding: 16px 20px !important;
            font-size: 16px !important;
            font-weight: 500 !important;
            color: var(--text-primary) !important;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
            box-shadow: 0 4px 12px rgba(0, 184, 169, 0.1) !important;
            font-family: 'Inter', sans-serif !important;
            line-height: 1.5 !important;
        }
        
        .stTextInput > div > div > input:focus,
        .stTextArea > div > div > textarea:focus,
        .stSelectbox > div > div > select:focus,
        .stNumberInput > div > div > input:focus,
        .stDateInput > div > div > input:focus,
        .stTimeInput > div > div > input:focus {
            background: var(--bg-input-focus) !important;
            border-color: var(--uel-primary) !important;
            box-shadow: 0 0 0 4px rgba(0, 184, 169, 0.2), 0 8px 24px rgba(0, 184, 169, 0.15) !important;
            transform: translateY(-2px) !important;
            outline: none !important;
        }
        
        .stTextInput > div > div > input::placeholder,
        .stTextArea > div > div > textarea::placeholder {
            color: var(--text-light) !important;
            font-weight: 400 !important;
            opacity: 0.8 !important;
        }
        
        /* Enhanced input labels */
        .stTextInput > label,
        .stTextArea > label,
        .stSelectbox > label,
        .stNumberInput > label,
        .stDateInput > label,
        .stTimeInput > label {
            font-weight: 600 !important;
            color: var(--text-primary) !important;
            font-size: 16px !important;
            margin-bottom: 8px !important;
            font-family: 'Poppins', sans-serif !important;
        }
        
        /* Enhanced multiselect styling */
        .stMultiSelect > div > div {
            background: var(--bg-input) !important;
            border: 3px solid var(--uel-primary-light) !important;
            border-radius: 16px !important;
            padding: 8px !important;
        }
        
        .stMultiSelect > div > div:focus-within {
            border-color: var(--uel-primary) !important;
            box-shadow: 0 0 0 4px rgba(0, 184, 169, 0.2) !important;
        }
        
        /* Enhanced slider styling */
        .stSlider > div > div > div > div {
            background: var(--uel-primary) !important;
        }
        
        .stSlider > div > div > div[role="slider"] {
            background: var(--uel-primary) !important;
            border: 3px solid var(--text-white) !important;
            box-shadow: 0 4px 12px rgba(0, 184, 169, 0.3) !important;
        }
        
        /* Enhanced checkbox and radio styling */
        .stCheckbox > label > div[data-testid="stCheckbox"] > div {
            border: 3px solid var(--uel-primary-light) !important;
            border-radius: 6px !important;
        }
        
        .stCheckbox > label > div[data-testid="stCheckbox"] > div[data-checked="true"] {
            background: var(--uel-primary) !important;
            border-color: var(--uel-primary) !important;
        }
        
        /* Enhanced card styling with better colors */
        .enhanced-card {
            background: var(--uel-card-gradient);
            border-radius: 24px;
            padding: 2.5rem;
            margin: 1.5rem 0;
            box-shadow: var(--shadow-soft);
            border: 1px solid rgba(0, 184, 169, 0.1);
            transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
            position: relative;
            overflow: hidden;
            animation: fadeInUp 0.6s ease-out;
            backdrop-filter: blur(10px);
        }
        
        .enhanced-card::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 6px;
            background: var(--uel-gradient);
            transform: scaleX(0);
            transition: transform 0.4s cubic-bezier(0.4, 0, 0.2, 1);
            transform-origin: left;
        }
        
        .enhanced-card:hover {
            transform: translateY(-12px);
            box-shadow: var(--shadow-strong);
            border-color: rgba(0, 184, 169, 0.2);
        }
        
        .enhanced-card:hover::before {
            transform: scaleX(1);
        }
        
        /* Enhanced message bubbles */
        .ai-message {
            background: var(--uel-card-gradient);
            border: 2px solid rgba(0, 184, 169, 0.2);
            border-radius: 24px 24px 24px 8px;
            padding: 2rem;
            margin: 1.5rem 0;
            border-left: 6px solid var(--uel-primary);
            box-shadow: var(--shadow-soft);
            animation: messageSlideIn 0.5s ease-out;
            position: relative;
            overflow: hidden;
            backdrop-filter: blur(10px);
        }
        
        .user-message {
            background: var(--uel-gradient);
            color: var(--text-white);
            border-radius: 24px 24px 8px 24px;
            padding: 2rem;
            margin: 1.5rem 0;
            text-align: right;
            box-shadow: var(--shadow-medium);
            animation: messageSlideInRight 0.5s ease-out;
            backdrop-filter: blur(10px);
        }
        
        /* Enhanced metric cards */
        .metric-card {
            background: var(--uel-card-gradient);
            padding: 2.5rem;
            border-radius: 20px;
            box-shadow: var(--shadow-soft);
            text-align: center;
            margin: 1rem 0;
            transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
            border: 2px solid transparent;
            position: relative;
            overflow: hidden;
            backdrop-filter: blur(10px);
        }
        
        .metric-card:hover {
            border-color: var(--uel-primary);
            transform: translateY(-8px) scale(1.02);
            box-shadow: var(--shadow-strong);
        }
        
        /* Enhanced feature cards */
        .feature-card {
            background: var(--uel-gradient);
            color: var(--text-white);
            padding: 2.5rem;
            border-radius: 24px;
            margin: 1.5rem 0;
            box-shadow: var(--shadow-medium);
            transition: all 0.5s cubic-bezier(0.4, 0, 0.2, 1);
            position: relative;
            overflow: hidden;
            transform-style: preserve-3d;
            backdrop-filter: blur(10px);
        }
        
        .feature-card:hover {
            transform: translateY(-12px) rotateX(2deg) scale(1.02);
            box-shadow: var(--shadow-glow);
        }
        
        /* Enhanced buttons with better visibility */
        .stButton > button {
            border-radius: 30px !important;
            background: var(--uel-gradient) !important;
            color: var(--text-white) !important;
            border: none !important;
            padding: 1rem 2.5rem !important;
            font-weight: 600 !important;
            font-size: 16px !important;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
            position: relative !important;
            overflow: hidden !important;
            box-shadow: var(--shadow-soft) !important;
            font-family: 'Poppins', sans-serif !important;
            letter-spacing: 0.5px !important;
            text-transform: none !important;
        }
        
        .stButton > button:hover {
            transform: translateY(-3px) scale(1.02) !important;
            box-shadow: var(--shadow-strong) !important;
        }
        
        .stButton > button:active {
            transform: translateY(-1px) scale(0.98) !important;
        }
        
        /* Enhanced navigation items */
        .nav-item {
            padding: 1.2rem 1.8rem;
            border-radius: 16px;
            margin-bottom: 0.8rem;
            cursor: pointer;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            position: relative;
            overflow: hidden;
            color: rgba(255, 255, 255, 0.9);
            font-weight: 500;
            backdrop-filter: blur(10px);
        }
        
        .nav-item:hover {
            background: rgba(255, 255, 255, 0.15);
            color: var(--text-white);
            transform: translateX(8px) scale(1.02);
            box-shadow: 0 4px 20px rgba(255, 255, 255, 0.1);
        }
        
        .nav-item.active {
            background: rgba(255, 255, 255, 0.2);
            color: var(--text-white);
            border-left: 5px solid var(--text-white);
            font-weight: 600;
        }
        
        /* Enhanced prediction card */
        .prediction-card {
            background: var(--uel-gradient);
            color: var(--text-white);
            padding: 4rem;
            border-radius: 32px;
            text-align: center;
            margin: 3rem 0;
            box-shadow: var(--shadow-strong);
            position: relative;
            overflow: hidden;
            animation: glowPulse 4s ease-in-out infinite alternate;
            backdrop-filter: blur(15px);
        }
        
        @keyframes glowPulse {
            from {
                box-shadow: var(--shadow-strong);
            }
            to {
                box-shadow: var(--shadow-glow), 0 0 50px rgba(0, 184, 169, 0.4);
            }
        }
        
        /* Enhanced charts container */
        .chart-container {
            background: var(--uel-card-gradient);
            padding: 2.5rem;
            border-radius: 24px;
            box-shadow: var(--shadow-soft);
            margin: 1.5rem 0;
            border: 1px solid rgba(0, 184, 169, 0.1);
            transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
            animation: fadeInScale 0.8s ease-out;
            backdrop-filter: blur(10px);
        }
        
        .chart-container:hover {
            transform: scale(1.02);
            box-shadow: var(--shadow-medium);
        }
        
        /* Enhanced status badges */
        .status-badge {
            display: inline-block;
            padding: 0.8rem 1.5rem;
            border-radius: 25px;
            font-size: 14px;
            font-weight: 600;
            text-align: center;
            text-transform: uppercase;
            letter-spacing: 0.8px;
            animation: badgeSlideIn 0.5s ease-out;
            position: relative;
            overflow: hidden;
            backdrop-filter: blur(10px);
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
        }
        
        /* Enhanced scrollbar */
        ::-webkit-scrollbar {
            width: 12px;
        }
        
        ::-webkit-scrollbar-track {
            background: rgba(0, 184, 169, 0.1);
            border-radius: 6px;
        }
        
        ::-webkit-scrollbar-thumb {
            background: var(--uel-gradient);
            border-radius: 6px;
            border: 2px solid transparent;
            background-clip: content-box;
        }
        
        ::-webkit-scrollbar-thumb:hover {
            background: var(--uel-gradient-reverse);
            background-clip: content-box;
        }
        
        /* Enhanced form styling */
        .stForm {
            background: var(--uel-card-gradient) !important;
            border-radius: 24px !important;
            padding: 2rem !important;
            box-shadow: var(--shadow-soft) !important;
            border: 1px solid rgba(0, 184, 169, 0.1) !important;
        }
        
        /* Enhanced expander styling */
        .streamlit-expanderHeader {
            background: var(--uel-card-gradient) !important;
            border: 2px solid rgba(0, 184, 169, 0.2) !important;
            border-radius: 16px !important;
            padding: 1rem !important;
            font-weight: 600 !important;
            color: var(--text-primary) !important;
        }
        
        .streamlit-expanderContent {
            background: var(--uel-card-gradient) !important;
            border: 2px solid rgba(0, 184, 169, 0.1) !important;
            border-top: none !important;
            border-radius: 0 0 16px 16px !important;
            padding: 1.5rem !important;
        }
        
        /* Enhanced tabs styling */
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
            background: var(--uel-card-gradient);
            padding: 8px;
            border-radius: 16px;
            box-shadow: var(--shadow-soft);
        }
        
        .stTabs [data-baseweb="tab"] {
            background: transparent;
            border-radius: 12px;
            color: var(--text-secondary);
            font-weight: 500;
            padding: 12px 24px;
            transition: all 0.3s ease;
        }
        
        .stTabs [data-baseweb="tab"]:hover {
            background: rgba(0, 184, 169, 0.1);
            color: var(--uel-primary);
        }
        
        .stTabs [aria-selected="true"] {
            background: var(--uel-primary) !important;
            color: var(--text-white) !important;
            font-weight: 600 !important;
        }
        
        /* Enhanced success/error/warning messages */
        .stSuccess {
            background: linear-gradient(135deg, rgba(76, 205, 196, 0.1), rgba(76, 205, 196, 0.05)) !important;
            border: 2px solid var(--uel-success) !important;
            border-radius: 16px !important;
            padding: 1rem 1.5rem !important;
            color: var(--text-primary) !important;
        }
        
        .stError {
            background: linear-gradient(135deg, rgba(229, 115, 115, 0.1), rgba(229, 115, 115, 0.05)) !important;
            border: 2px solid var(--uel-error) !important;
            border-radius: 16px !important;
            padding: 1rem 1.5rem !important;
            color: var(--text-primary) !important;
        }
        
        .stWarning {
            background: linear-gradient(135deg, rgba(255, 183, 77, 0.1), rgba(255, 183, 77, 0.05)) !important;
            border: 2px solid var(--uel-warning) !important;
            border-radius: 16px !important;
            padding: 1rem 1.5rem !important;
            color: var(--text-primary) !important;
        }
        
        .stInfo {
            background: linear-gradient(135deg, rgba(100, 181, 246, 0.1), rgba(100, 181, 246, 0.05)) !important;
            border: 2px solid var(--uel-info) !important;
            border-radius: 16px !important;
            padding: 1rem 1.5rem !important;
            color: var(--text-primary) !important;
        }
        
        /* Responsive design enhancements */
        @media (max-width: 768px) {
            .page-header {
                padding: 1.5rem;
                flex-direction: column;
                text-align: center;
            }
            
            .header-icon {
                margin-right: 0;
                margin-bottom: 1rem;
                width: 60px;
                height: 60px;
                font-size: 24px;
            }
            
            .feature-card,
            .enhanced-card {
                padding: 2rem;
                margin: 1rem 0;
            }
            
            .stTextInput > div > div > input,
            .stTextArea > div > div > textarea {
                font-size: 16px !important;
            }
        }
        
        /* Loading animations */
        .loading-spinner {
            display: inline-block;
            width: 32px;
            height: 32px;
            border: 4px solid rgba(0, 184, 169, 0.3);
            border-radius: 50%;
            border-top-color: var(--uel-primary);
            animation: spin 1s ease-in-out infinite;
        }
        
        @keyframes spin {
            to { transform: rotate(360deg); }
        }
        
        /* Enhanced animations */
        @keyframes slideInDown {
            from {
                opacity: 0;
                transform: translateY(-40px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }
        
        @keyframes fadeInUp {
            from {
                opacity: 0;
                transform: translateY(40px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }
        
        @keyframes bounceIn {
            0% {
                opacity: 0;
                transform: scale(0.3);
            }
            50% {
                opacity: 1;
                transform: scale(1.08);
            }
            70% {
                transform: scale(0.95);
            }
            100% {
                opacity: 1;
                transform: scale(1);
            }
        }
        
        @keyframes messageSlideIn {
            from {
                opacity: 0;
                transform: translateX(-30px);
            }
            to {
                opacity: 1;
                transform: translateX(0);
            }
        }
        
        @keyframes messageSlideInRight {
            from {
                opacity: 0;
                transform: translateX(30px);
            }
            to {
                opacity: 1;
                transform: translateX(0);
            }
        }
        
        @keyframes fadeInScale {
            from {
                opacity: 0;
                transform: scale(0.95);
            }
            to {
                opacity: 1;
                transform: scale(1);
            }
        }
        
        @keyframes badgeSlideIn {
            from {
                opacity: 0;
                transform: translateX(-15px);
            }
            to {
                opacity: 1;
                transform: translateX(0);
            }
        }
        
        /* Success and error animations */
        .success-animation {
            animation: successPulse 0.6s ease-out;
        }
        
        @keyframes successPulse {
            0% { transform: scale(1); }
            50% { transform: scale(1.05); }
            100% { transform: scale(1); }
        }
        
        .error-animation {
            animation: errorShake 0.5s ease-in-out;
        }
        
        @keyframes errorShake {
            0%, 100% { transform: translateX(0); }
            25% { transform: translateX(-8px); }
            75% { transform: translateX(8px); }
        }
    </style>
    """, unsafe_allow_html=True)


# =============================================================================
# CHAT MESSAGE MANAGEMENT (keeping original functionality)
# =============================================================================

class ChatMessageManager:
    """Enhanced chat message management with memory limits and persistence"""
    
    def __init__(self, max_messages=100, context_window=10):
        self.max_messages = max_messages
        self.context_window = context_window
        self.logger = logging.getLogger(__name__)
    
    def initialize_chat(self):
        """Initialize chat with welcome message and memory management"""
        if 'messages' not in st.session_state:
            st.session_state.messages = []
        
        if 'message_count' not in st.session_state:
            st.session_state.message_count = 0
        
        # Add welcome message if chat is empty
        if not st.session_state.messages:
            welcome_msg = self._create_welcome_message()
            self._add_message(welcome_msg)
    
    def _create_welcome_message(self):
        """Create personalized welcome message"""
        user_name = ""
        if st.session_state.get('current_student'):
            user_name = st.session_state.current_student.get('first_name', '')
        
        welcome_text = f"""👋 {'Hello ' + user_name + '! ' if user_name else 'Welcome to '}**University of East London AI Assistant**!

I'm your intelligent AI helper, powered by advanced language models. I can assist you with:

🎓 **Course Information** - Programs, requirements, fees  
📝 **Applications** - Guidance and process help  
🔮 **Predictions** - AI admission forecasting  
🎯 **Recommendations** - Personalized course matching  
📄 **Documents** - Verification and requirements  
💰 **Financial Aid** - Scholarships and funding  

**Quick Tips:**
• Use the quick action buttons below for common questions
• Set up your profile for personalized responses
• Try voice mode for hands-free interaction

How can I help you today?"""

        return {
            "id": self._generate_message_id(),
            "sender": "ai",
            "message": welcome_text,
            "timestamp": datetime.now().isoformat(),
            "type": "welcome",
            "metadata": {
                "system_message": True,
                "personalized": bool(user_name)
            }
        }
    
    def add_user_message(self, content, message_type="text", metadata=None):
        """Add user message with validation"""
        try:
            # Validate content
            if not content or not content.strip():
                raise ValueError("Message content cannot be empty")
            
            if len(content) > 5000:  # Reasonable limit
                content = content[:5000] + "... [message truncated]"
            
            message = {
                "id": self._generate_message_id(),
                "sender": "user", 
                "message": content.strip(),
                "timestamp": datetime.now().isoformat(),
                "type": message_type,
                "metadata": metadata or {}
            }
            
            self._add_message(message)
            return message
            
        except Exception as e:
            self.logger.error(f"Error adding user message: {e}")
            raise
    
    def add_ai_message(self, content, message_type="text", metadata=None):
        """Add AI message with enhanced formatting"""
        try:
            if not content:
                content = "I apologize, but I couldn't generate a response. Please try again."
            
            message = {
                "id": self._generate_message_id(),
                "sender": "ai",
                "message": content,
                "timestamp": datetime.now().isoformat(), 
                "type": message_type,
                "metadata": metadata or {}
            }
            
            self._add_message(message)
            return message
            
        except Exception as e:
            self.logger.error(f"Error adding AI message: {e}")
            # Add error message instead
            error_message = {
                "id": self._generate_message_id(),
                "sender": "ai",
                "message": "⚠️ I encountered an error processing your request. Please try again.",
                "timestamp": datetime.now().isoformat(),
                "type": "error",
                "metadata": {"error": str(e)}
            }
            self._add_message(error_message)
            return error_message
    
    def _add_message(self, message):
        """Internal method to add message with memory management"""
        st.session_state.messages.append(message)
        st.session_state.message_count += 1
        
        # Memory management - keep only recent messages
        if len(st.session_state.messages) > self.max_messages:
            # Keep welcome message + recent messages
            welcome_msgs = [msg for msg in st.session_state.messages if msg.get('type') == 'welcome']
            recent_msgs = st.session_state.messages[-(self.max_messages-1):]
            st.session_state.messages = welcome_msgs + recent_msgs
            self.logger.info(f"Trimmed messages to {len(st.session_state.messages)}")
    
    def clear_chat(self):
        """Clear chat history but keep welcome message"""
        try:
            welcome_msg = self._create_welcome_message()
            st.session_state.messages = [welcome_msg]
            st.session_state.message_count = 1
            self.logger.info("Chat cleared successfully")
            return True
        except Exception as e:
            self.logger.error(f"Error clearing chat: {e}")
            return False
    
    def _generate_message_id(self):
        """Generate unique message ID"""
        return f"msg_{int(time.time() * 1000)}_{random.randint(1000, 9999)}"
    
    def get_chat_statistics(self):
        """Get chat statistics"""
        if not st.session_state.messages:
            return {"total": 0, "user": 0, "ai": 0, "avg_length": 0}
        
        total = len(st.session_state.messages)
        user_msgs = len([m for m in st.session_state.messages if m['sender'] == 'user'])
        ai_msgs = len([m for m in st.session_state.messages if m['sender'] == 'ai'])
        
        all_lengths = [len(m['message']) for m in st.session_state.messages]
        avg_length = sum(all_lengths) / len(all_lengths) if all_lengths else 0
        
        return {
            "total": total,
            "user": user_msgs, 
            "ai": ai_msgs,
            "avg_length": round(avg_length, 1)
        }
    
    def export_chat_history(self):
        """Export chat history for download"""
        try:
            chat_data = {
                "export_timestamp": datetime.now().isoformat(),
                "total_messages": len(st.session_state.messages),
                "messages": st.session_state.messages,
                "user_profile": st.session_state.get('current_student', {}),
                "session_info": {
                    "session_id": st.session_state.get('user_id'),
                    "message_count": st.session_state.message_count
                }
            }
            
            return json.dumps(chat_data, indent=2)
            
        except Exception as e:
            self.logger.error(f"Error exporting chat: {e}")
            return None


class ChatErrorHandler:
    """Comprehensive error handling for chat functionality"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.error_count = 0
        self.last_error_time = None
        self.consecutive_errors = 0
        
    def handle_user_message_safely(self, user_input, ai_agent, message_manager, voice_response=False):
        """Safely handle user message with comprehensive error handling"""
        try:
            # Input validation
            if not user_input or not user_input.strip():
                st.warning("⚠️ Please enter a message before sending.")
                return False
            
            # Rate limiting check
            if self._is_rate_limited():
                st.error("⏳ Please wait a moment between messages.")
                return False
            
            # Add user message
            user_message = message_manager.add_user_message(user_input)
            
            # Show processing indicator
            with st.spinner("🤖 AI is thinking..."):
                try:
                    # Get AI response with timeout
                    response_data = self._get_ai_response_with_timeout(
                        ai_agent, user_input, timeout=30
                    )
                    
                    if response_data and response_data.get('response'):
                        # Add AI message
                        ai_message = message_manager.add_ai_message(
                            response_data['response'],
                            metadata={
                                "sentiment": response_data.get('sentiment'),
                                "llm_used": response_data.get('llm_used'),
                                "response_time": response_data.get('response_time'),
                                "voice_response": voice_response
                            }
                        )
                        
                        # Handle voice response if requested
                        if voice_response:
                            self._handle_voice_response(ai_agent, response_data['response'])
                        
                        # Store interaction
                        self._store_interaction(user_input, response_data)
                        
                        # Reset error count on success
                        self.consecutive_errors = 0
                        return True
                        
                    else:
                        raise Exception("Empty or invalid AI response")
                        
                except Exception as ai_error:
                    self.logger.error(f"AI response error: {ai_error}")
                    self._handle_ai_error(ai_error, message_manager)
                    return False
                    
        except Exception as e:
            self.logger.error(f"User message handling error: {e}")
            self._handle_general_error(e, message_manager)
            return False
    
    def handle_voice_conversation_safely(self, ai_agent, message_manager, voice_response=True):
        """Safely handle voice conversation with error recovery"""
        try:
            # Check voice service availability
            if not self._check_voice_service(ai_agent):
                return False
            
            with st.spinner("🎤 Listening... Please speak clearly"):
                try:
                    # Get speech input
                    speech_result = ai_agent.speech_to_text()
                    
                    if speech_result and not speech_result.startswith(('❌', '⏰', '⚠️')):
                        st.success(f"🎤 Heard: {speech_result}")
                        
                        # Process as regular message
                        return self.handle_user_message_safely(
                            speech_result, ai_agent, message_manager, voice_response
                        )
                    else:
                        # Handle speech recognition errors
                        self._handle_speech_error(speech_result, message_manager)
                        return False
                        
                except Exception as voice_error:
                    self.logger.error(f"Voice processing error: {voice_error}")
                    self._handle_voice_error(voice_error, message_manager)
                    return False
                    
        except Exception as e:
            self.logger.error(f"Voice conversation error: {e}")
            self._handle_general_error(e, message_manager)
            return False
    
    def _get_ai_response_with_timeout(self, ai_agent, user_input, timeout=30):
        """Get AI response with timeout protection"""
        try:
            # Prepare user context
            user_context = self._prepare_user_context()
            response_data = ai_agent.get_ai_response(user_input, user_context)
            return response_data
        except Exception as e:
            raise e
    
    def _handle_voice_response(self, ai_agent, response_text):
        """Handle voice response with error recovery"""
        try:
            with st.spinner("🔊 AI is speaking..."):
                success = ai_agent.text_to_speech(response_text)
                if success:
                    st.success("🔊 AI response played!")
                else:
                    st.warning("🔊 Voice response not available - showing text only")
        except Exception as e:
            self.logger.error(f"Voice response error: {e}")
            st.warning("🔊 Voice playback failed - showing text only")
    
    def _handle_ai_error(self, error, message_manager):
        """Handle AI-specific errors"""
        self.consecutive_errors += 1
        
        if "timeout" in str(error).lower():
            error_msg = "⏰ The AI is taking longer than usual. Please try a simpler question or try again."
        elif "connection" in str(error).lower():
            error_msg = "🔌 Connection issue with AI service. Please check your connection and try again."
        elif "model" in str(error).lower():
            error_msg = "🤖 AI model is temporarily unavailable. Please try again in a moment."
        else:
            error_msg = "🤔 I encountered an unexpected issue. Please rephrase your question and try again."
        
        # Add helpful suggestions for consecutive errors
        if self.consecutive_errors >= 3:
            error_msg += "\n\n💡 **Troubleshooting Tips:**\n• Try asking a simpler question\n• Check your internet connection\n• Refresh the page if issues persist"
        
        message_manager.add_ai_message(error_msg, message_type="error")
    
    def _handle_speech_error(self, error_result, message_manager):
        """Handle speech recognition errors"""
        if "No speech detected" in error_result:
            st.info("🎤 No speech detected. Please speak closer to the microphone.")
        elif "Could not understand" in error_result:
            st.warning("🎤 Couldn't understand the speech. Please speak more clearly.")
        elif "No microphone" in error_result:
            st.error("🎤 No microphone detected. Please check your microphone settings.")
        else:
            st.error(f"🎤 Speech recognition issue: {error_result}")
    
    def _handle_voice_error(self, error, message_manager):
        """Handle voice processing errors"""
        error_msg = "🎤 Voice feature is temporarily unavailable. Please type your message instead."
        
        if "microphone" in str(error).lower():
            error_msg += "\n\n💡 **Microphone Tips:**\n• Check microphone permissions\n• Ensure microphone is connected\n• Try refreshing the page"
        
        message_manager.add_ai_message(error_msg, message_type="error")
    
    def _handle_general_error(self, error, message_manager):
        """Handle general errors"""
        self.error_count += 1
        self.last_error_time = datetime.now()
        
        error_msg = "⚠️ Something unexpected happened. Please try again."
        
        # Add context-specific help
        if self.error_count >= 5:
            error_msg += "\n\n🔧 **If problems persist:**\n• Refresh the page\n• Clear your browser cache\n• Contact support if needed"
        
        message_manager.add_ai_message(error_msg, message_type="error")
    
    def _check_voice_service(self, ai_agent):
        """Check if voice service is available"""
        try:
            if not ai_agent.voice_service.is_available():
                st.error("🎤 Voice service not available. Please install required packages:")
                st.code("pip install SpeechRecognition pyttsx3 pyaudio")
                return False
            return True
        except Exception as e:
            self.logger.error(f"Voice service check error: {e}")
            return False
    
    def _is_rate_limited(self):
        """Check if user is rate limited (simple implementation)"""
        if not hasattr(self, '_last_message_time'):
            self._last_message_time = datetime.now()
            return False
        
        time_diff = (datetime.now() - self._last_message_time).total_seconds()
        self._last_message_time = datetime.now()
        
        return time_diff < 1  # Minimum 1 second between messages
    
    def _prepare_user_context(self):
        """Prepare user context safely"""
        try:
            user_context = {}
            
            # Add current student/profile data safely
            if st.session_state.get('current_student'):
                user_context.update(st.session_state.current_student)
            elif st.session_state.get('user_profile'):
                user_context.update(st.session_state.user_profile)
            
            # Add conversation context safely
            if st.session_state.get('messages'):
                user_context['conversation_history'] = st.session_state.messages[-5:]
            
            return user_context
            
        except Exception as e:
            self.logger.error(f"Error preparing user context: {e}")
            return {}
    
    def _store_interaction(self, user_input, response_data):
        """Store interaction safely"""
        try:
            if 'interaction_history' not in st.session_state:
                st.session_state.interaction_history = []
            
            interaction = {
                "user_input": user_input,
                "ai_response": response_data.get('response', ''),
                "timestamp": datetime.now().isoformat(),
                "sentiment": response_data.get('sentiment'),
                "llm_used": response_data.get('llm_used'),
                "response_time": response_data.get('response_time')
            }
            
            st.session_state.interaction_history.append(interaction)
            
            # Keep only recent interactions to manage memory
            if len(st.session_state.interaction_history) > 50:
                st.session_state.interaction_history = st.session_state.interaction_history[-50:]
                
        except Exception as e:
            self.logger.error(f"Error storing interaction: {e}")
    
    def get_error_stats(self):
        """Get error statistics"""
        return {
            "total_errors": self.error_count,
            "consecutive_errors": self.consecutive_errors,
            "last_error_time": self.last_error_time.isoformat() if self.last_error_time else None
        }


# =============================================================================
# ENHANCED NAVIGATION COMPONENTS
# =============================================================================

def render_sidebar():
    """Enhanced sidebar with UEL branding and animations"""
    with st.sidebar:
        # Get UEL logo
        logo_base64 = get_uel_logo_base64()
        
        # Enhanced UEL logo and branding
        if logo_base64:
            logo_html = f'<img src="data:image/jpeg;base64,{logo_base64}" style="width: 100%; height: 100%; object-fit: cover; border-radius: 12px;">'
        else:
            logo_html = '<div style="font-size: 2.5rem; color: var(--uel-primary);">🎓</div>'
        
        st.markdown(f"""
        <div style="text-align: center; padding: 1.5rem 0; margin-bottom: 2rem;">
            <div class="uel-logo" style="margin: 0 auto 1.5rem auto;">
                {logo_html}
            </div>
            <h2 style="color: white; margin: 0; font-weight: 700; font-size: 1.8rem; text-shadow: 0 2px 4px rgba(0,0,0,0.3);">UEL AI Assistant</h2>
            <p style="color: rgba(255,255,255,0.9); margin: 0.8rem 0 0 0; font-size: 1rem; font-weight: 500;">Enhanced Intelligence System</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown('<div style="border-top: 2px solid rgba(255,255,255,0.2); margin: 1.5rem 0;"></div>', unsafe_allow_html=True)
        
        # Enhanced navigation with animations
        st.markdown("#### 📋 Navigation")
        pages = {
            "🏠 Dashboard": "dashboard",
            "💬 AI Chat": "ai_chat",
            "👥 Students": "students",
            "📚 Courses": "courses",
            "📄 Applications": "applications",
            "🎯 Recommendations": "recommendations",
            "🔮 Predictions": "predictions",
            "📊 Analytics": "analytics"
        }
        
        for page_name, page_key in pages.items():
            if st.button(page_name, key=f"nav_{page_key}", use_container_width=True):
                st.session_state.current_page = page_key
                st.rerun()
        
        st.markdown("---")
        
        # System Status with enhanced styling
        st.markdown("#### ⚙️ System Status")
        
        try:
            # Safely get AI agent and feature status
            ai_agent = get_ai_agent()
            feature_status = ai_agent.get_feature_status()
            
            # Safe feature access with fallbacks
            features = [
                ("🤖 LLM Integration", "✅" if feature_status.get('llm_integration', False) else "❌"),
                ("🧠 ML Models", "✅" if feature_status.get('ml_predictions', False) else "❌"),
                ("😊 Sentiment Analysis", "✅" if feature_status.get('sentiment_analysis', False) else "❌"),
                ("🎯 Course Recommendations", "✅" if feature_status.get('course_recommendations', True) else "❌"),
                ("📄 Document Verification", "✅" if feature_status.get('document_verification', True) else "❌"),
                ("🎤 Voice Services", "✅" if feature_status.get('voice_services', False) else "❌"),
                ("🔍 Intelligent Search", "✅" if feature_status.get('intelligent_search', False) else "❌"),
                ("📊 Real-time Analytics", "✅" if feature_status.get('real_time_analytics', True) else "❌")
            ]
            
            for feature_name, status in features:
                color = "#4ECDC4" if status == "✅" else "#FF8A80"
                st.markdown(f"""
                <div style="display: flex; justify-content: space-between; align-items: center; padding: 0.5rem; margin: 0.3rem 0; background: rgba(255,255,255,0.1); border-radius: 8px;">
                    <span style="color: rgba(255,255,255,0.9); font-size: 0.9rem;">{feature_name}</span>
                    <span style="color: {color}; font-size: 1.1rem;">{status}</span>
                </div>
                """, unsafe_allow_html=True)
                
        except Exception as e:
            st.error(f"Error loading system status: {e}")
            # Show minimal status if there's an error
            st.markdown("🤖 LLM Integration: ❌")
            st.markdown("🧠 ML Models: ❌")
            st.markdown("😊 Sentiment Analysis: ❌")
            st.markdown("🎯 Course Recommendations: ✅")
            st.markdown("📄 Document Verification: ✅")
            st.markdown("🎤 Voice Services: ❌")
            st.markdown("🔍 Intelligent Search: ❌")
            st.markdown("📊 Real-time Analytics: ✅")
        
        st.markdown("---")
        
        # Quick Actions
        st.markdown("#### 🚀 Quick Actions")
        if st.button("🔄 Refresh System", use_container_width=True):
            st.session_state.ai_agent = None  # Force re-initialization
            st.rerun()
        
        if st.button("📋 Export Data", use_container_width=True):
            st.info("Export functionality coming soon!")
        
        if st.button("⚙️ Settings", use_container_width=True):
            st.session_state.current_page = "settings"
            st.rerun()
        
        st.markdown("---")
        
        # University Info with enhanced styling
        st.markdown("#### 🏫 Contact UEL")
        st.markdown("""
        <div style="background: rgba(255,255,255,0.15); padding: 1rem; border-radius: 12px; backdrop-filter: blur(10px);">
            <div style="color: rgba(255,255,255,0.9); margin-bottom: 0.5rem;">
                <strong>📧 Email:</strong><br>
                <span style="font-size: 0.9rem;">admissions@uel.ac.uk</span>
            </div>
            <div style="color: rgba(255,255,255,0.9); margin-bottom: 0.5rem;">
                <strong>📞 Phone:</strong><br>
                <span style="font-size: 0.9rem;">+44 20 8223 3000</span>
            </div>
            <div style="color: rgba(255,255,255,0.9);">
                <strong>🌐 Website:</strong><br>
                <a href="https://uel.ac.uk" target="_blank" style="color: #26C6DA; text-decoration: none; font-size: 0.9rem;">uel.ac.uk</a>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Session Info
        if st.checkbox("Show Debug Info"):
            st.markdown("#### 🔧 Debug Info")
            st.json({
                "Session ID": st.session_state.get('user_id', 'Unknown'),
                "Current Page": st.session_state.get('current_page', 'Unknown'),
                "Messages": len(st.session_state.get('messages', [])),
                "Session Start": str(st.session_state.get('session_start_time', 'Unknown'))
            })

# =============================================================================
# ENHANCED DASHBOARD PAGE
# =============================================================================

def render_dashboard():
    """Render enhanced dashboard with animations and UEL branding"""
    # Get UEL logo
    logo_base64 = get_uel_logo_base64()
    
    # Enhanced header with logo and animations
    if logo_base64:
        logo_html = f'<img src="data:image/jpeg;base64,{logo_base64}" style="width: 100%; height: 100%; object-fit: cover; border-radius: 15px;">'
    else:
        logo_html = '<div style="font-size: 2.5rem; color: var(--uel-primary);">🎓</div>'
    
    st.markdown(f"""
    <div class="page-header">
        <div class="header-icon">🏠</div>
        <div style="flex-grow: 1;">
            <h1 style="margin: 0; background: var(--uel-gradient); -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-weight: 800; font-size: 2.5rem;">
                UEL AI Dashboard
            </h1>
            <p style="margin: 0; color: var(--text-secondary); font-size: 1.2rem; font-weight: 500;">Enhanced University Intelligence System</p>
        </div>
        <div class="uel-logo floating-element">
            {logo_html}
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    ai_agent = get_ai_agent()
    
    # Enhanced welcome section with animations
    st.markdown("""
    <div class="enhanced-card" style="text-align: center; background: var(--uel-gradient); color: white; border-radius: 32px;">
        <h2 style="margin-top: 0; font-size: 3rem; font-weight: 800; text-shadow: 0 4px 8px rgba(0,0,0,0.2);">🌟 Welcome to UEL AI Assistant</h2>
        <p style="font-size: 1.3rem; opacity: 0.95; margin: 1.5rem 0; font-weight: 500;">Your intelligent companion for university admissions and academic guidance</p>
        <div style="margin-top: 2.5rem; display: flex; justify-content: center; flex-wrap: wrap; gap: 1rem;">
            <div style="display: inline-flex; align-items: center; padding: 1rem 2rem; background: rgba(255,255,255,0.25); border-radius: 30px; backdrop-filter: blur(10px); font-weight: 600;">
                🤖 AI-Powered
            </div>
            <div style="display: inline-flex; align-items: center; padding: 1rem 2rem; background: rgba(255,255,255,0.25); border-radius: 30px; backdrop-filter: blur(10px); font-weight: 600;">
                🎯 Personalized
            </div>
            <div style="display: inline-flex; align-items: center; padding: 1rem 2rem; background: rgba(255,255,255,0.25); border-radius: 30px; backdrop-filter: blur(10px); font-weight: 600;">
                ⚡ Real-time
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # System overview metrics with enhanced animations
    st.markdown("## 🚀 System Overview")
    
    analytics = ai_agent.get_system_analytics()
    feature_status = ai_agent.get_feature_status()
    
    col1, col2, col3, col4 = st.columns(4)
    
    metrics_data = [
        ("🤖 AI Engine", 'Online' if feature_status['llm_integration'] else 'Offline', "#00B8A9"),
        ("🧠 ML Models", 'Trained' if feature_status['ml_predictions'] else 'Training', "#4ECDC4"),
        ("📊 Analytics", f"{analytics['statistics']['total_interactions']}", "#1B4B5A"),
        ("⚡ Performance", f"{analytics['statistics']['error_rate_percent']:.1f}%", "#FF6B6B")
    ]
    
    for i, (col, (title, value, color)) in enumerate(zip([col1, col2, col3, col4], metrics_data)):
        with col:
            st.markdown(f"""
            <div class="metric-card" style="border-left: 6px solid {color}; animation-delay: {i*0.15}s; background: var(--uel-card-gradient);">
                <h3 style="color: {color}; margin: 0; font-size: 1.1rem; font-weight: 700;">{title}</h3>
                <p style="font-size: 2.2rem; font-weight: 800; margin: 1rem 0; color: var(--text-primary);">
                    {value}
                </p>
                <div style="width: 100%; height: 6px; background: rgba(0, 184, 169, 0.1); border-radius: 3px; overflow: hidden; margin-top: 1.5rem;">
                    <div style="height: 100%; background: {color}; width: 85%; border-radius: 3px; animation: progressLoad 2.5s ease-out;"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    # Enhanced feature showcase with 3D cards
    st.markdown("## 🧠 AI Capabilities")
    
    col1, col2, col3 = st.columns(3)
    
    features = [
        {
            "title": "🤖 Advanced LLM Integration",
            "description": "Powered by DeepSeek for intelligent conversations",
            "features": [
                "Natural language understanding",
                "Context-aware responses", 
                "Local deployment for privacy",
                "Multi-language support"
            ],
            "gradient": "linear-gradient(145deg, #667eea 0%, #764ba2 100%)"
        },
        {
            "title": "🔮 Predictive Analytics", 
            "description": "ML models for admission success prediction",
            "features": [
                "Random Forest & Gradient Boosting",
                "87% prediction accuracy",
                "Feature importance analysis", 
                "Personalized recommendations"
            ],
            "gradient": "linear-gradient(145deg, #f093fb 0%, #f5576c 100%)"
        },
        {
            "title": "📊 Real-time Analytics",
            "description": "Comprehensive sentiment and behavior analysis", 
            "features": [
                "Emotion detection",
                "Satisfaction tracking",
                "Usage pattern analysis",
                "Performance monitoring"
            ],
            "gradient": "linear-gradient(145deg, #4facfe 0%, #00f2fe 100%)"
        }
    ]
    
    for i, (col, feature) in enumerate(zip([col1, col2, col3], features)):
        with col:
            st.markdown(f"""
            <div class="feature-card" style="background: {feature['gradient']}; animation-delay: {i*0.25}s; border-radius: 28px; box-shadow: var(--shadow-strong);">
                <h3 style="margin-top: 0; font-size: 1.4rem; font-weight: 700;">{feature['title']}</h3>
                <p style="opacity: 0.95; font-size: 1.1rem; margin-bottom: 2rem; font-weight: 500;">{feature['description']}</p>
                <ul style="list-style: none; padding: 0; margin-top: 2rem;">
                    {"".join([f'<li style="margin: 1rem 0; padding-left: 1.5rem; position: relative; font-weight: 500;"><span style="position: absolute; left: 0; color: rgba(255,255,255,0.8);">•</span> {item}</li>' for item in feature['features']])}
                </ul>
            </div>
            """, unsafe_allow_html=True)
    
    # Quick actions with enhanced buttons
    st.markdown("## 🚀 Quick Actions")
    
    action_col1, action_col2, action_col3, action_col4 = st.columns(4)
    
    quick_actions = [
        ("💬 Start AI Chat", "ai_chat", "var(--uel-gradient)"),
        ("🔮 Get Predictions", "predictions", "linear-gradient(145deg, #667eea 0%, #764ba2 100%)"),
        ("🎯 Course Match", "recommendations", "linear-gradient(145deg, #f093fb 0%, #f5576c 100%)"),
        ("📄 Verify Documents", "documents", "linear-gradient(145deg, #4facfe 0%, #00f2fe 100%)")
    ]
    
    for i, (col, (label, page, gradient)) in enumerate(zip([action_col1, action_col2, action_col3, action_col4], quick_actions)):
        with col:
            if st.button(label, key=f"quick_action_{i}", use_container_width=True, type="primary"):
                st.session_state.current_page = page
                st.rerun()
    
    # System health and recent activity
    st.markdown("## 📈 System Health & Activity")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if PLOTLY_AVAILABLE and analytics['statistics']['total_interactions'] > 0:
            # Create a simple activity chart
            hours = list(range(24))
            activity = [random.randint(1, 10) for _ in hours]
            
            fig = px.bar(
                x=hours,
                y=activity,
                title="Hourly Activity Pattern",
                labels={'x': 'Hour', 'y': 'Interactions'},
                color=activity,
                color_continuous_scale=[[0, '#E8F6F5'], [1, '#00B8A9']]
            )
            fig.update_layout(height=300, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("📊 Activity charts will appear as data accumulates")
    
    with col2:
        st.markdown("### 🔧 System Components")
        
        components = [
            ("Data Manager", "✅ Operational", "🟢"),
            ("LLM Service", "✅ Connected" if feature_status['llm_integration'] else "❌ Offline", "🟢" if feature_status['llm_integration'] else "🔴"),
            ("ML Engine", "✅ Ready" if feature_status['ml_predictions'] else "⏳ Training", "🟢" if feature_status['ml_predictions'] else "🟡"),
            ("Voice Service", "✅ Available" if feature_status['voice_services'] else "❌ Disabled", "🟢" if feature_status['voice_services'] else "🔴"),
            ("Document AI", "✅ Active", "🟢")
        ]
        
        for component, status, indicator in components:
            st.markdown(f"""
            <div style="display: flex; justify-content: space-between; align-items: center; padding: 1rem; margin: 0.5rem 0; background: var(--uel-card-gradient); border-radius: 12px; box-shadow: 0 2px 8px rgba(0,0,0,0.1);">
                <span style="font-weight: 600; color: var(--text-primary);">{component}</span>
                <div style="display: flex; align-items: center; gap: 0.5rem;">
                    <span style="font-size: 0.9rem; color: var(--text-secondary);">{status}</span>
                    <span style="font-size: 1.2rem;">{indicator}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

# Keep the rest of the original functions but include the chat page with enhanced styling...

def render_chat_page():
    """Enhanced chat page with improved error handling and UX"""
    st.markdown("""
    <div class="page-header">
        <div class="header-icon">💬</div>
        <div>
            <h1 style="margin: 0; background: var(--uel-gradient); -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-weight: 800; font-size: 2.5rem;">AI Chat Assistant</h1>
            <p style="margin: 0; color: var(--text-secondary); font-size: 1.2rem; font-weight: 500;">Enhanced Intelligent UEL Support System</p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Add a floating chatbot animation
    st.markdown("""
    <div style="position: fixed; bottom: 30px; right: 30px; z-index: 1000;">
        <div style="
            width: 70px; 
            height: 70px; 
            background: var(--uel-gradient); 
            border-radius: 50%; 
            display: flex; 
            align-items: center; 
            justify-content: center; 
            font-size: 28px; 
            animation: float 4s ease-in-out infinite;
            box-shadow: var(--shadow-glow);
            border: 3px solid rgba(255,255,255,0.3);
        ">
            🤖
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize managers
    if 'chat_message_manager' not in st.session_state:
        st.session_state.chat_message_manager = ChatMessageManager()
    
    if 'chat_error_handler' not in st.session_state:
        st.session_state.chat_error_handler = ChatErrorHandler()
    
    message_manager = st.session_state.chat_message_manager
    error_handler = st.session_state.chat_error_handler
    
    # Initialize chat
    message_manager.initialize_chat()
    
    # Get AI agent safely
    try:
        ai_agent = get_ai_agent()
    except Exception as e:
        st.error(f"⚠️ Failed to initialize AI agent: {e}")
        st.info("💡 Please refresh the page or contact support if the issue persists.")
        return
    
    # Handle course inquiry from recommendations
    if handle_course_inquiry_from_recommendations():
        st.rerun()
    
    # Profile setup modal
    if st.session_state.get('show_profile_setup', False):
        render_profile_setup_modal()
        return
    
    # Show profile reminder if not set
    if not st.session_state.current_student:
        st.markdown("""
        <div style="background: linear-gradient(135deg, rgba(100, 181, 246, 0.1), rgba(100, 181, 246, 0.05)); border: 2px solid var(--uel-info); border-radius: 20px; padding: 1.5rem; margin-bottom: 2rem;">
            <h4 style="margin-top: 0; color: var(--uel-info);">💡 Personalize Your Experience</h4>
            <p style="margin-bottom: 1rem;">Create your profile to get personalized course recommendations and better AI responses!</p>
        """, unsafe_allow_html=True)
        
        if st.button("📝 Set Up Profile Now", type="primary"):
            st.session_state.show_profile_setup = True
            st.rerun()
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Chat interface layout
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Quick action buttons with enhanced styling
        st.markdown("### 🚀 Quick Actions")
        
        action_buttons = [
            ("🎓 Course Info", "Tell me about available courses at UEL"),
            ("📝 Application Help", "I need help with my UEL application"),
            ("💰 Fees & Scholarships", "What are the tuition fees and scholarship options?"),
            ("📄 Documents", "What documents do I need for my application?"),
            ("🔮 Admission Chances", "Can you predict my admission chances?"),
            ("🎯 Course Match", "Recommend courses based on my interests")
        ]
        
        button_cols = st.columns(3)
        for i, (label, message) in enumerate(action_buttons):
            with button_cols[i % 3]:
                if st.button(label, key=f"quick_action_{i}", use_container_width=True):
                    success = error_handler.handle_user_message_safely(message, ai_agent, message_manager)
                    if success:
                        st.rerun()
        
        # Chat messages display
        st.markdown("### 💬 Conversation")
        
        # Messages container
        messages_container = st.container()
        
        # Display messages with enhanced formatting
        with messages_container:
            for message in st.session_state.messages:
                render_enhanced_message(message)
        
        # Input area
        st.markdown("---")
        
        # Chat input form with voice integration
        st.markdown("### 💭 Ask the AI")
        
        # Voice mode toggle
        voice_mode = st.checkbox("🎤 Voice Conversation Mode", help="Enable voice input and automatic voice responses")
        
        if voice_mode:
            st.markdown("""
            <div style="background: linear-gradient(135deg, rgba(76, 205, 196, 0.1), rgba(76, 205, 196, 0.05)); border: 2px solid var(--uel-success); border-radius: 16px; padding: 1rem; margin-bottom: 1rem;">
                🎤 <strong>Voice Mode Active</strong> - Speak your question or type it. AI will respond with voice!
            </div>
            """, unsafe_allow_html=True)
        
        with st.form("chat_form", clear_on_submit=True):
            input_col1, input_col2, input_col3 = st.columns([6, 1, 1])
            
            with input_col1:
                user_input = st.text_area(
                    "Your message:",
                    placeholder="Ask about courses, admissions, requirements, or any UEL-related questions...",
                    height=100,
                    key="chat_input",
                    help="Type your question here or use voice input"
                )
            
            with input_col2:
                st.markdown("<br>", unsafe_allow_html=True)
                send_button = st.form_submit_button("Send 📤", use_container_width=True, type="primary")
            
            with input_col3:
                st.markdown("<br>", unsafe_allow_html=True)
                voice_button = st.form_submit_button("🎤 Voice", use_container_width=True)
            
            # Handle input submission
            if send_button and user_input.strip():
                success = error_handler.handle_user_message_safely(
                    user_input, ai_agent, message_manager, voice_response=voice_mode
                )
                if success:
                    st.rerun()
            
            if voice_button:
                success = error_handler.handle_voice_conversation_safely(
                    ai_agent, message_manager, voice_response=voice_mode
                )
                if success:
                    st.rerun()
    
    with col2:
        render_chat_sidebar(ai_agent, message_manager, error_handler)


def render_enhanced_message(message):
    """Render a single message with enhanced formatting and animations"""
    try:
        sender = message.get('sender', 'unknown')
        content = message.get('message', '')
        timestamp = message.get('timestamp', '')
        message_type = message.get('type', 'text')
        metadata = message.get('metadata', {})
        
        # Format timestamp
        try:
            dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            time_str = dt.strftime("%H:%M")
        except:
            time_str = "now"
        
        if sender == "ai":
            # Get indicators
            voice_indicator = " 🔊" if metadata.get('voice_response') else ""
            llm_info = metadata.get('llm_used', 'AI')
            
            # Choose styling based on message type
            if message_type == "error":
                border_color = "#E57373"
                bg_gradient = "linear-gradient(135deg, #ffebee, #ffcdd2)"
            elif message_type == "welcome":
                border_color = "#8b5cf6"
                bg_gradient = "linear-gradient(135deg, #f3f4f6, #e5e7eb)"
            else:
                border_color = "#00B8A9"
                bg_gradient = "var(--uel-card-gradient)"
            
            st.markdown(f"""
            <div class="ai-message" style="background: {bg_gradient}; border-left-color: {border_color};">
                <div style="display: flex; align-items: center; margin-bottom: 1rem;">
                    <div style="background: var(--uel-gradient); color: white; width: 40px; height: 40px; border-radius: 50%; display: flex; align-items: center; justify-content: center; margin-right: 1rem; box-shadow: 0 4px 12px rgba(0, 184, 169, 0.3);">🤖</div>
                    <div>
                        <strong style="color: var(--text-primary); font-size: 1.1rem;">UEL AI Assistant{voice_indicator}</strong>
                        <div style="font-size: 0.8rem; color: var(--text-light); font-weight: 500;">{time_str} • {llm_info}</div>
                    </div>
                </div>
                <div style="margin-left: 3rem; color: var(--text-primary); line-height: 1.6; font-size: 1rem;">{content}</div>
                {f'<div style="margin-top: 1rem; margin-left: 3rem;"><small style="color: #00B8A9; font-weight: 600;">🔊 Voice response played</small></div>' if metadata.get('voice_response') else ''}
            </div>
            """, unsafe_allow_html=True)
            
        elif sender == "user":
            user_name = "You"
            if st.session_state.get('current_student'):
                first_name = st.session_state.current_student.get('first_name', '')
                if first_name:
                    user_name = first_name
            
            st.markdown(f"""
            <div class="user-message">
                <div style="display: flex; align-items: center; justify-content: flex-end; margin-bottom: 1rem;">
                    <div style="text-align: right; margin-right: 1rem;">
                        <strong style="font-size: 1.1rem;">{user_name}</strong>
                        <div style="font-size: 0.8rem; color: rgba(255,255,255,0.8); font-weight: 500;">{time_str}</div>
                    </div>
                    <div style="background: rgba(255,255,255,0.25); color: white; width: 40px; height: 40px; border-radius: 50%; display: flex; align-items: center; justify-content: center; backdrop-filter: blur(10px);">👤</div>
                </div>
                <div style="margin-right: 3rem; line-height: 1.6; font-size: 1rem;">{content}</div>
            </div>
            """, unsafe_allow_html=True)
        
    except Exception as e:
        st.error(f"Error rendering message: {e}")


def render_chat_sidebar(ai_agent, message_manager, error_handler):
    """Enhanced sidebar with better error handling and new features"""
    
    # Profile section with enhanced styling
    st.markdown("### 👤 Your Profile")
    
    if st.session_state.current_student:
        student = st.session_state.current_student
        
        # Profile display card with enhanced styling
        st.markdown(f"""
        <div style="background: var(--uel-gradient); color: white; padding: 1.5rem; border-radius: 16px; margin-bottom: 1.5rem; box-shadow: var(--shadow-medium);">
            <div style="font-size: 1.2rem; font-weight: 700; margin-bottom: 1rem;">👤 {student.get('first_name', '')} {student.get('last_name', '')}</div>
            <div style="font-size: 1rem; line-height: 1.6;">
                🎓 <strong>Interest:</strong> {student.get('field_of_interest', 'Not specified')}<br>
                📚 <strong>Level:</strong> {student.get('academic_level', 'Not specified').title()}<br>
                🌍 <strong>Country:</strong> {student.get('country', 'Not specified')}
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("✏️ Edit Profile", use_container_width=True, key="edit_profile_sidebar"):
                st.session_state.show_profile_setup = True
                st.rerun()
        
        with col2:
            if st.button("🎯 Get Recommendations", use_container_width=True, key="get_recs_sidebar"):
                st.session_state.current_page = "recommendations"
                st.rerun()
        
        st.success("✅ Profile active!")
        
    else:
        st.markdown("""
        <div style="background: linear-gradient(135deg, rgba(100, 181, 246, 0.1), rgba(100, 181, 246, 0.05)); border: 2px solid var(--uel-info); border-radius: 16px; padding: 1.5rem; text-align: center;">
            <h4 style="margin-top: 0; color: var(--uel-info);">💡 Create Your Profile</h4>
            <p style="margin-bottom: 1rem;">Get personalized responses and recommendations!</p>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("📝 Create Profile", use_container_width=True, type="primary", key="create_profile_sidebar"):
            st.session_state.show_profile_setup = True
            st.rerun()
    
    # Chat Statistics with enhanced styling
    st.markdown("---")
    st.markdown("### 📊 Chat Statistics")
    
    try:
        stats = message_manager.get_chat_statistics()
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Messages", stats['total'], delta=None)
            st.metric("Your Messages", stats['user'], delta=None)
        with col2:
            st.metric("AI Responses", stats['ai'], delta=None)
            st.metric("Avg Length", f"{stats['avg_length']:.0f}", delta=None)
    
    except Exception as e:
        st.warning("Stats temporarily unavailable")
    
    # Voice controls with enhanced styling
    st.markdown("---")
    st.markdown("### 🎤 Voice Features")

    # Show voice service status
    try:
        if ai_agent and ai_agent.voice_service.is_available():
            st.success("✅ Voice service ready")
        else:
            st.error("❌ Voice service unavailable")
            st.info("Install: `pip install SpeechRecognition pyttsx3 pyaudio`")
    except:
        st.error("❌ Voice service error")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔊 Repeat Last", use_container_width=True, help="Replay AI's last response", key="voice_repeat"):
            if st.session_state.messages:
                last_ai_message = None
                for msg in reversed(st.session_state.messages):
                    if msg['sender'] == 'ai' and msg.get('type') != 'error':
                        last_ai_message = msg
                        break
                
                if last_ai_message and ai_agent:
                    with st.spinner("🔊 Speaking..."):
                        try:
                            success = ai_agent.text_to_speech(last_ai_message['message'])
                            if success:
                                st.success("🔊 Playing response!")
                            else:
                                st.error("❌ Voice not available")
                        except:
                            st.error("❌ Voice error")
                else:
                    st.warning("No AI messages to repeat")
            else:
                st.warning("No conversation yet")

    with col2:
        if st.button("🎤 Voice Test", use_container_width=True, help="Test voice input", key="voice_test"):
            if ai_agent and ai_agent.voice_service.is_available():
                success = error_handler.handle_voice_conversation_safely(
                    ai_agent, message_manager, voice_response=False
                )
                if success:
                    st.rerun()
            else:
                st.error("Voice service not available")
    
    # Chat controls with enhanced styling
    st.markdown("---")
    st.markdown("### 🛠️ Chat Controls")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🗑️ Clear Chat", use_container_width=True, key="clear_chat_sidebar"):
            if message_manager.clear_chat():
                st.success("✅ Chat cleared!")
                st.rerun()
            else:
                st.error("❌ Failed to clear chat")
    
    with col2:
        if st.button("💾 Export Chat", use_container_width=True, key="export_chat_sidebar"):
            chat_data = message_manager.export_chat_history()
            if chat_data:
                st.download_button(
                    label="📥 Download",
                    data=chat_data,
                    file_name=f"uel_chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
            else:
                st.error("❌ Export failed")
    
    # System info with enhanced styling
    try:
        error_stats = error_handler.get_error_stats()
        
        if error_stats['total_errors'] > 0:
            st.warning(f"⚠️ {error_stats['total_errors']} errors occurred")
        else:
            st.success("✅ No errors detected")
    except Exception as e:
        st.info("System info unavailable")

# Helper functions for the chat system
def handle_course_inquiry_from_recommendations():
    """Handle course inquiries that come from the recommendations page"""
    if st.session_state.get('course_inquiry'):
        course_name = st.session_state.course_inquiry
        inquiry_message = f"Can you tell me more about the {course_name} program at UEL? I'm particularly interested in the curriculum, career prospects, and what makes this program special."
        
        # Add this message to the chat
        ai_agent = get_ai_agent()
        # Note: You'll need to integrate this with your chat system
        
        # Clear the inquiry flag
        del st.session_state.course_inquiry
        
        return True
    return False

def render_profile_setup_modal():
    """Render profile setup modal with comprehensive form"""
    st.markdown("---")
    st.markdown("### 👤 Set Up Your Profile")
    st.markdown("*Get personalized AI responses and recommendations*")
    
    with st.form("profile_setup_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**📋 Basic Information**")
            first_name = st.text_input(
                "First Name *", 
                value=st.session_state.current_student.get('first_name', '') if st.session_state.current_student else '',
                placeholder="John",
                help="Enter your first name"
            )
            last_name = st.text_input(
                "Last Name *", 
                value=st.session_state.current_student.get('last_name', '') if st.session_state.current_student else '',
                placeholder="Smith",
                help="Enter your last name"
            )
            email = st.text_input(
                "Email", 
                value=st.session_state.current_student.get('email', '') if st.session_state.current_student else '',
                placeholder="john.smith@email.com",
                help="Your email address"
            )
            phone = st.text_input(
                "Phone", 
                value=st.session_state.current_student.get('phone', '') if st.session_state.current_student else '',
                placeholder="+44 7XXX XXXXXX",
                help="Your phone number"
            )
        
        with col2:
            st.markdown("**🎓 Academic Information**")
            field_of_interest = st.text_input(
                "Field of Interest *", 
                value=st.session_state.current_student.get('field_of_interest', '') if st.session_state.current_student else '',
                placeholder="Computer Science",
                help="Your main area of academic interest"
            )
            academic_level = st.selectbox(
                "Academic Level *", 
                ["High School", "Undergraduate", "Graduate", "Postgraduate"],
                index=0 if not st.session_state.current_student else 
                ["high school", "undergraduate", "graduate", "postgraduate"].index(
                    st.session_state.current_student.get('academic_level', 'undergraduate')
                ) if st.session_state.current_student.get('academic_level', 'undergraduate').lower() in 
                ["high school", "undergraduate", "graduate", "postgraduate"] else 0,
                help="Your current academic level"
            )
            country = st.text_input(
                "Country", 
                value=st.session_state.current_student.get('country', '') if st.session_state.current_student else '',
                placeholder="United Kingdom",
                help="Your country of residence"
            )
            nationality = st.text_input(
                "Nationality", 
                value=st.session_state.current_student.get('nationality', '') if st.session_state.current_student else '',
                placeholder="British",
                help="Your nationality"
            )
        
        # Additional interests
        st.markdown("**🎯 Interests & Goals** *(Optional)*")
        interests = st.multiselect(
            "Areas of Interest",
            ["Artificial Intelligence", "Data Science", "Cybersecurity", "Business Management", 
             "Finance", "Marketing", "Psychology", "Engineering", "Arts", "Medicine", "Law", "Education"],
            default=st.session_state.current_student.get('interests', []) if st.session_state.current_student else [],
            help="Select areas that interest you"
        )
        
        career_goals = st.text_area(
            "Career Goals",
            value=st.session_state.current_student.get('career_goals', '') if st.session_state.current_student else '',
            placeholder="Describe your career aspirations...",
            height=100,
            help="What are your career goals and aspirations?"
        )
        
        # Form buttons
        col3, col4 = st.columns(2)
        with col3:
            save_profile = st.form_submit_button("💾 Save Profile", type="primary", use_container_width=True)
        with col4:
            cancel_profile = st.form_submit_button("❌ Cancel", use_container_width=True)
        
        if save_profile and first_name and last_name and field_of_interest:
            # Create comprehensive profile
            profile_data = {
                'id': st.session_state.get('user_id'),
                'first_name': first_name,
                'last_name': last_name,
                'email': email,
                'phone': phone,
                'field_of_interest': field_of_interest,
                'academic_level': academic_level.lower(),
                'country': country,
                'nationality': nationality,
                'interests': interests + ([field_of_interest] if field_of_interest not in interests else []),
                'career_goals': career_goals,
                'created_date': datetime.now().isoformat(),
                'updated_date': datetime.now().isoformat()
            }
            
            # Store in both current_student and user_profile for compatibility
            st.session_state.current_student = profile_data
            st.session_state.user_profile = profile_data
            st.session_state.show_profile_setup = False
            
            st.success(f"✅ Profile saved for {first_name} {last_name}! You'll now get personalized AI responses.")
            st.balloons()  # Celebration effect
            st.rerun()
        
        elif save_profile:
            st.error("❌ Please fill in all required fields marked with *")
        
        if cancel_profile:
            st.session_state.show_profile_setup = False
            st.rerun()

# =============================================================================
# ENHANCED PREDICTIONS PAGE
# =============================================================================

def render_predictions_page():
    """Render enhanced AI-powered admission predictions with animations"""
    st.markdown("""
    <div class="page-header">
        <div class="header-icon">🔮</div>
        <div>
            <h1 style="margin: 0; background: var(--uel-gradient); -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-weight: 800; font-size: 2.5rem;">Advanced Admission Predictions</h1>
            <p style="margin: 0; color: var(--text-secondary); font-size: 1.2rem; font-weight: 500;">AI-powered admission success forecasting with comprehensive analysis</p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    ai_agent = get_ai_agent()
    
    # Enhanced information box with animations
    st.markdown("""
    <div class="enhanced-card" style="background: linear-gradient(145deg, rgba(102, 126, 234, 0.1), rgba(118, 75, 162, 0.1)); border-left: 6px solid var(--uel-primary);">
        <h3 style="margin-top: 0; color: var(--uel-primary); font-weight: 700; font-size: 1.4rem;">🎯 Enhanced Prediction System</h3>
        <p style="font-size: 1.1rem; color: var(--text-primary); margin-bottom: 1.5rem;">Our AI analyzes multiple factors including:</p>
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 1.5rem; margin-top: 2rem;">
            <div style="padding: 1rem; background: rgba(0, 184, 169, 0.15); border-radius: 12px; border: 1px solid rgba(0, 184, 169, 0.2);">
                <strong>🎓 Education level compatibility</strong>
            </div>
            <div style="padding: 1rem; background: rgba(0, 184, 169, 0.15); border-radius: 12px; border: 1px solid rgba(0, 184, 169, 0.2);">
                <strong>📚 Academic performance analysis</strong>
            </div>
            <div style="padding: 1rem; background: rgba(0, 184, 169, 0.15); border-radius: 12px; border: 1px solid rgba(0, 184, 169, 0.2);">
                <strong>🌐 English proficiency assessment</strong>
            </div>
            <div style="padding: 1rem; background: rgba(0, 184, 169, 0.15); border-radius: 12px; border: 1px solid rgba(0, 184, 169, 0.2);">
                <strong>💼 Work experience evaluation</strong>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Load courses from CSV data
    try:
        courses_data = ai_agent.data_manager.courses_df
        if courses_data.empty:
            st.error("❌ Course data not available. Please ensure courses.csv is loaded.")
            return
        
        st.success(f"✅ Loaded {len(courses_data)} courses from database")
    except Exception as e:
        st.error(f"❌ Error loading course data: {e}")
        return
    
    # Enhanced ML status check
    if not ai_agent.predictive_engine.models_trained:
        st.markdown("""
        <div class="enhanced-card" style="background: linear-gradient(145deg, rgba(245, 158, 11, 0.1), rgba(245, 158, 11, 0.15)); border-left: 6px solid var(--uel-warning);">
            <h3 style="margin-top: 0; color: var(--uel-warning); font-weight: 700;">⚠️ ML Models Training Required</h3>
            <p style="font-size: 1.1rem;">Our system needs to train machine learning models for accurate predictions.</p>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.info("""
            🧠 **Training the AI Prediction Models**
            
            Our system will train machine learning models using:
            - Random Forest for admission classification
            - Gradient Boosting for success probability
            - 11 enhanced features including education compatibility
            - Synthetic training data if real data is limited
            """)
        
        with col2:
            if st.button("🚀 Train Models Now", type="primary", use_container_width=True):
                with st.spinner("🧠 Training ML models... This may take a moment."):
                    try:
                        ai_agent.predictive_engine.train_models()
                        
                        if ai_agent.predictive_engine.models_trained:
                            st.success("✅ Models trained successfully!")
                            st.balloons()
                            time.sleep(1)
                            st.rerun()
                        else:
                            st.error("❌ Model training failed. Using fallback prediction system.")
                            
                    except Exception as e:
                        st.error(f"❌ Training error: {e}")
                        st.info("💡 Using fallback prediction system for basic functionality.")
    
    st.markdown("## 📋 Enter Your Information")
    
    with st.form("enhanced_prediction_form"):
        # Personal Information Section with enhanced styling
        st.markdown("""
        <div style="background: var(--uel-light-gradient); padding: 2rem; border-radius: 16px; margin-bottom: 2rem; border: 1px solid rgba(0, 184, 169, 0.2);">
            <h3 style="margin-top: 0; color: var(--uel-primary); font-weight: 700; font-size: 1.4rem;">👤 Personal Information</h3>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            first_name = st.text_input("First Name", placeholder="John", help="Enter your first name")
            last_name = st.text_input("Last Name", placeholder="Smith", help="Enter your last name")
            email = st.text_input("Email (optional)", placeholder="john.smith@email.com", help="Your email address")
        
        with col2:
            nationality = st.selectbox("Nationality", [
                "United Kingdom", "United States", "India", "China", 
                "Nigeria", "Pakistan", "Canada", "Germany", "France",
                "Australia", "Brazil", "Other"
            ], help="Select your nationality")
            age = st.number_input("Age", min_value=16, max_value=60, value=22, help="Your current age")
            application_date = st.date_input("Intended Application Date", datetime.now(), help="When do you plan to apply?")
        
        st.markdown("---")
        
        # Current Education Section with enhanced styling
        st.markdown("""
        <div style="background: var(--uel-light-gradient); padding: 2rem; border-radius: 16px; margin-bottom: 2rem; border: 1px solid rgba(0, 184, 169, 0.2);">
            <h3 style="margin-top: 0; color: var(--uel-primary); font-weight: 700; font-size: 1.4rem;">🎓 Current Education Status</h3>
        </div>
        """, unsafe_allow_html=True)
        
        col3, col4 = st.columns(2)
        
        with col3:
            current_education = st.selectbox(
                "Current/Highest Education Level",
                [
                    "High School", "High School with A-Levels", "Diploma", "Foundation Program",
                    "Undergraduate (in progress)", "Bachelor's Degree", "Graduate Certificate",
                    "Postgraduate (in progress)", "Master's Degree", "MBA", "PhD (in progress)", "PhD"
                ],
                help="Select your current or highest completed education level"
            )
            
            gpa = st.slider(
                "GPA/Grade Average (0.0-4.0)",
                0.0, 4.0, 3.0, 0.1,
                help="Your cumulative GPA on a 4.0 scale"
            )
            
            # Enhanced GPA feedback with animations
            if gpa >= 3.7:
                st.markdown('<div class="success-animation">🌟 Excellent GPA for your level!</div>', unsafe_allow_html=True)
            elif gpa >= 3.3:
                st.info(f"👍 Good GPA for {current_education} level")
            elif gpa >= 2.7:
                st.warning(f"📊 Average GPA for {current_education} level")
            else:
                st.markdown('<div class="error-animation">⚠️ Below average GPA - consider improvement strategies</div>', unsafe_allow_html=True)
        
        with col4:
            major_field = st.text_input(
                "Current/Previous Major",
                placeholder="e.g., Computer Science, Business, Psychology",
                help="Your field of study in current/previous education"
            )
            
            ielts_score = st.slider(
                "IELTS Score (or equivalent)",
                0.0, 9.0, 6.5, 0.5,
                help="Your English proficiency score"
            )
            
            # Enhanced IELTS feedback with animations
            if ielts_score >= 7.5:
                st.markdown('<div class="success-animation">🌟 Excellent English proficiency!</div>', unsafe_allow_html=True)
            elif ielts_score >= 6.5:
                st.info("👍 Good English proficiency")
            elif ielts_score >= 6.0:
                st.warning("📊 Meets minimum requirements")
            else:
                st.markdown('<div class="error-animation">⚠️ May need English improvement</div>', unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Enhanced Target Course Section
        st.markdown("""
        <div style="background: var(--uel-light-gradient); padding: 2rem; border-radius: 16px; margin-bottom: 2rem; border: 1px solid rgba(0, 184, 169, 0.2);">
            <h3 style="margin-top: 0; color: var(--uel-primary); font-weight: 700; font-size: 1.4rem;">🎯 Target Course Selection</h3>
        </div>
        """, unsafe_allow_html=True)
        
        # Course filtering with enhanced UI
        col5, col6 = st.columns([2, 1])
        
        with col5:
            departments = sorted(courses_data['department'].unique()) if 'department' in courses_data.columns else ['All Departments']
            selected_department = st.selectbox(
                "Filter by Department",
                ['All Departments'] + list(departments),
                help="Narrow down courses by department"
            )
            
            levels = sorted(courses_data['level'].unique()) if 'level' in courses_data.columns else ['All Levels']
            selected_level = st.selectbox(
                "Filter by Level",
                ['All Levels'] + list(levels),
                help="Filter by academic level"
            )
        
        with col6:
            search_term = st.text_input(
                "Search Courses",
                placeholder="e.g., Computer Science, AI, Business",
                help="Search for specific courses"
            )
            
            show_trending = st.checkbox(
                "Show High-Demand Courses Only",
                help="Filter courses with trending score ≥ 7.0"
            )
        
        # Filter courses based on selections
        filtered_courses = courses_data.copy()
        
        if selected_department != 'All Departments':
            filtered_courses = filtered_courses[filtered_courses['department'] == selected_department]
        
        if selected_level != 'All Levels':
            filtered_courses = filtered_courses[filtered_courses['level'] == selected_level]
        
        if search_term:
            search_mask = (
                filtered_courses['course_name'].str.contains(search_term, case=False, na=False) |
                filtered_courses['description'].str.contains(search_term, case=False, na=False) |
                filtered_courses['keywords'].str.contains(search_term, case=False, na=False)
            )
            filtered_courses = filtered_courses[search_mask]
        
        if show_trending and 'trending_score' in filtered_courses.columns:
            filtered_courses = filtered_courses[filtered_courses['trending_score'] >= 7.0]
        
        # Course selection with enhanced display
        if len(filtered_courses) == 0:
            st.warning("⚠️ No courses match your filters. Please adjust your search criteria.")
            course_applied = None
            selected_course_info = None
        else:
            st.success(f"✅ Found {len(filtered_courses)} courses matching your criteria")
            
            # Create enhanced course options
            course_options = []
            course_info_map = {}
            
            for _, course in filtered_courses.iterrows():
                fees_int = course.get('fees_international', 'N/A')
                trending = course.get('trending_score', 0)
                trending_indicator = " 🔥" if trending >= 8.0 else " 📈" if trending >= 7.0 else ""
                
                display_name = f"{course['course_name']} ({course.get('level', 'N/A')}) - £{fees_int:,}{trending_indicator}"
                course_options.append(display_name)
                course_info_map[display_name] = course
            
            selected_course_display = st.selectbox(
                "Select Target Course",
                course_options,
                help="Choose the course you want to apply for"
            )
            
            course_applied = course_info_map[selected_course_display]['course_name']
            selected_course_info = course_info_map[selected_course_display]
        
        # Display selected course information with enhanced styling
        if selected_course_info is not None:
            st.markdown("#### 📚 Selected Course Details")
            
            course_col1, course_col2 = st.columns(2)
            
            with course_col1:
                st.markdown(f"""
                <div class="enhanced-card" style="padding: 1.5rem; border-left: 4px solid var(--uel-primary);">
                    <strong style="color: var(--uel-primary);">Course:</strong> {selected_course_info['course_name']}<br><br>
                    <strong style="color: var(--uel-primary);">Level:</strong> {selected_course_info.get('level', 'N/A')}<br><br>
                    <strong style="color: var(--uel-primary);">Department:</strong> {selected_course_info.get('department', 'N/A')}<br><br>
                    <strong style="color: var(--uel-primary);">Duration:</strong> {selected_course_info.get('duration', 'N/A')}
                </div>
                """, unsafe_allow_html=True)
            
            with course_col2:
                fees_dom = selected_course_info.get('fees_domestic', 0)
                fees_int = selected_course_info.get('fees_international', 0)
                min_gpa = selected_course_info.get('min_gpa', 0)
                min_ielts = selected_course_info.get('min_ielts', 0)
                trending = selected_course_info.get('trending_score', 0)
                
                st.markdown(f"""
                <div class="enhanced-card" style="padding: 1.5rem; border-left: 4px solid var(--uel-secondary);">
                    <strong style="color: var(--uel-secondary);">Fees (Domestic):</strong> £{fees_dom:,}<br><br>
                    <strong style="color: var(--uel-secondary);">Fees (International):</strong> £{fees_int:,}<br><br>
                    <strong style="color: var(--uel-secondary);">Min GPA:</strong> {min_gpa}<br><br>
                    <strong style="color: var(--uel-secondary);">Min IELTS:</strong> {min_ielts}<br><br>
                    <strong style="color: var(--uel-secondary);">Trending Score:</strong> {trending}/10
                </div>
                """, unsafe_allow_html=True)
        
        # Work Experience Section
        st.markdown("---")
        st.markdown("""
        <div style="background: var(--uel-light-gradient); padding: 2rem; border-radius: 16px; margin-bottom: 2rem; border: 1px solid rgba(0, 184, 169, 0.2);">
            <h3 style="margin-top: 0; color: var(--uel-primary); font-weight: 700; font-size: 1.4rem;">💼 Experience & Background</h3>
        </div>
        """, unsafe_allow_html=True)
        
        col7, col8 = st.columns(2)
        
        with col7:
            work_experience = st.number_input(
                "Relevant Work Experience (years)",
                0, 20, 0,
                help="Years of work experience related to your target field"
            )
            
            financial_support = st.selectbox(
                "Financial Support Status",
                ["Self-funded", "Family Support", "Employer Sponsored", "Scholarship", "Student Loan", "Other"],
                help="How will you fund your studies?"
            )
        
        with col8:
            extracurriculars = st.multiselect(
                "Relevant Activities/Achievements",
                [
                    "Research Publications", "Academic Awards", "Leadership Roles",
                    "Volunteer Work", "Professional Certifications", "Internships",
                    "Projects/Portfolio", "Sports Achievements", "Cultural Activities"
                ],
                help="Select any relevant activities or achievements"
            )
            
            preferred_start_date = st.selectbox(
                "Preferred Start Date",
                ["September 2024", "January 2025", "September 2025", "January 2026", "Flexible"],
                help="When would you like to start your studies?"
            )
        
        # Submit button with enhanced styling
        st.markdown("---")
        predict_button = st.form_submit_button(
            "🔮 Get Comprehensive Prediction Analysis",
            use_container_width=True,
            type="primary"
        )
    
    # Process prediction with enhanced animations
    if predict_button and first_name and last_name and course_applied:
        # Prepare comprehensive applicant data
        applicant_data = {
            'name': f"{first_name} {last_name}",
            'email': email,
            'gpa': gpa,
            'ielts_score': ielts_score,
            'work_experience_years': work_experience,
            'course_applied': course_applied,
            'nationality': nationality,
            'application_date': application_date.strftime('%Y-%m-%d'),
            'current_education': current_education,
            'major_field': major_field,
            'age': age,
            'extracurriculars': extracurriculars,
            'financial_support': financial_support,
            'preferred_start_date': preferred_start_date,
            'course_info': selected_course_info.to_dict() if selected_course_info is not None else {}
        }
        
        # Enhanced loading animation
        with st.spinner("🧠 AI is conducting comprehensive analysis..."):
            progress = st.progress(0)
            status = st.empty()
            
            loading_steps = [
                ("📊 Analyzing academic credentials...", 15),
                ("🎓 Evaluating education progression...", 30),
                ("📈 Comparing with successful applicants...", 45),
                ("🔍 Assessing program compatibility...", 60),
                ("💼 Analyzing work experience relevance...", 75),
                ("✨ Generating personalized insights...", 90),
                ("🎯 Finalizing prediction report...", 100)
            ]
            
            for step_text, progress_value in loading_steps:
                status.text(step_text)
                progress.progress(progress_value)
                time.sleep(0.5)
            
            # Get prediction
            try:
                prediction_result = ai_agent.predict_admission_success(applicant_data)
            except Exception as e:
                st.error(f"❌ Prediction error: {e}")
                prediction_result = {'error': str(e)}
            
            # Clear progress indicators
            progress.empty()
            status.empty()
        
        if 'error' not in prediction_result:
            # Extract results
            success_prob = prediction_result.get('success_probability', 0.5)
            confidence = prediction_result.get('confidence', 0.7)
            education_compatibility = prediction_result.get('education_compatibility', 0.7)
            
            # Determine overall status with enhanced styling
            if success_prob >= 0.8:
                color = "#10b981"
                status = "Excellent"
                icon = "🎉"
                message = "Outstanding application profile!"
                gradient = "linear-gradient(145deg, #10b981, #059669)"
            elif success_prob >= 0.6:
                color = "#3b82f6"
                status = "Good"
                icon = "👍"
                message = "Strong application with good chances"
                gradient = "linear-gradient(145deg, #3b82f6, #1e40af)"
            elif success_prob >= 0.4:
                color = "#f59e0b"
                status = "Moderate"
                icon = "⚠️"
                message = "Decent chances with improvements"
                gradient = "linear-gradient(145deg, #f59e0b, #d97706)"
            else:
                color = "#ef4444"
                status = "Challenging"
                icon = "📚"
                message = "Significant preparation needed"
                gradient = "linear-gradient(145deg, #ef4444, #dc2626)"
            
            # Enhanced prediction display with animations
            st.markdown(f"""
            <div class="prediction-card" style="background: {gradient}; position: relative; overflow: hidden; border-radius: 32px;">
                <div style="position: relative; z-index: 2;">
                    <div style="font-size: 5rem; margin-bottom: 1.5rem; animation: bounceIn 1s ease-out;">{icon}</div>
                    <h1 style="margin: 0; font-size: 4rem; animation: fadeInUp 1s ease-out 0.2s both; font-weight: 800;">{success_prob*100:.1f}%</h1>
                    <h2 style="margin: 1rem 0; animation: fadeInUp 1s ease-out 0.4s both; font-size: 2rem; font-weight: 700;">{status} Admission Chances</h2>
                    <p style="margin: 1rem 0; font-size: 1.3rem; opacity: 0.95; animation: fadeInUp 1s ease-out 0.6s both; font-weight: 500;">{message}</p>
                    <p style="margin: 0; opacity: 0.9; animation: fadeInUp 1s ease-out 0.8s both; font-size: 1.1rem;">for {course_applied}</p>
                    <p style="margin: 0; opacity: 0.8; animation: fadeInUp 1s ease-out 1s both; font-size: 1rem;">AI Confidence: {confidence*100:.1f}%</p>
                </div>
                <div style="position: absolute; top: -50%; left: -50%; width: 200%; height: 200%; background: conic-gradient(from 0deg, transparent, rgba(255,255,255,0.1), transparent); animation: rotate 12s linear infinite; z-index: 1;"></div>
            </div>
            """, unsafe_allow_html=True)
            
            # Enhanced detailed analysis sections
            recommendations = prediction_result.get('recommendations', [])
            risk_factors = prediction_result.get('risk_factors', [])
            insights = prediction_result.get('insights', {})
            
            # Tabbed interface for detailed analysis
            st.markdown("## 📋 Detailed Analysis")
            
            tab1, tab2, tab3, tab4 = st.tabs(["💪 Strengths", "📈 Improvements", "💡 Recommendations", "⚠️ Risk Factors"])
            
            with tab1:
                st.markdown("### Your Application Strengths")
                strengths = insights.get('strengths', [])
                if strengths:
                    for i, strength in enumerate(strengths):
                        st.markdown(f"""
                        <div class="enhanced-card" style="animation-delay: {i*0.1}s; border-left: 6px solid #10b981;">
                            ✅ {strength}
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.info("💡 Continue building your profile to develop clear strengths")
            
            with tab2:
                st.markdown("### Areas for Improvement")
                improvements = insights.get('areas_for_improvement', [])
                if improvements:
                    for i, improvement in enumerate(improvements):
                        st.markdown(f"""
                        <div class="enhanced-card" style="animation-delay: {i*0.1}s; border-left: 6px solid #f59e0b;">
                            📌 {improvement}
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.success("🌟 Your profile is well-balanced!")
            
            with tab3:
                st.markdown("### Personalized Recommendations")
                if recommendations:
                    for i, rec in enumerate(recommendations):
                        st.markdown(f"""
                        <div class="enhanced-card" style="animation-delay: {i*0.1}s; border-left: 6px solid #3b82f6;">
                            **{i+1}.** {rec}
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.success("✅ Your application profile looks strong!")
            
            with tab4:
                st.markdown("### Risk Assessment")
                if risk_factors:
                    for i, risk in enumerate(risk_factors):
                        st.markdown(f"""
                        <div class="enhanced-card" style="animation-delay: {i*0.1}s; border-left: 6px solid #ef4444;">
                            ⚠️ {risk}
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.success("✅ No significant risk factors identified")
            
            # Store prediction result
            st.session_state.prediction_results[f"{first_name} {last_name}"] = {
                **prediction_result,
                'applicant_data': applicant_data,
                'course_info': selected_course_info.to_dict() if selected_course_info is not None else {},
                'timestamp': datetime.now().isoformat()
            }
            
            # Enhanced action buttons
            st.markdown("---")
            st.markdown("### 🚀 Next Steps")
            
            action_col1, action_col2, action_col3, action_col4 = st.columns(4)
            
            with action_col1:
                if st.button("🎯 Find Similar Courses", use_container_width=True):
                    st.session_state.user_profile = applicant_data
                    st.session_state.current_page = "recommendations"
                    st.rerun()
            
            with action_col2:
                if st.button("💬 Discuss with AI", use_container_width=True):
                    st.session_state.current_page = "ai_chat"
                    if selected_course_info is not None:
                        st.session_state.course_inquiry = course_applied
                    st.rerun()
            
            with action_col3:
                if st.button("📝 Start Application", use_container_width=True):
                    st.session_state.current_page = "applications"
                    st.rerun()
            
            with action_col4:
                if st.button("📄 Download Report", use_container_width=True):
                    # Generate downloadable report
                    report_content = f"""
COMPREHENSIVE ADMISSION PREDICTION REPORT
========================================

APPLICANT: {first_name} {last_name}
DATE: {datetime.now().strftime('%Y-%m-%d')}

PREDICTION RESULTS
-----------------
Success Probability: {success_prob*100:.1f}%
Status: {status}
AI Confidence: {confidence*100:.1f}%

TARGET COURSE
------------
Course: {course_applied}
Level: {selected_course_info.get('level', 'N/A') if selected_course_info is not None else 'N/A'}

ACADEMIC PROFILE
---------------
Current Education: {current_education}
GPA: {gpa}/4.0
IELTS: {ielts_score}/9.0
Work Experience: {work_experience} years

RECOMMENDATIONS
--------------
{chr(10).join(f'• {rec}' for rec in recommendations)}

Generated by UEL AI Prediction System
© University of East London
                    """
                    
                    st.download_button(
                        label="💾 Save Complete Report",
                        data=report_content,
                        file_name=f"admission_prediction_{first_name}_{last_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                        mime="text/plain"
                    )
        
        else:
            st.markdown(f"""
            <div class="enhanced-card error-animation" style="border-left: 6px solid #ef4444;">
                ❌ Prediction Error: {prediction_result.get('error', 'Unknown error')}
            </div>
            """, unsafe_allow_html=True)
    
    elif predict_button and not course_applied:
        st.error("❌ Please select a target course from the available options")
    elif predict_button:
        st.error("❌ Please fill in all required fields (Name and Course)")

# =============================================================================
# UTILITY FUNCTIONS FOR ENHANCED UI
# =============================================================================

def format_time(iso_string: str) -> str:
    """Format ISO timestamp to readable time"""
    try:
        dt = datetime.fromisoformat(iso_string.replace('Z', '+00:00'))
        return dt.strftime("%H:%M")
    except:
        return "now"

def render_sample_data_pages():
    """Render pages that use sample data with enhanced styling"""
    if st.session_state.current_page == "students":
        render_students_page()
    elif st.session_state.current_page == "courses":
        render_courses_page()
    elif st.session_state.current_page == "applications":
        render_applications_page()

def render_students_page():
    """Render enhanced student management page"""
    st.markdown("""
    <div class="page-header">
        <div class="header-icon">👥</div>
        <div>
            <h1 style="margin: 0; background: var(--uel-gradient); -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-weight: 800; font-size: 2.5rem;">Student Management</h1>
            <p style="margin: 0; color: var(--text-secondary); font-size: 1.2rem; font-weight: 500;">Comprehensive student profiles and academic tracking</p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="enhanced-card" style="text-align: center;">
        <h2 style="margin-top: 0; color: var(--uel-primary); font-weight: 700; font-size: 2rem;">👥 Student Management System</h2>
        <p style="font-size: 1.1rem; color: var(--text-primary);">This comprehensive student management interface would include:</p>
        <div style="margin-top: 2rem;">
            <div style="display: inline-block; padding: 1rem 1.5rem; background: rgba(0, 184, 169, 0.15); border-radius: 25px; margin: 0.5rem; border: 1px solid rgba(0, 184, 169, 0.3);">
                📝 Student Profiles
            </div>
            <div style="display: inline-block; padding: 1rem 1.5rem; background: rgba(0, 184, 169, 0.15); border-radius: 25px; margin: 0.5rem; border: 1px solid rgba(0, 184, 169, 0.3);">
                📊 Academic Tracking
            </div>
            <div style="display: inline-block; padding: 1rem 1.5rem; background: rgba(0, 184, 169, 0.15); border-radius: 25px; margin: 0.5rem; border: 1px solid rgba(0, 184, 169, 0.3);">
                📞 Communication Hub
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

def render_courses_page():
    """Render enhanced course explorer page"""
    st.markdown("""
    <div class="page-header">
        <div class="header-icon">🎓</div>
        <div>
            <h1 style="margin: 0; background: var(--uel-gradient); -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-weight: 800; font-size: 2.5rem;">Course Explorer</h1>
            <p style="margin: 0; color: var(--text-secondary); font-size: 1.2rem; font-weight: 500;">Discover and explore UEL's comprehensive course catalog</p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="enhanced-card" style="text-align: center;">
        <h2 style="margin-top: 0; color: var(--uel-primary); font-weight: 700; font-size: 2rem;">📚 Course Catalog System</h2>
        <p style="font-size: 1.1rem; color: var(--text-primary);">An interactive course explorer featuring:</p>
        <div style="margin-top: 2rem;">
            <div style="display: inline-block; padding: 1rem 1.5rem; background: rgba(0, 184, 169, 0.15); border-radius: 25px; margin: 0.5rem; border: 1px solid rgba(0, 184, 169, 0.3);">
                🔍 Advanced Search
            </div>
            <div style="display: inline-block; padding: 1rem 1.5rem; background: rgba(0, 184, 169, 0.15); border-radius: 25px; margin: 0.5rem; border: 1px solid rgba(0, 184, 169, 0.3);">
                📊 Course Comparison
            </div>
            <div style="display: inline-block; padding: 1rem 1.5rem; background: rgba(0, 184, 169, 0.15); border-radius: 25px; margin: 0.5rem; border: 1px solid rgba(0, 184, 169, 0.3);">
                💼 Career Pathways
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

def render_applications_page():
    """Render enhanced applications tracking page"""
    st.markdown("""
    <div class="page-header">
        <div class="header-icon">📝</div>
        <div>
            <h1 style="margin: 0; background: var(--uel-gradient); -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-weight: 800; font-size: 2.5rem;">Applications Tracking</h1>
            <p style="margin: 0; color: var(--text-secondary); font-size: 1.2rem; font-weight: 500;">Monitor and manage application progress in real-time</p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="enhanced-card" style="text-align: center;">
        <h2 style="margin-top: 0; color: var(--uel-primary); font-weight: 700; font-size: 2rem;">📋 Application Management</h2>
        <p style="font-size: 1.1rem; color: var(--text-primary);">A comprehensive application tracking system with:</p>
        <div style="margin-top: 2rem;">
            <div style="display: inline-block; padding: 1rem 1.5rem; background: rgba(0, 184, 169, 0.15); border-radius: 25px; margin: 0.5rem; border: 1px solid rgba(0, 184, 169, 0.3);">
                📊 Progress Tracking
            </div>
            <div style="display: inline-block; padding: 1rem 1.5rem; background: rgba(0, 184, 169, 0.15); border-radius: 25px; margin: 0.5rem; border: 1px solid rgba(0, 184, 169, 0.3);">
                📄 Document Status
            </div>
            <div style="display: inline-block; padding: 1rem 1.5rem; background: rgba(0, 184, 169, 0.15); border-radius: 25px; margin: 0.5rem; border: 1px solid rgba(0, 184, 169, 0.3);">
                🔔 Smart Notifications
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# =============================================================================
# ENHANCED MAIN APPLICATION ROUTING
# =============================================================================

def main():
    """Enhanced main application entry point"""
    # Setup enhanced Streamlit
    setup_streamlit()
    
    # Initialize session state
    initialize_session_state()
    
    # Add enhanced page transition animations
    st.markdown("""
    <style>
        .main > div {
            animation: pageSlideIn 0.5s ease-out;
        }
        
        @keyframes pageSlideIn {
            from {
                opacity: 0;
                transform: translateX(20px);
            }
            to {
                opacity: 1;
                transform: translateX(0);
            }
        }
        
        /* Loading overlay */
        .loading-overlay {
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: rgba(0, 184, 169, 0.1);
            display: flex;
            align-items: center;
            justify-content: center;
            z-index: 9999;
            animation: fadeOut 1s ease-out 2s forwards;
        }
        
        @keyframes fadeOut {
            to {
                opacity: 0;
                pointer-events: none;
            }
        }
        
        @keyframes rotate {
            from { transform: rotate(0deg); }
            to { transform: rotate(360deg); }
        }
        
        @keyframes progressLoad {
            from { width: 0%; }
            to { width: 85%; }
        }
    </style>
    """, unsafe_allow_html=True)

    # Initialize session state variables
    if 'chat_message_manager' not in st.session_state:
        st.session_state.chat_message_manager = None
    
    if 'chat_error_handler' not in st.session_state:
        st.session_state.chat_error_handler = None
    
    # Get AI agent early with error handling
    try:
        ai_agent = get_ai_agent()
    except Exception as e:
        st.error(f"⚠️ Failed to initialize AI system: {e}")
        st.info("💡 Please refresh the page or check your system configuration.")
        return
    
    # Render enhanced sidebar
    render_sidebar()
    
    # Enhanced page routing with error handling
    try:
        page = st.session_state.current_page
        
        # Route to appropriate enhanced page
        if page == "dashboard":
            render_dashboard()
        elif page == "ai_chat":
            render_chat_page()
        elif page == "predictions":
            render_predictions_page()
        elif page == "recommendations":
            # Add a simple placeholder for recommendations page
            st.markdown("""
            <div class="page-header">
                <div class="header-icon">🎯</div>
                <div>
                    <h1 style="margin: 0; background: var(--uel-gradient); -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-weight: 800; font-size: 2.5rem;">Course Recommendations</h1>
                    <p style="margin: 0; color: var(--text-secondary); font-size: 1.2rem; font-weight: 500;">AI-powered personalized course matching</p>
                </div>
            </div>
            """, unsafe_allow_html=True)
            st.info("🚧 Recommendations page is being enhanced. Please use the original system for full functionality.")
        elif page == "documents":
            # Add a simple placeholder for documents page
            st.markdown("""
            <div class="page-header">
                <div class="header-icon">📄</div>
                <div>
                    <h1 style="margin: 0; background: var(--uel-gradient); -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-weight: 800; font-size: 2.5rem;">Document Verification</h1>
                    <p style="margin: 0; color: var(--text-secondary); font-size: 1.2rem; font-weight: 500;">AI-powered document analysis</p>
                </div>
            </div>
            """, unsafe_allow_html=True)
            st.info("🚧 Document verification page is being enhanced. Please use the original system for full functionality.")
        elif page == "analytics":
            # Add a simple placeholder for analytics page
            st.markdown("""
            <div class="page-header">
                <div class="header-icon">📊</div>
                <div>
                    <h1 style="margin: 0; background: var(--uel-gradient); -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-weight: 800; font-size: 2.5rem;">Advanced Analytics</h1>
                    <p style="margin: 0; color: var(--text-secondary); font-size: 1.2rem; font-weight: 500;">Comprehensive insights and metrics</p>
                </div>
            </div>
            """, unsafe_allow_html=True)
            st.info("🚧 Analytics page is being enhanced. Please use the original system for full functionality.")
        elif page in ["students", "courses", "applications"]:
            render_sample_data_pages()
        else:
            # Default to dashboard for unknown pages
            st.session_state.current_page = "dashboard"
            st.rerun()
    
    except Exception as e:
        st.error(f"❌ Page rendering error: {e}")
        
        # Enhanced error display
        st.markdown(f"""
        <div class="enhanced-card error-animation" style="border-left: 6px solid #ef4444; text-align: center;">
            <h3 style="color: #ef4444; margin-top: 0; font-weight: 700;">🔧 System Error</h3>
            <p style="font-size: 1.1rem;">An error occurred while loading the page. Please try the following:</p>
            <ul style="text-align: left; display: inline-block; font-size: 1rem;">
                <li>Refresh the page</li>
                <li>Clear your browser cache</li>
                <li>Check your internet connection</li>
                <li>Contact support if the issue persists</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        # Provide quick navigation back to dashboard
        if st.button("🏠 Return to Dashboard", type="primary", use_container_width=True):
            st.session_state.current_page = "dashboard"
            st.rerun()

if __name__ == "__main__":
    main()