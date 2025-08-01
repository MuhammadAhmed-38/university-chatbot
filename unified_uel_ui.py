# Complete UEL AI Assistant - Full Feature Implementation with Authentication
# Enhanced unified_uel_ui.py with ALL original features + password system + bug fixes

import streamlit as st
import pandas as pd
import numpy as np
import json
import threading
import logging
import time
import random
import base64
import hashlib
import io
import zipfile
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import scipy.stats as stats
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
import seaborn as sns



# Import from enhanced system
from unified_uel_ai_system import (
    UELAISystem, UserProfile, ProfileManager,
    format_currency, format_date, format_duration, 
    get_status_color, get_level_color, generate_sample_data,
    config, PLOTLY_AVAILABLE, get_logger
)

# Create module logger
logger = get_logger(__name__)

# =============================================================================
# PERSISTENT STORAGE SYSTEM WITH AUTHENTICATION
# =============================================================================

def hash_password(password: str) -> str:
    """Hash password for storage"""
    return hashlib.sha256(password.encode()).hexdigest()

def save_profile_to_storage(profile_data: Dict, password: str):
    """Save profile with password protection to session state"""
    try:
        hashed_password = hash_password(password)
        storage_data = {
            'profile': profile_data,
            'password_hash': hashed_password,
            'created_date': datetime.now().isoformat(),
            'last_access': datetime.now().isoformat()
        }
        
        profile_id = profile_data.get('id', 'default_profile')
        st.session_state[f'stored_profile_{profile_id}'] = storage_data
        st.session_state['last_saved_profile_id'] = profile_id
        
        return True
    except Exception as e:
        logger.error(f"Profile storage error: {e}")
        return False

def load_profile_from_storage(profile_id: str, password: str) -> Optional[Dict]:
    """Load profile with password verification"""
    try:
        storage_key = f'stored_profile_{profile_id}'
        if storage_key in st.session_state:
            stored_data = st.session_state[storage_key]
            hashed_password = hash_password(password)
            
            if stored_data['password_hash'] == hashed_password:
                stored_data['last_access'] = datetime.now().isoformat()
                return stored_data['profile']
        return None
    except Exception as e:
        logger.error(f"Profile loading error: {e}")
        return None

def get_available_profiles() -> List[str]:
    """Get list of available stored profiles"""
    profiles = []
    for key in st.session_state.keys():
        if key.startswith('stored_profile_'):
            profile_id = key.replace('stored_profile_', '')
            profiles.append(profile_id)
    return profiles

def render_profile_login_dialog():
    """Render profile login dialog"""
    if st.session_state.get('show_profile_login', False):
        st.markdown("""
        <div class="enhanced-card" style="background: var(--uel-gradient); color: white; text-align: center;">
            <h3 style="margin-top: 0;">🔐 Login to Your Profile</h3>
            <p>Enter your credentials to access your saved profile</p>
        </div>
        """, unsafe_allow_html=True)
        
        available_profiles = get_available_profiles()
        
        if available_profiles:
            with st.form("profile_login_form"):
                if len(available_profiles) == 1:
                    selected_profile = available_profiles[0]
                    profile_data = st.session_state.get(f'stored_profile_{selected_profile}', {})
                    profile_info = profile_data.get('profile', {})
                    st.info(f"Logging into: {profile_info.get('first_name', '')} {profile_info.get('last_name', '')}")
                else:
                    profile_options = []
                    for pid in available_profiles:
                        profile_data = st.session_state.get(f'stored_profile_{pid}', {})
                        profile_info = profile_data.get('profile', {})
                        name = f"{profile_info.get('first_name', '')} {profile_info.get('last_name', '')}"
                        profile_options.append((f"{name} ({pid})", pid))
                    
                    selected_index = st.selectbox("Select Profile", range(len(profile_options)), 
                                                format_func=lambda x: profile_options[x][0])
                    selected_profile = profile_options[selected_index][1]
                
                password = st.text_input("Password", type="password", help="Enter your profile password")
                
                col1, col2 = st.columns(2)
                with col1:
                    login_submitted = st.form_submit_button("🔐 Login", type="primary", use_container_width=True)
                with col2:
                    cancel_login = st.form_submit_button("❌ Cancel", use_container_width=True)
                
                if login_submitted and password:
                    profile_data = load_profile_from_storage(selected_profile, password)
                    if profile_data:
                        st.session_state.current_profile = profile_data
                        st.session_state.profile_active = True
                        st.session_state.show_profile_login = False
                        st.success("✅ Login successful! Welcome back!")
                        time.sleep(1)
                        st.rerun()
                    else:
                        st.error("❌ Invalid password or profile not found")
                
                if cancel_login:
                    st.session_state.show_profile_login = False
                    st.rerun()
        else:
            st.info("📝 No saved profiles found. Create a new profile to get started.")
            if st.button("Create New Profile", type="primary", key="create_new_profile_3000"):
                st.session_state.show_profile_login = False
                st.session_state.current_page = 'profile_setup'
                st.rerun()

# =============================================================================
# ENHANCED SESSION STATE MANAGEMENT WITH FULL FEATURE SUPPORT
# =============================================================================

def initialize_comprehensive_session_state():
    """Initialize comprehensive session state with all features"""
    defaults = {
        # Profile Management (Central to everything)
        'profile_active': False,
        'current_profile': None,
        'profile_completion': 0.0,
        'profile_setup_step': 1,
        'profile_required_shown': False,
        'profile_edit_mode': False,
        'profile_backup': None,
        'show_profile_login': False,
        
      
        # Interview preparation state
        'active_interview_session': None,
        'interview_settings': {},
        'practice_questions': [],
        'current_practice_index': 0,
        'practice_responses': [],
        'current_voice_response': '',
        'recording_active': False,
        'interview_history': [],
        'interview_performance_data': {},
   

        # Navigation and UI State
        'current_page': 'dashboard',
        'last_activity': datetime.now(),
        'profile_last_updated': None,
        'sidebar_collapsed': False,
        'theme_mode': 'light',
        
        # User session and persistence
        'user_id': f"user_{int(time.time())}",
        'session_start_time': datetime.now(),
        'session_data': {},
        'auto_save_enabled': True,
        'last_save_time': datetime.now(),
        
        # AI interactions (profile-aware)
        'messages': [],
        'interaction_history': [],
        'sentiment_history': [],
        'conversation_context': [],
        'ai_response_cache': {},
        'voice_enabled': False,
        
        # Feature state (profile-integrated)
        'prediction_results': {},
        'recommendation_cache': {},
        'document_uploads': {},
        'verification_results': {},
        'search_filters': {},
        'export_queue': [],
        
        # System state
        'ai_agent': None,
        'profile_manager': None,
        'feature_usage_stats': {},
        'error_log': [],
        'performance_metrics': {},
        
        # UI state management
        'show_profile_editor': False,
        'show_profile_completion_tips': False,
        'show_document_verifier': False,
        'show_voice_settings': False,
        'show_export_dialog': False,
        'show_analytics_filters': False,
        'feature_access_attempts': 0,
        
        # Advanced features
        'comparison_mode': False,
        'selected_courses_for_comparison': [],
        'notification_preferences': {
            'email': True,
            'push': False,
            'sms': False
        },
        
        # Onboarding and help
        'onboarding_completed': False,
        'welcome_shown': False,
        'feature_tour_completed': {},
        'help_topics_viewed': []
    }
    
    for key, default_value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default_value

# =============================================================================
# COMPREHENSIVE PROFILE EDITOR SYSTEM (RESTORED)
# =============================================================================

def check_profile_access(feature_name: str, flexible: bool = True) -> bool:
    """Check if the user has access to a specific feature based on profile status."""
    try:
        if not st.session_state.get('profile_active', False):
            if flexible:
                st.info(f"💡 {feature_name} is available in basic mode. Create a profile for enhanced personalized features!")
                col1, col2 = st.columns(2)
                with col1:
                    if st.button(f"📝 Create Profile", key=f"create_profile_{feature_name}", use_container_width=True):
                        st.session_state.current_page = 'profile_setup'
                        st.rerun()
                with col2:
                    if st.button(f"🔐 Login", key=f"login_profile_{feature_name}", use_container_width=True):
                        st.session_state.show_profile_login = True
                        st.rerun()
                return True  # Allow access in basic mode
            else:
                st.error(f"❌ Access to {feature_name} requires an active profile. Please complete your profile setup.")
                col1, col2 = st.columns(2)
                with col1:
                    if st.button(f"📝 Create Profile", key=f"create_profile_{feature_name}", use_container_width=True):
                        st.session_state.current_page = 'profile_setup'
                        st.rerun()
                with col2:
                    if st.button(f"🔐 Login", key=f"login_profile_{feature_name}", use_container_width=True):
                        st.session_state.show_profile_login = True
                        st.rerun()
                return False
        
        profile = st.session_state.get('current_profile')
        if not profile:
            st.error(f"❌ No profile found. Please complete your profile to access {feature_name}.")
            return False
        
        completion_threshold = 60.0  # Require at least 60% profile completion
        completion = profile.get('profile_completion', 0.0)
        
        if completion < completion_threshold:
            st.warning(f"⚠️ Your profile is only {completion:.0f}% complete. Complete your profile to at least {completion_threshold}% to access {feature_name}.")
            st.button("📝 Complete Profile Now", key=f"complete_profile_{feature_name}", on_click=lambda: set_page('profile_setup'))
            return False
        
        # Increment feature access attempts
        st.session_state.feature_access_attempts = st.session_state.get('feature_access_attempts', 0) + 1
        return True
    
    except Exception as e:
        st.error(f"❌ Error checking profile access for {feature_name}: {str(e)}")
        logger.error(f"Profile access check error for {feature_name}: {str(e)}")
        return False

def set_page(page: str):
    """Helper function to set the current page and rerun."""
    st.session_state.current_page = page
    st.rerun()

def render_sidebar_profile_section():
    """Render the profile section in the sidebar"""
    with st.sidebar:
        st.markdown("""
        <div style="padding: 1rem; border-bottom: 1px solid #e2e8f0;">
            <h3 style="margin: 0; color: var(--uel-primary); font-size: 1.5rem;">👤 Your Profile</h3>
        </div>
        """, unsafe_allow_html=True)
        
        if st.session_state.current_profile is None:
            st.warning("⚠️ No profile found. Please complete your profile setup.")
            col1, col2 = st.columns(2)
            with col1:
                if st.button("📝 Create", key="sidebar_setup_profile", use_container_width=True):
                    st.session_state.current_page = 'profile_setup'
                    st.rerun()
            with col2:
                if st.button("🔐 Login", key="sidebar_login_profile", use_container_width=True):
                    st.session_state.show_profile_login = True
                    st.rerun()
            return
        
        profile = st.session_state.current_profile
        first_name = profile.get('first_name', 'User')
        completion = profile.get('profile_completion', 0)
        
        st.markdown(f"""
        <div class="enhanced-card" style="background: linear-gradient(135deg, rgba(0, 184, 169, 0.1), rgba(0, 184, 169, 0.05)); padding: 1rem; border-radius: 10px; margin: 0.5rem 0;">
            <div style="display: flex; align-items: center; justify-content: space-between;">
                <span style="font-weight: 600; color: var(--uel-primary);">{first_name}</span>
                <span style="font-size: 0.9rem; color: var(--text-secondary);">{completion:.0f}% Complete</span>
            </div>
            <div style="background: #e2e8f0; border-radius: 8px; height: 8px; margin: 0.5rem 0;">
                <div style="background: var(--uel-primary); height: 100%; border-radius: 8px; width: {completion}%; transition: width 0.3s ease;"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("✏️ Edit Profile", key="sidebar_edit_profile", use_container_width=True):
                st.session_state.show_profile_editor = True
                st.session_state.current_page = 'profile_setup'
                st.rerun()
        with col2:
            if st.button("🚪 Logout", key="sidebar_logout", use_container_width=True):
                st.session_state.current_profile = None
                st.session_state.profile_active = False
                st.session_state.current_page = 'dashboard'
                st.session_state.messages = []
                st.rerun()

def render_profile_status_bar():
    """Render a profile status bar showing completion percentage"""
    if st.session_state.current_profile is None:
        st.info("💡 Welcome to UEL AI Assistant! Create a profile for personalized features.")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("📝 Create Profile", use_container_width=True, key="__create_profile_3001"):
                st.session_state.current_page = 'profile_setup'
                st.rerun()
        with col2:
            if st.button("🔐 Login to Profile", use_container_width=True, key="__login_to_profile_3002"):
                st.session_state.show_profile_login = True
                st.rerun()
        return
    
    profile = st.session_state.current_profile
    completion = profile.get('profile_completion', 0)
    
    st.markdown(f"""
    <div class="enhanced-card" style="background: linear-gradient(135deg, rgba(0, 184, 169, 0.1), rgba(0, 184, 169, 0.05)); padding: 1rem; border-radius: 10px; margin-bottom: 1rem;">
        <h3 style="margin: 0; color: var(--uel-primary); font-size: 1.2rem;">📊 Profile Completion: {completion:.0f}%</h3>
        <div style="background: #e2e8f0; border-radius: 8px; height: 12px; margin: 0.5rem 0;">
            <div style="background: var(--uel-primary); height: 100%; border-radius: 8px; width: {completion}%; transition: width 0.3s ease;"></div>
        </div>
        <p style="font-size: 0.9rem; color: var(--text-secondary); margin: 0;">
            Complete your profile for better recommendations and predictions
        </p>
    </div>
    """, unsafe_allow_html=True)

def render_comprehensive_profile_editor():
    """Render complete profile editing interface"""
    st.markdown("""
    <div class="page-header">
        <div class="header-icon">✏️</div>
        <div>
            <h1 style="margin: 0; background: var(--uel-gradient); -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-weight: 800; font-size: 2.5rem;">Edit Your Profile</h1>
            <p style="margin: 0; color: var(--text-secondary); font-size: 1.2rem; font-weight: 500;">Update your information to get better recommendations</p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    profile = st.session_state.current_profile
    
    # Profile completion progress
    completion = profile.get('profile_completion', 0)
    st.markdown(f"""
    <div class="enhanced-card" style="background: linear-gradient(135deg, rgba(0, 184, 169, 0.1), rgba(0, 184, 169, 0.05));">
        <h3 style="margin-top: 0; color: var(--uel-primary);">📊 Profile Completion: {completion:.0f}%</h3>
        <div style="background: #e2e8f0; border-radius: 10px; height: 20px; margin: 1rem 0;">
            <div style="background: var(--uel-primary); height: 100%; border-radius: 10px; width: {completion}%; transition: width 0.3s ease;"></div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Tabbed editing interface
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "👤 Basic Info", 
        "🎓 Academic", 
        "🎯 Interests", 
        "💼 Professional", 
        "⚙️ Preferences"
    ])
    
    with tab1:
        render_basic_info_editor(profile)
    
    with tab2:
        render_academic_info_editor(profile)
    
    with tab3:
        render_interests_editor(profile)
    
    with tab4:
        render_professional_editor(profile)
    
    with tab5:
        render_preferences_editor(profile)
    
    with tab6:
        render_research_evaluation_interface()
    
    # Profile actions
    st.markdown("---")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("💾 Save Changes", type="primary", use_container_width=True, key="__save_changes_3003"):
            save_profile_changes()
    
    with col2:
        if st.button("🔄 Reset Changes", use_container_width=True, key="__reset_changes_3004"):
            reset_profile_changes()
    
    with col3:
        if st.button("📤 Export Profile", use_container_width=True, key="__export_profile_3005"):
            export_profile_data()
    
    with col4:
        if st.button("🗑️ Delete Profile", use_container_width=True, key="___delete_profile_3006"):
            show_delete_profile_confirmation()

def render_basic_info_editor(profile: Dict):
    """Render basic information editing section"""
    st.markdown("### 👤 Personal Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.text_input(
            "First Name *",
            value=profile.get('first_name', ''),
            key="edit_first_name",
            help="Your legal first name"
        )
        
        st.text_input(
            "Last Name *",
            value=profile.get('last_name', ''),
            key="edit_last_name",
            help="Your legal last name"
        )
        
        st.text_input(
            "Email Address",
            value=profile.get('email', ''),
            key="edit_email",
            help="Your primary email address"
        )
        
        st.text_input(
            "Phone Number",
            value=profile.get('phone', ''),
            key="edit_phone",
            help="Your contact phone number"
        )
    
    with col2:
        countries = ["", "United Kingdom", "United States", "India", "China", "Nigeria", 
                    "Pakistan", "Canada", "Germany", "France", "Australia", "Brazil", "Other"]
        
        current_country_index = 0
        if profile.get('country') in countries:
            current_country_index = countries.index(profile.get('country'))
        
        st.selectbox(
            "Country of Residence *",
            countries,
            index=current_country_index,
            key="edit_country",
            help="Your current country of residence"
        )
        
        st.text_input(
            "Nationality",
            value=profile.get('nationality', ''),
            key="edit_nationality",
            help="Your nationality"
        )
        
        st.text_input(
            "City",
            value=profile.get('city', ''),
            key="edit_city",
            help="Your current city"
        )
        
        st.text_input(
            "Postal Code",
            value=profile.get('postal_code', ''),
            key="edit_postal_code",
            help="Your postal/zip code"
        )
    
    # Date of birth with validation
    current_dob = profile.get('date_of_birth')
    if current_dob:
        try:
            if isinstance(current_dob, str):
                dob_value = datetime.fromisoformat(current_dob).date()
            else:
                dob_value = current_dob
        except:
            dob_value = None
    else:
        dob_value = None
    
    st.date_input(
        "Date of Birth",
        value=dob_value,
        min_value=datetime(1950, 1, 1).date(),
        max_value=datetime.now().date(),
        key="edit_date_of_birth",
        help="Your date of birth (used for age calculations)"
    )

def render_academic_info_editor(profile: Dict):
    """Render academic information editing section"""
    st.markdown("### 🎓 Academic Background")
    
    col1, col2 = st.columns(2)
    
    with col1:
        academic_levels = ["", "High School", "High School Graduate", "Undergraduate Student", 
                          "Bachelor's Graduate", "Postgraduate Student", "Master's Graduate", 
                          "PhD Student", "PhD Graduate", "Working Professional"]
        
        current_level_index = 0
        if profile.get('academic_level') in academic_levels:
            current_level_index = academic_levels.index(profile.get('academic_level'))
        
        st.selectbox(
            "Current Academic Level *",
            academic_levels,
            index=current_level_index,
            key="edit_academic_level",
            help="Your current educational status"
        )
        
        st.text_input(
            "Current/Previous Institution",
            value=profile.get('current_institution', ''),
            key="edit_current_institution",
            help="Name of your educational institution"
        )
        
        st.text_input(
            "Current/Previous Major/Subject",
            value=profile.get('current_major', ''),
            key="edit_current_major",
            help="Your field of study"
        )
        
        st.number_input(
            "Expected/Actual Graduation Year",
            min_value=2020,
            max_value=2035,
            value=int(profile.get('graduation_year', datetime.now().year)),
            key="edit_graduation_year",
            help="When you graduated or expect to graduate"
        )
    
    with col2:
        st.number_input(
            "GPA/Grade Average (0.0-4.0)",
            min_value=0.0,
            max_value=4.0,
            value=float(profile.get('gpa', 3.0)),
            step=0.1,
            key="edit_gpa",
            help="Your cumulative GPA on a 4.0 scale"
        )
        
        st.number_input(
            "IELTS Score",
            min_value=0.0,
            max_value=9.0,
            value=float(profile.get('ielts_score', 6.5)),
            step=0.5,
            key="edit_ielts_score",
            help="Your IELTS English proficiency score"
        )
        
        st.number_input(
            "TOEFL Score",
            min_value=0,
            max_value=120,
            value=int(profile.get('toefl_score', 0)),
            key="edit_toefl_score",
            help="Your TOEFL score (if taken)"
        )
        
        st.text_input(
            "Other English Certifications",
            value=profile.get('other_english_cert', ''),
            key="edit_other_english_cert",
            help="Other English language certifications"
        )
    
    # Additional academic information
    st.markdown("#### 📚 Additional Academic Details")
    
    col3, col4 = st.columns(2)
    
    with col3:
        previous_applications = profile.get('previous_applications', [])
        if isinstance(previous_applications, list):
            previous_apps_text = ', '.join(previous_applications)
        else:
            previous_apps_text = str(previous_applications) if previous_applications else ''
        
        st.text_area(
            "Previous University Applications",
            value=previous_apps_text,
            key="edit_previous_applications",
            help="List any previous university applications (comma-separated)"
        )
    
    with col4:
        rejected_courses = profile.get('rejected_courses', [])
        if isinstance(rejected_courses, list):
            rejected_text = ', '.join(rejected_courses)
        else:
            rejected_text = str(rejected_courses) if rejected_courses else ''
        
        st.text_area(
            "Previously Rejected Courses",
            value=rejected_text,
            key="edit_rejected_courses",
            help="Courses where applications were unsuccessful"
        )

def render_interests_editor(profile: Dict):
    """Render interests and goals editing section"""
    st.markdown("### 🎯 Interests & Career Goals")
    
    # Primary field of interest
    st.text_input(
        "Primary Field of Interest *",
        value=profile.get('field_of_interest', ''),
        key="edit_field_of_interest",
        help="Your main academic interest area"
    )
    
    # Multiple interests selection
    available_interests = [
        "Artificial Intelligence", "Data Science", "Cybersecurity", "Web Development", "Software Engineering",
        "Business Management", "Finance", "Marketing", "Human Resources", "Entrepreneurship",
        "Psychology", "Counseling", "Social Work", "Education", "Research",
        "Engineering", "Architecture", "Design", "Arts", "Creative Writing",
        "Medicine", "Nursing", "Healthcare", "Biology", "Chemistry",
        "Law", "International Relations", "Politics", "History", "Philosophy",
        "Media Studies", "Communications", "Journalism", "Film Production"
    ]
    
    current_interests = profile.get('interests', [])
    if not isinstance(current_interests, list):
        current_interests = []
    
    st.multiselect(
        "Additional Areas of Interest",
        available_interests,
        default=current_interests,
        key="edit_interests",
        help="Select all areas that interest you"
    )
    
    # Career goals and aspirations
    col1, col2 = st.columns(2)
    
    with col1:
        st.text_area(
            "Career Goals & Aspirations",
            value=profile.get('career_goals', ''),
            key="edit_career_goals",
            height=150,
            help="Describe your career aspirations and goals"
        )
        
        target_industries = ["", "Technology", "Finance", "Healthcare", "Education", "Government",
                           "Non-Profit", "Media & Entertainment", "Sports", "Retail", "Manufacturing",
                           "Consulting", "Startups", "Research & Development", "Other"]
        
        current_industry_index = 0
        if profile.get('target_industry') in target_industries:
            current_industry_index = target_industries.index(profile.get('target_industry'))
        
        st.selectbox(
            "Target Industry",
            target_industries,
            index=current_industry_index,
            key="edit_target_industry",
            help="Your preferred industry to work in"
        )
    
    with col2:
        st.text_area(
            "Skills & Competencies",
            value=', '.join(profile.get('professional_skills', [])) if profile.get('professional_skills') else '',
            key="edit_professional_skills",
            height=150,
            help="List your key skills (comma-separated)"
        )
        
        preferred_courses = profile.get('preferred_courses', [])
        if isinstance(preferred_courses, list):
            preferred_text = ', '.join(preferred_courses)
        else:
            preferred_text = str(preferred_courses) if preferred_courses else ''
        
        st.text_area(
            "Preferred Courses/Programs",
            value=preferred_text,
            key="edit_preferred_courses",
            help="Courses you're specifically interested in"
        )

def render_professional_editor(profile: Dict):
    """Render professional background editing section"""
    st.markdown("### 💼 Professional Background")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.number_input(
            "Years of Work Experience",
            min_value=0,
            max_value=50,
            value=int(profile.get('work_experience_years', 0)),
            key="edit_work_experience_years",
            help="Total years of relevant work experience"
        )
        
        st.text_input(
            "Current Job Title",
            value=profile.get('current_job_title', ''),
            key="edit_current_job_title",
            help="Your current or most recent job title"
        )
        
        st.text_area(
            "Work Experience Description",
            value=profile.get('work_experience_description', ''),
            key="edit_work_experience_description",
            height=100,
            help="Brief description of your work experience"
        )
    
    with col2:
        st.text_input(
            "Current/Previous Company",
            value=profile.get('current_company', ''),
            key="edit_current_company",
            help="Name of your current or previous employer"
        )
        
        employment_status = st.selectbox(
            "Employment Status",
            ["", "Student", "Employed Full-time", "Employed Part-time", "Self-employed", "Unemployed", "Retired"],
            index=0,
            key="edit_employment_status",
            help="Your current employment situation"
        )
        
        st.text_area(
            "Professional Achievements",
            value=profile.get('professional_achievements', ''),
            key="edit_professional_achievements",
            height=100,
            help="Key achievements in your career"
        )

def render_preferences_editor(profile: Dict):
    """Render preferences and settings editing section"""
    st.markdown("### ⚙️ Study Preferences & Settings")
    
    col1, col2 = st.columns(2)
    
    with col1:
        study_modes = ["Full-time", "Part-time", "Online", "Flexible/Hybrid"]
        current_mode_index = 0
        if profile.get('preferred_study_mode') in study_modes:
            current_mode_index = study_modes.index(profile.get('preferred_study_mode'))
        
        st.selectbox(
            "Preferred Study Mode",
            study_modes,
            index=current_mode_index,
            key="edit_preferred_study_mode",
            help="How you prefer to study"
        )
        
        budget_ranges = ["Not sure", "Under £10,000", "£10,000 - £15,000", "£15,000 - £20,000", "Over £20,000"]
        current_budget_index = 0
        if profile.get('budget_range') in budget_ranges:
            current_budget_index = budget_ranges.index(profile.get('budget_range'))
        
        st.selectbox(
            "Budget Range (annual)",
            budget_ranges,
            index=current_budget_index,
            key="edit_budget_range",
            help="Your budget for tuition fees"
        )
        
        st.text_input(
            "Preferred Start Date",
            value=profile.get('preferred_start_date', ''),
            key="edit_preferred_start_date",
            help="When you'd like to start your studies"
        )
    
    with col2:
        st.markdown("#### 🔔 Notification Preferences")
        
        st.checkbox(
            "Email Notifications",
            value=profile.get('ai_preferences', {}).get('email_notifications', True),
            key="edit_email_notifications",
            help="Receive email updates and reminders"
        )
        
        st.checkbox(
            "AI Personalization",
            value=profile.get('ai_preferences', {}).get('personalization_enabled', True),
            key="edit_personalization_enabled",
            help="Allow AI to personalize responses based on your profile"
        )
        
        st.checkbox(
            "Data Analytics",
            value=profile.get('ai_preferences', {}).get('analytics_enabled', True),
            key="edit_analytics_enabled",
            help="Enable data collection for improving recommendations"
        )
        
        st.selectbox(
            "Preferred Language",
            ["English", "Spanish", "French", "German", "Mandarin", "Arabic"],
            index=0,
            key="edit_preferred_language",
            help="Your preferred language for communication"
        )



# =============================================================================
# INTERVIEW PREPARATION CENTER (NEW FEATURE)
# =============================================================================

def render_interview_preparation_center():
    """Render comprehensive interview preparation center"""
    # Check profile access
    if not check_profile_access("Interview Preparation", flexible=False):
        return
    
    st.markdown("""
    <div class="page-header">
        <div class="header-icon">🎤</div>
        <div>
            <h1 style="margin: 0; background: var(--uel-gradient); -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-weight: 800; font-size: 2.5rem;">Interview Preparation Center</h1>
            <p style="margin: 0; color: var(--text-secondary); font-size: 1.2rem; font-weight: 500;">AI-powered mock interviews and preparation tools</p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Get AI agent
    ai_agent = get_enhanced_ai_agent()
    
    # Check if interview system is available
    if not hasattr(ai_agent, 'interview_system') or ai_agent.interview_system is None:
        st.error("❌ Interview preparation system not available. Please ensure the enhanced system is properly initialized.")
        return
    
    profile = st.session_state.current_profile
    
    # Personalized welcome
    st.markdown(f"""
    <div class="enhanced-card" style="background: linear-gradient(135deg, rgba(139, 92, 246, 0.1), rgba(139, 92, 246, 0.05)); border-left: 6px solid #8b5cf6;">
        <h3 style="margin-top: 0; color: #8b5cf6;">🎯 Personalized Interview Preparation for {profile.get('first_name', 'Student')}</h3>
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 1rem; margin-top: 1rem;">
            <div><strong>Program Interest:</strong> {profile.get('field_of_interest', 'Not specified')}</div>
            <div><strong>Academic Level:</strong> {profile.get('academic_level', 'Not specified')}</div>
            <div><strong>Target Universities:</strong> University of East London</div>
            <div><strong>Preparation Level:</strong> {'Advanced' if profile.get('interaction_count', 0) > 20 else 'Intermediate' if profile.get('interaction_count', 0) > 5 else 'Beginner'}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Main tabs for interview preparation
    tab1, tab2, tab3, tab4, tab5, = st.tabs([
        "🚀 Start Mock Interview", 
        "📚 Question Practice", 
        "📊 Performance Analytics", 
        "💡 Interview Tips", 
        "📋 Interview History"
        
    ])
    
    with tab1:
        render_mock_interview_interface(ai_agent.interview_system, profile)
    
    with tab2:
        render_question_practice_interface(ai_agent.interview_system, profile)
    
    with tab3:
        render_interview_analytics(ai_agent.interview_system, profile)
    
    with tab4:
        render_interview_tips_guidance(ai_agent.interview_system, profile)
    
    with tab5:
        render_interview_history(ai_agent.interview_system, profile)

def render_mock_interview_interface(interview_system, profile: Dict):
    """Render mock interview interface"""
    st.markdown("### 🚀 AI-Powered Mock Interview")
    
    # Check for active interview session
    active_interview = st.session_state.get('active_interview_session')
    
    if not active_interview:
        # Interview setup
        st.markdown("#### 🎯 Customize Your Mock Interview")
        
        col1, col2 = st.columns(2)
        
        with col1:
            interview_type = st.selectbox(
                "Interview Type",
                ["undergraduate_admission", "postgraduate_admission", "subject_specific", "scholarship_interview"],
                format_func=lambda x: x.replace('_', ' ').title(),
                help="Choose the type of interview you want to practice"
            )
            
            include_voice = st.checkbox(
                "🎤 Use Voice Responses",
                value=False,
                help="Practice speaking your answers aloud (requires microphone)"
            )
        
        with col2:
            difficulty_level = st.selectbox(
                "Difficulty Level",
                ["Beginner", "Intermediate", "Advanced"],
                index=1,
                help="Choose based on your interview experience"
            )
            
            focus_area = st.selectbox(
                "Focus Area",
                ["General Questions", "Subject-Specific", "Behavioral Questions", "Research Questions"],
                help="What type of questions to emphasize"
            )
        
        # Interview preparation tips
        st.markdown("""
        <div class="enhanced-card" style="background: rgba(34, 197, 94, 0.1); border-left: 4px solid #22c55e;">
            <h4 style="margin-top: 0; color: #22c55e;">📋 Before You Start</h4>
            <ul>
                <li>Find a quiet space with good lighting</li>
                <li>Test your microphone if using voice responses</li>
                <li>Have a notepad ready for jotting down thoughts</li>
                <li>Treat this like a real interview - dress appropriately</li>
                <li>Take your time to think before answering</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        # Start interview button
        if st.button("🎤 Start Mock Interview", type="primary", use_container_width=True, key="start_mock_interview"):
            with st.spinner("🤖 AI is preparing your personalized interview..."):
                interview_session = interview_system.create_personalized_interview(profile)
                
                if 'error' not in interview_session:
                    # Start the interview
                    start_result = interview_system.start_interview_session(interview_session['id'])
                    
                    if 'error' not in start_result:
                        st.session_state.active_interview_session = interview_session['id']
                        st.session_state.interview_settings = {
                            'use_voice': include_voice,
                            'difficulty': difficulty_level,
                            'focus_area': focus_area
                        }
                        st.success("🎉 Interview session started! Get ready for your first question.")
                        st.rerun()
                    else:
                        st.error(f"❌ Failed to start interview: {start_result.get('error')}")
                else:
                    st.error(f"❌ Failed to create interview: {interview_session.get('error')}")
    
    else:
        # Active interview session
        render_active_interview_session(interview_system, active_interview)

def render_active_interview_session(interview_system, interview_id: str):
    """Render active interview session interface"""
    st.markdown("### 🎤 Interview in Progress")
    
    # Get current question
    question_data = interview_system.get_next_question(interview_id)
    
    if question_data.get('status') == 'completed':
        st.success("🎉 Interview completed! Generating your performance report...")
        
        # Complete interview and show results
        completion_result = interview_system._complete_interview(interview_id)
        
        if 'error' not in completion_result:
            render_interview_completion_results(completion_result)
            
            # Clear active session
            if 'active_interview_session' in st.session_state:
                del st.session_state.active_interview_session
            
            if st.button("🔄 Start New Interview", type="primary"):
                st.rerun()
        else:
            st.error(f"❌ Error completing interview: {completion_result.get('error')}")
        
        return
    
    if 'error' in question_data:
        st.error(f"❌ Error getting question: {question_data.get('error')}")
        return
    
    # Display current question
    question = question_data['question']
    question_num = question_data['question_number']
    total_questions = question_data['total_questions']
    time_remaining = question_data.get('time_remaining', 20)
    
    # Progress indicator
    progress = question_num / total_questions
    st.progress(progress)
    st.markdown(f"**Question {question_num} of {total_questions}** | ⏱️ Time remaining: ~{time_remaining} minutes")
    
    # Question display
    st.markdown(f"""
    <div class="enhanced-card" style="background: linear-gradient(135deg, rgba(59, 130, 246, 0.1), rgba(59, 130, 246, 0.05)); border-left: 6px solid #3b82f6;">
        <h3 style="margin-top: 0; color: #3b82f6;">❓ Interview Question</h3>
        <h4 style="color: var(--text-primary); font-size: 1.2rem; line-height: 1.6;">{question}</h4>
    </div>
    """, unsafe_allow_html=True)
    
    # Question tips
    tips = question_data.get('tips', [])
    if tips:
        with st.expander("💡 Tips for this question", expanded=False):
            for tip in tips:
                st.markdown(f"• {tip}")
    
    # Response interface
    use_voice = st.session_state.get('interview_settings', {}).get('use_voice', False)
    
    if use_voice:
        render_voice_response_interface(interview_system, interview_id, question)
    else:
        render_text_response_interface(interview_system, interview_id, question)

def render_voice_response_interface(interview_system, interview_id: str, question: str):
    """Render voice response interface"""
    st.markdown("#### 🎤 Voice Response")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col1:
        if st.button("🎤 Start Recording", type="primary", use_container_width=True):
            st.session_state.recording_active = True
            st.rerun()
    
    with col2:
        if st.session_state.get('recording_active', False):
            st.warning("🔴 Recording... Speak your answer clearly")
            
            # Get AI agent for voice service
            ai_agent = get_enhanced_ai_agent()
            if ai_agent.voice_service.is_available():
                with st.spinner("🎧 Listening to your response..."):
                    voice_response = ai_agent.voice_service.speech_to_text()
                
                if voice_response and not voice_response.startswith("❌"):
                    st.session_state.current_voice_response = voice_response
                    st.session_state.recording_active = False
                    st.success(f"✅ Response recorded: {voice_response[:100]}...")
                else:
                    st.error(voice_response)
                    st.session_state.recording_active = False
            else:
                st.error("❌ Voice service not available")
                st.session_state.recording_active = False
    
    with col3:
        if st.button("⏹️ Stop & Submit", use_container_width=True):
            voice_response = st.session_state.get('current_voice_response', '')
            if voice_response:
                submit_interview_response(interview_system, interview_id, voice_response)
            else:
                st.error("❌ No voice response recorded")
    
    # Show recorded response
    if st.session_state.get('current_voice_response'):
        st.markdown("#### 📝 Your Recorded Response:")
        st.text_area("Response", value=st.session_state.current_voice_response, height=150, disabled=True)

def render_text_response_interface(interview_system, interview_id: str, question: str):
    """Render text response interface"""
    st.markdown("#### ✍️ Type Your Response")
    
    # Response input
    response = st.text_area(
        "Your answer:",
        placeholder="Take your time to provide a thoughtful, detailed answer...",
        height=200,
        help="Aim for 1-3 minutes of speaking time (roughly 150-400 words)"
    )
    
    # Response timer
    if response:
        word_count = len(response.split())
        estimated_time = word_count / 2.5  # Rough estimate: 150 words per minute
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("📝 Word Count", word_count)
        with col2:
            st.metric("⏱️ Est. Speaking Time", f"{estimated_time:.1f} min")
        with col3:
            if word_count < 50:
                st.error("Too short")
            elif word_count > 400:
                st.warning("Too long")
            else:
                st.success("Good length")
    
    # Submit response
    col1, col2 = st.columns([3, 1])
    
    with col1:
        if st.button("📤 Submit Response", type="primary", use_container_width=True, disabled=not response.strip()):
            submit_interview_response(interview_system, interview_id, response)
    
    with col2:
        if st.button("⏭️ Skip Question", use_container_width=True):
            submit_interview_response(interview_system, interview_id, "Question skipped")

def submit_interview_response(interview_system, interview_id: str, response: str):
    with st.spinner("🤖 AI is analyzing your response..."):
        result = interview_system.submit_response(interview_id, response)

    if 'error' not in result:
        # Show immediate feedback
        if result.get('analysis'):
            analysis = result['analysis']

            # Quick feedback - Updated to reflect new metrics
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("🎯 Relevance", f"{analysis.get('relevance_score', 0)*100:.1f}%")
            with col2:
                st.metric("💬 Coherence", f"{analysis.get('coherence_score', 0)*100:.1f}%")
            with col3:
                st.metric("💡 Content", f"{analysis.get('content_score', 0)*100:.1f}%")
            with col4:
                length = analysis.get('response_length', 'appropriate')
                if length == 'appropriate':
                    st.success("✅ Good length")
                elif length == 'too_short':
                    st.warning("⚠️ Too short")
                else:
                    st.warning("⚠️ Too long")

            # AI feedback if available
            if analysis.get('ai_feedback'):
                with st.expander("🤖 Detailed AI Feedback", expanded=True):
                    st.markdown(analysis['ai_feedback'])
                    if analysis.get('missed_points'):
                        st.markdown(f"**Missed Points:** {', '.join(analysis['missed_points'])}")
                    if analysis.get('irrelevant_info'):
                        st.markdown(f"**Irrelevant Info:** {', '.join(analysis['irrelevant_info'])}")
                    if analysis.get('suggestions_for_improvement'):
                        st.markdown(f"**Suggestions:** {analysis['suggestions_for_improvement']}")

        # Clear session state for next question
        if 'current_voice_response' in st.session_state:
            del st.session_state.current_voice_response

        st.success("✅ Response recorded! Moving to next question...")
        time.sleep(2)
        st.rerun()
    else:
        st.error(f"❌ Error submitting response: {result.get('error')}")

def render_interview_completion_results(completion_result: Dict):
    st.markdown("### 🎉 Interview Complete - Performance Report")

    performance = completion_result.get('performance_report', {})
    improvements = completion_result.get('improvement_suggestions', [])
    summary = completion_result.get('interview_summary', {})
    all_responses_analysis = [r['analysis'] for r in completion_result.get('responses', [])] # Access individual response analysis

    # Overall score display (remains similar, but now based on new metrics)
    overall_score = performance.get('overall_score', 0)
    grade = performance.get('grade', 'N/A')

    st.markdown(f"""
    <div style="background: linear-gradient(135deg, #10b981, #059669); color: white; padding: 3rem; border-radius: 20px; text-align: center; margin: 2rem 0;">
        <h1 style="margin: 0; font-size: 3rem; font-weight: 800;">{overall_score:.1f}%</h1>
        <h2 style="margin: 1rem 0; font-size: 1.8rem;">{grade}</h2>
        <p style="font-size: 1.1rem; opacity: 0.9;">Overall Interview Performance Score</p>
    </div>
    """, unsafe_allow_html=True)

    # Detailed breakdown with new metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("💬 Coherence", f"{performance.get('communication_score', 0):.1f}%", help="How well-structured and logical your answers were.")
    with col2:
        st.metric("🎯 Relevance", f"{performance.get('relevance_score', 0):.1f}%", help="How directly your answers addressed the questions.")
    with col3:
        st.metric("💡 Content Depth", f"{performance.get('confidence_score', 0):.1f}%", help="Quality, insightfulness, and detail of your answers.")
    with col4:
        st.metric("⏱️ Avg Time", f"{summary.get('duration_minutes', 0):.1f} min", help="Total duration of the interview session.")

    # Strengths and improvements (now populated from LLM analysis)
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 💪 Your Strengths")
        strengths = performance.get('strengths', [])
        if strengths:
            for strength in strengths:
                st.success(f"✅ {strength}")
        else:
            st.info("No specific strengths identified yet. Keep practicing!")

    with col2:
        st.markdown("#### 📈 Areas for Improvement")
        areas = performance.get('areas_for_improvement', [])
        if areas:
            for area in areas:
                st.warning(f"📌 {area}")
        else:
            st.info("Great job! No major areas for improvement identified.")

    # Personalized Improvement Plan (from LLM suggestions)
    if improvements:
        st.markdown("#### 💡 Personalized Improvement Plan")
        for i, suggestion in enumerate(improvements, 1):
            st.markdown(f"**{i}.** {suggestion}")
    else:
        st.info("No specific improvement suggestions at this time.")

    # Interview summary
    st.markdown("#### 📋 Interview Summary")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Questions Answered", summary.get('questions_answered', 0))
    with col2:
        st.metric("Duration", f"{summary.get('duration_minutes', 0):.1f} min")
    with col3:
        st.metric("Interview Type", summary.get('interview_type', 'Unknown').replace('_', ' ').title())

    # Display detailed feedback for each question
    st.markdown("#### 📝 Detailed Question-by-Question Feedback")
    if all_responses_analysis:
        for i, response_analysis in enumerate(all_responses_analysis, 1):
            with st.expander(f"Question {i} Feedback (Relevance: {response_analysis.get('relevance_score',0):.1f}, Coherence: {response_analysis.get('coherence_score',0):.1f}, Content: {response_analysis.get('content_score',0):.1f})"):
                st.markdown(f"**Question:** {completion_result['responses'][i-1]['question']}")
                st.markdown(f"**Your Response:** {completion_result['responses'][i-1]['response']}")
                st.markdown("---")
                st.markdown(f"**AI Feedback:** {response_analysis.get('ai_feedback', 'No specific feedback.')}")
                st.markdown(f"**Strengths:** {', '.join(response_analysis.get('strengths', ['N/A']))}")
                st.markdown(f"**Weaknesses:** {', '.join(response_analysis.get('weaknesses', ['N/A']))}")
                st.markdown(f"**Missed Key Points:** {', '.join(response_analysis.get('missed_points', ['None']))}")
                st.markdown(f"**Irrelevant Information:** {', '.join(response_analysis.get('irrelevant_info', ['None']))}")
                st.markdown(f"**Suggestions for this question:** {response_analysis.get('suggestions_for_improvement', 'N/A')}")
    else:
        st.info("No detailed response analysis available.")


def render_question_practice_interface(interview_system, profile: Dict):
    """Render question practice interface"""
    st.markdown("### 📚 Interview Question Practice")
    
    # Question category selection
    col1, col2 = st.columns(2)
    
    with col1:
        category = st.selectbox(
            "Question Category",
            ["general", "computer_science", "business", "engineering", "psychology", "behavioral", "postgraduate"],
            format_func=lambda x: x.replace('_', ' ').title(),
            help="Choose the type of questions to practice"
        )
    
    with col2:
        num_questions = st.selectbox(
            "Number of Questions",
            [1, 3, 5, 10],
            index=1,
            help="How many questions to practice"
        )
    
    if st.button("🎯 Generate Practice Questions", type="primary", key="generate_practice_questions"):
        # Get questions from the interview system
        try:
            questions = interview_system.question_banks.get(category, [])
            if questions:
                selected_questions = random.sample(questions, min(num_questions, len(questions)))
                st.session_state.practice_questions = selected_questions
                st.session_state.current_practice_index = 0
                st.session_state.practice_responses = []
                st.success(f"✅ Generated {len(selected_questions)} practice questions!")
            else:
                st.error("❌ No questions available for this category")
        except Exception as e:
            st.error(f"❌ Error generating questions: {e}")
    
    # Display practice questions
    if st.session_state.get('practice_questions'):
        render_practice_question_session(interview_system)

def render_practice_question_session(interview_system):
    """Render practice question session"""
    questions = st.session_state.practice_questions
    current_index = st.session_state.get('current_practice_index', 0)
    
    if current_index >= len(questions):
        st.success("🎉 Practice session completed!")
        
        # Show summary of practice session
        responses = st.session_state.get('practice_responses', [])
        if responses:
            st.markdown("#### 📊 Practice Session Summary")
            
            avg_words = np.mean([len(r.split()) for r in responses])
            st.metric("Average Response Length", f"{avg_words:.0f} words")
            
            for i, (q, r) in enumerate(zip(questions, responses), 1):
                with st.expander(f"Question {i}: {q[:50]}..."):
                    st.markdown(f"**Question:** {q}")
                    st.markdown(f"**Your Response:** {r}")
        
        if st.button("🔄 Start New Practice Session"):
            # Clear practice session
            for key in ['practice_questions', 'current_practice_index', 'practice_responses']:
                if key in st.session_state:
                    del st.session_state[key]
            st.rerun()
        
        return
    
    # Current question
    current_question = questions[current_index]
    
    st.markdown(f"#### Question {current_index + 1} of {len(questions)}")
    
    st.markdown(f"""
    <div class="enhanced-card" style="background: rgba(245, 158, 11, 0.1); border-left: 4px solid #f59e0b;">
        <h4 style="margin-top: 0; color: #f59e0b;">{current_question}</h4>
    </div>
    """, unsafe_allow_html=True)
    
    # Response area
    response = st.text_area(
        "Your practice response:",
        placeholder="Practice your answer here...",
        height=150,
        key=f"practice_response_{current_index}"
    )
    
    # Action buttons
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📤 Submit & Next", type="primary", disabled=not response.strip()):
            # Store response
            if 'practice_responses' not in st.session_state:
                st.session_state.practice_responses = []
            
            st.session_state.practice_responses.append(response)
            st.session_state.current_practice_index += 1
            
            st.success("✅ Response saved! Moving to next question...")
            time.sleep(1)
            st.rerun()
    
    with col2:
        if st.button("⏭️ Skip Question"):
            st.session_state.current_practice_index += 1
            st.rerun()
    
    with col3:
        if st.button("💡 Get Tips"):
            tips = interview_system._get_question_tips(current_question)
            st.info("💡 Tips:\n" + "\n".join([f"• {tip}" for tip in tips]))

def render_interview_analytics(interview_system, profile: Dict):
    """Render interview performance analytics"""
    st.markdown("### 📊 Interview Performance Analytics")
    
    # Get user's interview history
    user_history = interview_system.get_user_interview_history(profile.get('id'))
    
    if not user_history:
        st.info("📈 No interview history available. Complete some mock interviews to see your analytics!")
        return
    
    # Performance overview
    scores = [interview.get('performance_score', 0) for interview in user_history if interview.get('performance_score')]
    
    if scores:
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            avg_score = np.mean(scores)
            st.metric("📈 Average Score", f"{avg_score:.1f}%")
        
        with col2:
            latest_score = scores[0] if scores else 0
            previous_score = scores[1] if len(scores) > 1 else 0
            improvement = latest_score - previous_score
            st.metric("🚀 Latest Score", f"{latest_score:.1f}%", delta=f"{improvement:+.1f}%")
        
        with col3:
            best_score = max(scores)
            st.metric("🏆 Best Score", f"{best_score:.1f}%")
        
        with col4:
            total_interviews = len(user_history)
            st.metric("🎤 Total Interviews", total_interviews)
        
        # Performance trend chart
        if len(scores) > 1:
            st.markdown("#### 📈 Performance Trend")
            
            trend_data = pd.DataFrame({
                'Interview': range(1, len(scores) + 1),
                'Score': scores[::-1]  # Reverse to show chronological order
            })
            
            fig = px.line(
                trend_data, 
                x='Interview', 
                y='Score',
                title="Interview Performance Over Time",
                markers=True
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        # Skills breakdown
        st.markdown("#### 🎯 Skills Analysis")
        
        # Aggregate performance data
        comm_scores = []
        rel_scores = []
        conf_scores = []
        
        for interview in user_history:
            if interview.get('responses'):
                responses = interview['responses']
                comm_scores.extend([r['analysis'].get('coherence_score', 0.7) for r in responses])
                rel_scores.extend([r['analysis'].get('relevance_score', 0.7) for r in responses])
                # Confidence calculation would be more complex in real implementation
                conf_scores.append(0.8)  # Placeholder
        
        if comm_scores:
            skills_data = pd.DataFrame({
                'Skill': ['Communication', 'Relevance', 'Confidence'],
                'Average Score': [
                    np.mean(comm_scores) * 100,
                    np.mean(rel_scores) * 100,
                    np.mean(conf_scores) * 100
                ]
            })
            
            fig = px.bar(
                skills_data,
                x='Skill',
                y='Average Score',
                title="Skills Breakdown",
                color='Average Score',
                color_continuous_scale='viridis'
            )
            fig.update_layout(height=400, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

def render_interview_tips_guidance(interview_system, profile: Dict):
    """Render interview tips and guidance"""
    st.markdown("### 💡 Interview Tips & Guidance")
    
    # Personalized tips based on profile
    field_of_interest = profile.get('field_of_interest', '').lower()
    academic_level = profile.get('academic_level', '').lower()
    
    # Determine relevant tip categories
    tip_categories = ['general']
    
    if any(word in field_of_interest for word in ['computer', 'technology', 'software', 'data']):
        tip_categories.append('technical')
    
    if 'postgraduate' in academic_level or 'masters' in academic_level or 'phd' in academic_level:
        tip_categories.append('postgraduate')
    
    tip_categories.append('behavioral')
    
    # Display tips in tabs
    if len(tip_categories) > 1:
        tip_tabs = st.tabs([cat.replace('_', ' ').title() + " Tips" for cat in tip_categories])
        
        for tab, category in zip(tip_tabs, tip_categories):
            with tab:
                tips = interview_system.get_interview_tips(category)
                for i, tip in enumerate(tips, 1):
                    st.markdown(f"**{i}.** {tip}")
    else:
        # Single category
        tips = interview_system.get_interview_tips(tip_categories[0])
        for i, tip in enumerate(tips, 1):
            st.markdown(f"**{i}.** {tip}")
    
    # Interview preparation checklist
    st.markdown("#### ✅ Interview Preparation Checklist")
    
    checklist_items = [
        "Research the university and specific program thoroughly",
        "Review your application and personal statement",
        "Prepare specific examples of your achievements",
        "Practice common interview questions out loud",
        "Prepare thoughtful questions to ask the interviewer",
        "Plan your outfit and test your technology (for virtual interviews)",
        "Review current events in your field of study",
        "Practice good posture and eye contact",
        "Prepare your introduction (tell me about yourself)",
        "Get a good night's sleep before the interview"
    ]
    
    for item in checklist_items:
        st.checkbox(item, key=f"checklist_{hash(item)}")
    
    # Common mistakes to avoid
    st.markdown("#### ❌ Common Interview Mistakes to Avoid")
    
    mistakes = [
        "**Not researching the university** - Show you've done your homework",
        "**Speaking too fast** - Take your time and speak clearly",
        "**Being too modest** - Confidently discuss your achievements",
        "**Not asking questions** - Prepare thoughtful questions about the program",
        "**Giving vague answers** - Use specific examples and the STAR method",
        "**Appearing disinterested** - Show enthusiasm and passion",
        "**Not practicing** - Practice common questions beforehand",
        "**Poor body language** - Maintain good posture and eye contact"
    ]
    
    for mistake in mistakes:
        st.markdown(f"• {mistake}")
    
    # STAR method explanation
    st.markdown("#### 🌟 The STAR Method for Behavioral Questions")
    
    st.markdown("""
    <div class="enhanced-card">
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 1rem;">
            <div style="text-align: center; padding: 1rem; background: rgba(59, 130, 246, 0.1); border-radius: 10px;">
                <h4 style="color: #3b82f6; margin-top: 0;">🎯 Situation</h4>
                <p>Describe the context and background</p>
            </div>
            <div style="text-align: center; padding: 1rem; background: rgba(16, 185, 129, 0.1); border-radius: 10px;">
                <h4 style="color: #10b981; margin-top: 0;">📋 Task</h4>
                <p>Explain what you needed to accomplish</p>
            </div>
            <div style="text-align: center; padding: 1rem; background: rgba(245, 158, 11, 0.1); border-radius: 10px;">
                <h4 style="color: #f59e0b; margin-top: 0;">⚡ Action</h4>
                <p>Detail the steps you took</p>
            </div>
            <div style="text-align: center; padding: 1rem; background: rgba(139, 92, 246, 0.1); border-radius: 10px;">
                <h4 style="color: #8b5cf6; margin-top: 0;">🏆 Result</h4>
                <p>Share the outcome and what you learned</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

def render_interview_history(interview_system, profile: Dict):
    """Render interview history"""
    st.markdown("### 📋 Your Interview History")
    
    # Get user's interview history
    user_history = interview_system.get_user_interview_history(profile.get('id'))
    
    if not user_history:
        st.info("📚 No interview history yet. Complete some mock interviews to build your history!")
        return
    
    # Display interviews
    for i, interview in enumerate(user_history):
        with st.expander(
            f"🎤 Interview #{len(user_history) - i} - {interview.get('interview_type', 'Unknown').replace('_', ' ').title()} "
            f"({interview.get('performance_score', 0):.0f}%)",
            expanded=(i == 0)  # Expand most recent
        ):
            # Interview details
            col1, col2, col3 = st.columns(3)
            
            with col1:
                created_time = interview.get('created_time', '')
                if created_time:
                    dt = datetime.fromisoformat(created_time)
                    st.markdown(f"**📅 Date:** {dt.strftime('%Y-%m-%d')}")
                
                st.markdown(f"**🎯 Type:** {interview.get('interview_type', 'Unknown').replace('_', ' ').title()}")
            
            with col2:
                st.markdown(f"**❓ Questions:** {len(interview.get('questions', []))}")
                st.markdown(f"**✅ Completed:** {len(interview.get('responses', []))}")
            
            with col3:
                score = interview.get('performance_score', 0)
                st.markdown(f"**📊 Score:** {score:.1f}%")
                
                status = interview.get('status', 'unknown')
                if status == 'completed':
                    st.success("✅ Completed")
                else:
                    st.warning(f"⚠️ {status.title()}")
            
            # Questions and responses
            if interview.get('responses'):
                st.markdown("#### 💬 Questions & Responses")
                
                for j, response_data in enumerate(interview['responses'][:3], 1):  # Show first 3
                    with st.container():
                        st.markdown(f"**Q{j}:** {response_data['question']}")
                        
                        # Response preview
                        response_text = response_data['response']
                        preview = response_text[:200] + "..." if len(response_text) > 200 else response_text
                        st.markdown(f"**A{j}:** {preview}")
                        
                        # Analysis summary
                        analysis = response_data.get('analysis', {})
                        word_count = analysis.get('word_count', 0)
                        length_assessment = analysis.get('response_length', 'unknown')
                        
                        st.caption(f"📝 {word_count} words • {length_assessment}")
                        
                        st.markdown("---")
                
                if len(interview['responses']) > 3:
                    st.info(f"... and {len(interview['responses']) - 3} more responses")
    
    # Export interview history
    if st.button("📥 Export Interview History", key="export_interview_history"):
        export_interview_history(user_history, profile)

def export_interview_history(interview_history: List[Dict], profile: Dict):
    """Export interview history to downloadable format"""
    try:
        # Create comprehensive report
        report_content = f"""
# Interview History Report

**Student:** {profile.get('first_name', '')} {profile.get('last_name', '')}
**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Summary
- Total Interviews: {len(interview_history)}
- Average Score: {np.mean([i.get('performance_score', 0) for i in interview_history if i.get('performance_score')]):.1f}%
- Interview Types: {', '.join(set([i.get('interview_type', 'Unknown') for i in interview_history]))}

## Interview Details
"""
        
        for i, interview in enumerate(interview_history, 1):
            report_content += f"""
### Interview {i}
- **Date:** {interview.get('created_time', 'Unknown')[:10]}
- **Type:** {interview.get('interview_type', 'Unknown').replace('_', ' ').title()}
- **Score:** {interview.get('performance_score', 0):.1f}%
- **Questions:** {len(interview.get('questions', []))}
- **Responses:** {len(interview.get('responses', []))}

"""
            
            # Add top responses
            if interview.get('responses'):
                report_content += "**Sample Responses:**\n"
                for j, response in enumerate(interview['responses'][:2], 1):
                    report_content += f"""
Q{j}: {response['question']}
A{j}: {response['response'][:300]}...

"""
        
        report_content += f"""
---
Report generated by UEL AI Interview Preparation Center
© {datetime.now().year} University of East London
"""
        
        # Download button
        st.download_button(
            label="📥 Download Interview History Report",
            data=report_content,
            file_name=f"uel_interview_history_{profile.get('first_name', 'student')}_{datetime.now().strftime('%Y%m%d')}.txt",
            mime="text/plain"
        )
        
        st.success("✅ Interview history report generated successfully!")
        
    except Exception as e:
        st.error(f"❌ Export failed: {e}")
        module_logger.error(f"Interview history export error: {e}")

# =============================================================================
# ADVANCED DOCUMENT VERIFICATION SYSTEM (RESTORED)
# =============================================================================

def render_advanced_document_verification():
    """Render comprehensive document verification interface"""
    st.markdown("""
    <div class="page-header">
        <div class="header-icon">📄</div>
        <div>
            <h1 style="margin: 0; background: var(--uel-gradient); -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-weight: 800; font-size: 2.5rem;">Document Verification Center</h1>
            <p style="margin: 0; color: var(--text-secondary); font-size: 1.2rem; font-weight: 500;">AI-powered document analysis and verification</p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Document verification status overview
    render_document_status_overview()
    
    # Document upload and verification interface
    tab1, tab2, tab3, tab4 = st.tabs([
        "📤 Upload Documents", 
        "🔍 Verification Results", 
        "📊 Document Analytics", 
        "📋 Required Documents"
    ])
    
    with tab1:
        render_document_upload_interface()
    
    with tab2:
        render_verification_results_interface()
    
    with tab3:
        render_document_analytics_interface()
    
    with tab4:
        render_required_documents_interface()

def render_document_status_overview():
    """Render document verification status overview"""
    profile = st.session_state.current_profile
    verification_results = st.session_state.verification_results
    
    # Calculate verification statistics
    total_docs = len(verification_results)
    verified_docs = sum(1 for result in verification_results.values() if result.get('verification_status') == 'verified')
    pending_docs = sum(1 for result in verification_results.values() if result.get('verification_status') == 'needs_review')
    rejected_docs = sum(1 for result in verification_results.values() if result.get('verification_status') == 'rejected')
    
    # Status cards
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "📄 Total Documents",
            total_docs,
            help="Total documents uploaded for verification"
        )
    
    with col2:
        st.metric(
            "✅ Verified",
            verified_docs,
            delta=f"{(verified_docs/max(total_docs,1)*100):.0f}%" if total_docs > 0 else "0%",
            help="Documents that passed verification"
        )
    
    with col3:
        st.metric(
            "⏳ Pending Review",
            pending_docs,
            delta=f"{(pending_docs/max(total_docs,1)*100):.0f}%" if total_docs > 0 else "0%",
            help="Documents requiring manual review"
        )
    
    with col4:
        st.metric(
            "❌ Rejected",
            rejected_docs,
            delta=f"{(rejected_docs/max(total_docs,1)*100):.0f}%" if total_docs > 0 else "0%",
            help="Documents that failed verification"
        )

def render_document_upload_interface():
    """Render document upload and verification interface"""
    st.markdown("### 📤 Document Upload & Verification")
    
    # Document type selection
    document_types = {
        "Academic Transcript": {
            "code": "transcript",
            "description": "Official academic records from your institution",
            "accepted_formats": ["PDF", "JPG", "PNG"],
            "max_size": "10MB",
            "requirements": ["Institution seal", "Official signature", "Clear grade details"]
        },
        "IELTS Certificate": {
            "code": "ielts_certificate", 
            "description": "English language proficiency test results",
            "accepted_formats": ["PDF", "JPG", "PNG"],
            "max_size": "5MB",
            "requirements": ["Test date", "Test center", "All band scores", "Overall score"]
        },
        "Passport": {
            "code": "passport",
            "description": "Government-issued passport for identity verification",
            "accepted_formats": ["JPG", "PNG", "PDF"],
            "max_size": "5MB",
            "requirements": ["Clear photo page", "Readable text", "Valid expiry date"]
        },
        "Personal Statement": {
            "code": "personal_statement",
            "description": "Your written statement of purpose",
            "accepted_formats": ["PDF", "DOC", "DOCX"],
            "max_size": "2MB",
            "requirements": ["Word count 500-1000", "Clear formatting", "Relevant content"]
        },
        "Reference Letter": {
            "code": "reference_letter",
            "description": "Letter of recommendation from academic or professional referee",
            "accepted_formats": ["PDF", "DOC", "DOCX"],
            "max_size": "5MB",
            "requirements": ["Referee details", "Institution letterhead", "Signature"]
        }
    }
    
    selected_doc_type = st.selectbox(
        "📋 Select Document Type",
        list(document_types.keys()),
        help="Choose the type of document you want to upload"
    )
    
    if selected_doc_type:
        doc_info = document_types[selected_doc_type]
        
        # Display document requirements
        st.markdown(f"""
        <div class="enhanced-card" style="background: rgba(59, 130, 246, 0.1); border-left: 4px solid #3b82f6;">
            <h4 style="margin-top: 0; color: #3b82f6;">📋 {selected_doc_type} Requirements</h4>
            <p><strong>Description:</strong> {doc_info['description']}</p>
            <p><strong>Accepted Formats:</strong> {', '.join(doc_info['accepted_formats'])}</p>
            <p><strong>Maximum Size:</strong> {doc_info['max_size']}</p>
            <p><strong>Requirements:</strong></p>
            <ul>{''.join(f'<li>{req}</li>' for req in doc_info['requirements'])}</ul>
        </div>
        """, unsafe_allow_html=True)
        
        # File upload
        uploaded_file = st.file_uploader(
            f"📁 Upload {selected_doc_type}",
            type=[fmt.lower() for fmt in doc_info['accepted_formats']],
            help=f"Upload your {selected_doc_type.lower()} file"
        )
        
        if uploaded_file:
            # Display file information
            file_size_mb = uploaded_file.size / (1024 * 1024)
            st.success(f"✅ File uploaded: {uploaded_file.name} ({file_size_mb:.2f} MB)")
            
            # Additional document information form
            with st.form(f"document_info_{doc_info['code']}"):
                st.markdown("#### 📝 Additional Information")
                
                additional_info = {}
                
                if doc_info['code'] == "transcript":
                    col1, col2 = st.columns(2)
                    with col1:
                        additional_info['institution_name'] = st.text_input("Institution Name *")
                        additional_info['graduation_date'] = st.date_input("Graduation Date")
                    with col2:
                        additional_info['overall_grade'] = st.text_input("Overall Grade/GPA *")
                        additional_info['degree_level'] = st.selectbox("Degree Level", 
                            ["High School", "Bachelor's", "Master's", "PhD", "Other"])
                
                elif doc_info['code'] == "ielts_certificate":
                    col1, col2 = st.columns(2)
                    with col1:
                        additional_info['test_date'] = st.date_input("Test Date *")
                        additional_info['test_center'] = st.text_input("Test Center *")
                    with col2:
                        additional_info['overall_score'] = st.number_input("Overall Score *", 0.0, 9.0, 6.5, 0.5)
                        additional_info['test_report_number'] = st.text_input("Test Report Form Number")
                
                elif doc_info['code'] == "passport":
                    col1, col2 = st.columns(2)
                    with col1:
                        additional_info['passport_number'] = st.text_input("Passport Number *")
                        additional_info['nationality'] = st.text_input("Nationality *")
                    with col2:
                        additional_info['issue_date'] = st.date_input("Issue Date")
                        additional_info['expiry_date'] = st.date_input("Expiry Date *")
                
                elif doc_info['code'] == "personal_statement":
                    additional_info['word_count'] = st.number_input("Approximate Word Count", 0, 2000, 500)
                    additional_info['target_program'] = st.text_input("Target Program/Course")
                    additional_info['statement_focus'] = st.text_area("Main Focus Areas", height=100)
                
                elif doc_info['code'] == "reference_letter":
                    col1, col2 = st.columns(2)
                    with col1:
                        additional_info['referee_name'] = st.text_input("Referee Name *")
                        additional_info['referee_position'] = st.text_input("Referee Position *")
                    with col2:
                        additional_info['referee_institution'] = st.text_input("Referee Institution *")
                        additional_info['referee_email'] = st.text_input("Referee Email")
                
                # Submit for verification
                if st.form_submit_button("🔍 Verify Document", type="primary", use_container_width=True):
                    verify_uploaded_document(uploaded_file, doc_info['code'], additional_info)

def verify_uploaded_document(uploaded_file, doc_type: str, additional_info: Dict):
    """Process and verify uploaded document"""
    try:
        with st.spinner("🤖 AI is analyzing your document..."):
            # Simulate document processing
            time.sleep(2)  # Simulate processing time
            
            # Prepare document data
            document_data = {
                "file_name": uploaded_file.name,
                "file_size": uploaded_file.size,
                "file_type": uploaded_file.type,
                "upload_timestamp": datetime.now().isoformat(),
                **additional_info
            }
            
            # Get AI agent for verification
            ai_agent = get_enhanced_ai_agent()
            verification_result = ai_agent.document_verifier.verify_document(document_data, doc_type)
            
            # Store verification result
            doc_id = verification_result.get('document_id', f"doc_{int(time.time())}")
            st.session_state.verification_results[doc_id] = verification_result
            
            # Display verification results
            display_verification_results(verification_result)
            
            # Update profile interaction
            if st.session_state.current_profile:
                profile = UserProfile.from_dict(st.session_state.current_profile)
                profile.add_interaction("document_verification")
                st.session_state.current_profile = profile.to_dict()
            
            logger.info(f"Document verified: {uploaded_file.name} - Status: {verification_result.get('verification_status')}")
            
    except Exception as e:
        st.error(f"❌ Document verification failed: {e}")
        logger.error(f"Document verification error: {e}")

def display_verification_results(result: Dict):
    """Display comprehensive verification results"""
    status = result.get('verification_status', 'unknown')
    confidence = result.get('confidence_score', 0.0)
    doc_type = result.get('document_type', 'unknown')
    
    # Status display with styling
    if status == "verified":
        st.success("🎉 Document Successfully Verified!")
        status_color = "#10b981"
        status_icon = "✅"
    elif status == "needs_review":
        st.warning("⚠️ Document Requires Manual Review")
        status_color = "#f59e0b"
        status_icon = "🔍"
    elif status == "rejected":
        st.error("❌ Document Verification Failed")
        status_color = "#ef4444"
        status_icon = "❌"
    else:
        st.info("ℹ️ Verification Status Unknown")
        status_color = "#6b7280"
        status_icon = "❓"
    
    # Detailed results display
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown(f"""
        <div style="background: {status_color}; color: white; padding: 2rem; border-radius: 16px; text-align: center;">
            <h2 style="margin: 0; font-size: 3rem;">{status_icon}</h2>
            <h3 style="margin: 1rem 0;">{status.replace('_', ' ').title()}</h3>
            <p style="margin: 0;">Document Type: {doc_type.replace('_', ' ').title()}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.metric("🎯 Confidence Score", f"{confidence:.1%}")
        st.metric("📅 Verified On", datetime.now().strftime("%Y-%m-%d"))
        st.metric("🆔 Document ID", result.get('document_id', 'N/A')[:8] + "...")
    
    # Issues and recommendations
    issues = result.get('issues_found', [])
    recommendations = result.get('recommendations', [])
    
    if issues:
        st.markdown("#### ⚠️ Issues Identified")
        for issue in issues:
            st.markdown(f"• {issue}")
    
    if recommendations:
        st.markdown("#### 💡 Recommendations")
        for rec in recommendations:
            st.markdown(f"• {rec}")
    
    # Verified fields breakdown
    verified_fields = result.get('verified_fields', {})
    if verified_fields:
        st.markdown("#### 📋 Field Verification Details")
        
        verification_df = pd.DataFrame([
            {
                "Field": field.replace('_', ' ').title(),
                "Status": "✅ Verified" if data['verified'] else "❌ Not Verified",
                "Confidence": f"{data['confidence']:.1%}",
                "Value": str(data['value'])[:50] + "..." if len(str(data['value'])) > 50 else str(data['value'])
            }
            for field, data in verified_fields.items()
        ])
        
        st.dataframe(verification_df, use_container_width=True)

def render_verification_results_interface():
    """Render verification results interface"""
    st.markdown("### 🔍 Verification Results")
    
    verification_results = st.session_state.verification_results
    
    if not verification_results:
        st.info("📄 No documents have been verified yet. Upload documents in the previous tab.")
        return
    
    # Display all verification results
    for doc_id, result in verification_results.items():
        with st.expander(f"📄 {result.get('document_type', 'Unknown').title()} - {result.get('verification_status', 'Unknown').title()}", expanded=False):
            display_verification_results(result)

def render_document_analytics_interface():
    """Render document analytics interface"""
    st.markdown("### 📊 Document Analytics")
    
    verification_results = st.session_state.verification_results
    
    if not verification_results:
        st.info("📊 No document data available for analytics.")
        return
    
    # Document type distribution
    doc_types = [result.get('document_type', 'Unknown') for result in verification_results.values()]
    type_counts = pd.Series(doc_types).value_counts()
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.pie(
            values=type_counts.values,
            names=type_counts.index,
            title="Document Types Distribution"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Verification status distribution
        statuses = [result.get('verification_status', 'Unknown') for result in verification_results.values()]
        status_counts = pd.Series(statuses).value_counts()
        
        fig = px.bar(
            x=status_counts.index,
            y=status_counts.values,
            title="Verification Status Distribution",
            color=status_counts.values,
            color_continuous_scale='viridis'
        )
        st.plotly_chart(fig, use_container_width=True)

def render_required_documents_interface():
    """Render required documents checklist"""
    st.markdown("### 📋 Required Documents Checklist")
    
    profile = st.session_state.current_profile
    academic_level = profile.get('academic_level', '').lower() if profile else 'undergraduate'
    
    # Define required documents based on academic level
    required_docs = {
        'undergraduate': [
            {"name": "Academic Transcript", "code": "transcript", "required": True, "description": "Official high school or previous university transcripts"},
            {"name": "English Proficiency Certificate", "code": "ielts_certificate", "required": True, "description": "IELTS, TOEFL, or equivalent test results"},
            {"name": "Passport Copy", "code": "passport", "required": True, "description": "Clear copy of passport information page"},
            {"name": "Personal Statement", "code": "personal_statement", "required": True, "description": "Written statement of purpose and goals"},
            {"name": "Letters of Recommendation", "code": "reference_letter", "required": False, "description": "Academic or professional references"}
        ],
        'postgraduate': [
            {"name": "Bachelor's Degree Certificate", "code": "transcript", "required": True, "description": "Official university degree certificate and transcripts"},
            {"name": "English Proficiency Certificate", "code": "ielts_certificate", "required": True, "description": "IELTS 6.5+ or equivalent for postgraduate study"},
            {"name": "Passport Copy", "code": "passport", "required": True, "description": "Valid passport information page"},
            {"name": "Research Proposal", "code": "personal_statement", "required": True, "description": "Detailed research proposal for your intended study"},
            {"name": "Academic References", "code": "reference_letter", "required": True, "description": "Letters from academic supervisors or professors"},
            {"name": "CV/Resume", "code": "cv", "required": False, "description": "Detailed academic and professional CV"}
        ]
    }
    
    # Get appropriate document list
    level_key = 'postgraduate' if 'postgraduate' in academic_level or 'masters' in academic_level or 'phd' in academic_level else 'undergraduate'
    docs_needed = required_docs.get(level_key, required_docs['undergraduate'])
    
    st.markdown(f"#### Required for {level_key.title()} Applications")
    
    # Check which documents have been uploaded
    uploaded_types = [result.get('document_type') for result in st.session_state.verification_results.values()]
    
    for doc in docs_needed:
        doc_uploaded = doc['code'] in uploaded_types
        status_icon = "✅" if doc_uploaded else "❌" if doc['required'] else "⭕"
        requirement_text = "Required" if doc['required'] else "Optional"
        
        st.markdown(f"""
        <div class="enhanced-card" style="border-left: 4px solid {'#10b981' if doc_uploaded else '#ef4444' if doc['required'] else '#f59e0b'};">
            <div style="display: flex; align-items: center; justify-content: space-between;">
                <div>
                    <h4 style="margin: 0; color: var(--text-primary);">{status_icon} {doc['name']}</h4>
                    <p style="margin: 0.5rem 0; color: var(--text-secondary);">{doc['description']}</p>
                    <small style="color: var(--text-light);">{requirement_text}</small>
                </div>
                <div style="text-align: right;">
                    {"<span style='color: #10b981; font-weight: bold;'>Uploaded ✓</span>" if doc_uploaded else f"<span style='color: #ef4444; font-weight: bold;'>Missing</span>" if doc['required'] else "<span style='color: #f59e0b;'>Optional</span>"}
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Progress summary
    total_required = sum(1 for doc in docs_needed if doc['required'])
    uploaded_required = sum(1 for doc in docs_needed if doc['required'] and doc['code'] in uploaded_types)
    progress = (uploaded_required / total_required * 100) if total_required > 0 else 0
    
    st.markdown(f"### 📊 Document Upload Progress")
    st.progress(progress / 100)
    st.markdown(f"**{uploaded_required}/{total_required} required documents uploaded ({progress:.0f}% complete)**")

# =============================================================================
# COMPREHENSIVE ANALYTICS DASHBOARD (RESTORED)
# =============================================================================

def render_comprehensive_analytics_dashboard():
    """Render advanced analytics dashboard with visualizations"""
    st.markdown("""
    <div class="page-header">
        <div class="header-icon">📊</div>
        <div>
            <h1 style="margin: 0; background: var(--uel-gradient); -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-weight: 800; font-size: 2.5rem;">Analytics Dashboard</h1>
            <p style="margin: 0; color: var(--text-secondary); font-size: 1.2rem; font-weight: 500;">Comprehensive insights and data analysis</p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Analytics filter controls
    render_analytics_filters()
    
    # Main analytics tabs
    tab1, tab2, tab3, tab4, tab5,tab6 = st.tabs([
        "📊 Overview", 
        "👤 Profile Analytics", 
        "🎯 Course Analytics", 
        "🔮 Prediction Analytics", 
        "📈 System Performance",
	"🔬 Research Evaluation"
    ])
    
    with tab1:
        render_overview_analytics()
    
    with tab2:
        render_profile_analytics()
    
    with tab3:
        render_course_analytics()
    
    with tab4:
        render_prediction_analytics()
    
    with tab5:
        render_system_performance_analytics()

def render_analytics_filters():
    """Render analytics filtering controls"""
    with st.expander("🔧 Analytics Filters", expanded=False):
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            date_range = st.date_input(
                "Date Range",
                value=[datetime.now() - timedelta(days=30), datetime.now()],
                key="analytics_date_range"
            )
        
        with col2:
            time_period = st.selectbox(
                "Time Period",
                ["Last 7 days", "Last 30 days", "Last 3 months", "Last 6 months", "All time"],
                index=1,
                key="analytics_time_period"
            )
        
        with col3:
            data_source = st.selectbox(
                "Data Source",
                ["All Sources", "Profile Data", "Chat Interactions", "Course Recommendations", "Predictions"],
                key="analytics_data_source"
            )
        
        with col4:
            visualization_type = st.selectbox(
                "Chart Type",
                ["Auto", "Bar Chart", "Line Chart", "Pie Chart", "Scatter Plot", "Heatmap"],
                key="analytics_viz_type"
            )

def render_overview_analytics():
    """Render overview analytics with key metrics"""
    st.markdown("### 📊 System Overview")
    
    # Generate sample analytics data
    analytics_data = generate_analytics_data()
    
    # Key performance indicators
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "👥 Active Users",
            analytics_data['active_users'],
            delta=f"+{analytics_data['user_growth']}%",
            help="Users active in the last 30 days"
        )
    
    with col2:
        st.metric(
            "💬 Total Interactions",
            analytics_data['total_interactions'],
            delta=f"+{analytics_data['interaction_growth']}",
            help="Total AI chat interactions"
        )
    
    with col3:
        st.metric(
            "🎯 Recommendations Generated",
            analytics_data['recommendations_generated'],
            delta=f"+{analytics_data['recommendation_growth']}%",
            help="Course recommendations provided"
        )
    
    with col4:
        st.metric(
            "🔮 Predictions Made",
            analytics_data['predictions_made'],
            delta=f"+{analytics_data['prediction_growth']}%",
            help="Admission predictions generated"
        )
    
    # Usage trends visualization
    st.markdown("### 📈 Usage Trends")
    
    # Generate time series data
    dates = pd.date_range(start=datetime.now() - timedelta(days=30), end=datetime.now(), freq='D')
    usage_data = pd.DataFrame({
        'Date': dates,
        'Chat_Interactions': np.random.poisson(15, len(dates)),
        'Course_Recommendations': np.random.poisson(8, len(dates)),
        'Admission_Predictions': np.random.poisson(5, len(dates)),
        'Document_Verifications': np.random.poisson(3, len(dates))
    })
    
    fig = px.line(
        usage_data, 
        x='Date', 
        y=['Chat_Interactions', 'Course_Recommendations', 'Admission_Predictions', 'Document_Verifications'],
        title="Daily Feature Usage Trends",
        labels={'value': 'Usage Count', 'variable': 'Feature'}
    )
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)
    
    # Feature popularity
    col1, col2 = st.columns(2)
    
    with col1:
        feature_usage = {
            'AI Chat': analytics_data['total_interactions'],
            'Course Recommendations': analytics_data['recommendations_generated'],
            'Admission Predictions': analytics_data['predictions_made'],
            'Document Verification': analytics_data['documents_verified'],
            'Profile Management': analytics_data['profile_updates']
        }
        
        fig = px.pie(
            values=list(feature_usage.values()),
            names=list(feature_usage.keys()),
            title="Feature Usage Distribution"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # User satisfaction scores
        satisfaction_data = pd.DataFrame({
            'Feature': list(feature_usage.keys()),
            'Satisfaction': [4.2, 4.5, 4.1, 3.9, 4.3],
            'Usage_Count': list(feature_usage.values())
        })
        
        fig = px.scatter(
            satisfaction_data,
            x='Usage_Count',
            y='Satisfaction',
            size='Usage_Count',
            color='Feature',
            title="Feature Usage vs Satisfaction",
            labels={'Usage_Count': 'Usage Count', 'Satisfaction': 'User Satisfaction (1-5)'}
        )
        st.plotly_chart(fig, use_container_width=True)

def render_profile_analytics():
    """Render profile-specific analytics"""
    st.markdown("### 👤 Profile Analytics")
    
    profile = st.session_state.current_profile
    if not profile:
        st.warning("No profile data available for analysis")
        return
    
    # Profile completion analysis
    col1, col2 = st.columns(2)
    
    with col1:
        completion = profile.get('profile_completion', 0)
        
        # Completion gauge chart
        fig = go.Figure(go.Indicator(
            mode = "gauge+number+delta",
            value = completion,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Profile Completion"},
            delta = {'reference': 80},
            gauge = {
                'axis': {'range': [None, 100]},
                'bar': {'color': "#00B8A9"},
                'steps': [
                    {'range': [0, 50], 'color': "#FFE66D"},
                    {'range': [50, 80], 'color': "#4ECDC4"},
                    {'range': [80, 100], 'color': "#00B8A9"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 90
                }
            }
        ))
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Profile field completion breakdown
        field_completion = {
            'Basic Info': 100 if all([profile.get('first_name'), profile.get('last_name'), profile.get('country')]) else 60,
            'Academic': 100 if all([profile.get('academic_level'), profile.get('gpa'), profile.get('field_of_interest')]) else 70,
            'Professional': 100 if profile.get('work_experience_years', 0) > 0 else 30,
            'Interests': 100 if profile.get('interests') and profile.get('career_goals') else 50,
            'Preferences': 100 if all([profile.get('preferred_study_mode'), profile.get('budget_range')]) else 40
        }
        
        fig = px.bar(
            x=list(field_completion.keys()),
            y=list(field_completion.values()),
            title="Profile Section Completion",
            color=list(field_completion.values()),
            color_continuous_scale="viridis"
        )
        fig.update_layout(height=300, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    # Activity timeline
    st.markdown("### 📅 Activity Timeline")
    
    # Generate activity data
    activity_data = generate_user_activity_timeline(profile)
    
    if activity_data:
        activity_df = pd.DataFrame(activity_data)
        
        fig = px.timeline(
            activity_df,
            x_start="start_date",
            x_end="end_date", 
            y="activity",
            color="category",
            title="Your UEL AI Assistant Activity Timeline"
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    # Interest analysis
    interests = profile.get('interests', [])
    if interests:
        st.markdown("### 🎯 Interest Analysis")
        
        # Interest categories
        interest_categories = categorize_interests(interests)
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.bar(
                x=list(interest_categories.keys()),
                y=list(interest_categories.values()),
                title="Interest Categories",
                color=list(interest_categories.values()),
                color_continuous_scale="plasma"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Interest word cloud simulation with bar chart
            interest_weights = {interest: len(interest.split()) for interest in interests}
            
            fig = px.bar(
                x=list(interest_weights.values()),
                y=list(interest_weights.keys()),
                orientation='h',
                title="Interest Areas by Complexity",
                color=list(interest_weights.values()),
                color_continuous_scale="blues"
            )
            st.plotly_chart(fig, use_container_width=True)

def render_course_analytics():
    """Render course recommendation analytics"""
    st.markdown("### 🎯 Course Recommendation Analytics")
    
    # Get course data
    ai_agent = get_enhanced_ai_agent()
    courses_df = ai_agent.data_manager.courses_df
    
    if courses_df.empty:
        st.warning("No course data available for analysis")
        return
    
    # Course distribution analysis
    col1, col2 = st.columns(2)
    
    with col1:
        if 'level' in courses_df.columns:
            level_counts = courses_df['level'].value_counts()
            
            fig = px.pie(
                values=level_counts.values,
                names=level_counts.index,
                title="Courses by Academic Level"
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        if 'department' in courses_df.columns:
            dept_counts = courses_df['department'].value_counts().head(10)
            
            fig = px.bar(
                x=dept_counts.index,
                y=dept_counts.values,
                title="Courses by Department (Top 10)"
            )
            # Fixed: Use update_layout instead of update_xaxis
            fig.update_layout(xaxis={'tickangle': 45})
            st.plotly_chart(fig, use_container_width=True)
    
    # Fee analysis
    if 'fees_international' in courses_df.columns:
        st.markdown("### 💰 Fee Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.histogram(
                courses_df,
                x='fees_international',
                title="International Fee Distribution",
                nbins=20,
                color_discrete_sequence=['#00B8A9']
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            if 'level' in courses_df.columns:
                fig = px.box(
                    courses_df,
                    x='level',
                    y='fees_international',
                    title="Fee Range by Academic Level"
                )
                st.plotly_chart(fig, use_container_width=True)
    
    # Course popularity and trending
    if 'trending_score' in courses_df.columns:
        st.markdown("### 🔥 Course Popularity Trends")
        
        trending_courses = courses_df.nlargest(10, 'trending_score')
        
        fig = px.bar(
            trending_courses,
            x='trending_score',
            y='course_name',
            orientation='h',
            title="Top 10 Trending Courses",
            color='trending_score',
            color_continuous_scale='viridis'
        )
        st.plotly_chart(fig, use_container_width=True)

def render_prediction_analytics():
    """Render admission prediction analytics"""
    st.markdown("### 🔮 Admission Prediction Analytics")
    
    # Generate prediction analytics data
    prediction_data = generate_prediction_analytics_data()
    
    # Prediction accuracy metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "🎯 Model Accuracy",
            f"{prediction_data['model_accuracy']:.1%}",
            help="Overall prediction model accuracy"
        )
    
    with col2:
        st.metric(
            "📊 Predictions Made",
            prediction_data['total_predictions'],
            delta=f"+{prediction_data['prediction_growth']}%",
            help="Total predictions generated"
        )
    
    with col3:
        st.metric(
            "⚡ Avg Response Time",
            f"{prediction_data['avg_response_time']:.2f}s",
            help="Average prediction generation time"
        )
    
    # Prediction distribution
    col1, col2 = st.columns(2)
    
    with col1:
        # Success probability distribution
        prob_ranges = ['0-20%', '21-40%', '41-60%', '61-80%', '81-100%']
        prob_counts = [12, 18, 25, 30, 15]  # Sample data
        
        fig = px.bar(
            x=prob_ranges,
            y=prob_counts,
            title="Admission Probability Distribution",
            color=prob_counts,
            color_continuous_scale='RdYlGn'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Prediction confidence levels
        confidence_data = {
            'High Confidence': 45,
            'Medium Confidence': 35, 
            'Low Confidence': 20
        }
        
        fig = px.pie(
            values=list(confidence_data.values()),
            names=list(confidence_data.keys()),
            title="Prediction Confidence Levels"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Feature importance analysis
    st.markdown("### 📊 Feature Importance in Predictions")
    
    feature_importance = {
        'GPA': 0.35,
        'IELTS Score': 0.25,
        'Work Experience': 0.20,
        'Academic Level': 0.15,
        'Course Difficulty': 0.05
    }
    
    fig = px.bar(
        x=list(feature_importance.keys()),
        y=list(feature_importance.values()),
        title="Most Important Factors in Admission Predictions",
        color=list(feature_importance.values()),
        color_continuous_scale='blues'
    )
    st.plotly_chart(fig, use_container_width=True)

def render_system_performance_analytics():
    """Render system performance analytics"""
    st.markdown("### 📈 System Performance Analytics")
    
    # Performance metrics
    performance_data = get_system_performance_data()
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "🚀 System Uptime",
            f"{performance_data['uptime']:.1%}",
            help="System availability percentage"
        )
    
    with col2:
        st.metric(
            "⚡ Avg Response Time",
            f"{performance_data['avg_response_time']:.2f}s",
            delta=f"{performance_data['response_time_delta']:.2f}s",
            help="Average system response time"
        )
    
    with col3:
        st.metric(
            "💾 Memory Usage",
            f"{performance_data['memory_usage']:.1f}%",
            help="Current memory utilization"
        )
    
    with col4:
        st.metric(
            "🔄 Cache Hit Rate",
            f"{performance_data['cache_hit_rate']:.1%}",
            help="Percentage of requests served from cache"
        )
    
    # Performance trends
    st.markdown("### 📊 Performance Trends")
    
    # Generate performance time series
    dates = pd.date_range(start=datetime.now() - timedelta(hours=24), end=datetime.now(), freq='H')
    perf_data = pd.DataFrame({
        'DateTime': dates,
        'Response_Time': np.random.normal(1.2, 0.3, len(dates)),
        'Memory_Usage': np.random.normal(65, 10, len(dates)),
        'CPU_Usage': np.random.normal(45, 15, len(dates)),
        'Active_Sessions': np.random.poisson(8, len(dates))
    })
    
    # Multi-line performance chart
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Response Time (s)', 'Memory Usage (%)', 'CPU Usage (%)', 'Active Sessions'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    fig.add_trace(go.Scatter(x=perf_data['DateTime'], y=perf_data['Response_Time'], 
                            name='Response Time', line=dict(color='#00B8A9')), row=1, col=1)
    fig.add_trace(go.Scatter(x=perf_data['DateTime'], y=perf_data['Memory_Usage'], 
                            name='Memory Usage', line=dict(color='#E63946')), row=1, col=2)
    fig.add_trace(go.Scatter(x=perf_data['DateTime'], y=perf_data['CPU_Usage'], 
                            name='CPU Usage', line=dict(color='#FFE66D')), row=2, col=1)
    fig.add_trace(go.Scatter(x=perf_data['DateTime'], y=perf_data['Active_Sessions'], 
                            name='Active Sessions', line=dict(color='#4ECDC4')), row=2, col=2)
    
    fig.update_layout(height=600, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

# =============================================================================
# VOICE SERVICES INTEGRATION (RESTORED)
# =============================================================================

def render_voice_services_interface():
    """Render voice services control interface"""
    st.markdown("### 🎤 Voice Services")
    
    ai_agent = get_enhanced_ai_agent()
    voice_available = ai_agent.voice_service.is_available() if hasattr(ai_agent, 'voice_service') else False
    
    if not voice_available:
        st.warning("🎤 Voice services are not available. Please install required dependencies.")
        st.code("pip install SpeechRecognition pyttsx3 pyaudio")
        return
    
    # Voice controls
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🎤 Start Voice Input", use_container_width=True, key="__start_voice_input_3007"):
            handle_voice_input_advanced()
    
    with col2:
        if st.button("🔊 Test Text-to-Speech", use_container_width=True, key="__test_text_to_speec_3008"):
            test_text_to_speech()
    
    with col3:
        if st.button("⚙️ Voice Settings", use_container_width=True, key="___voice_settings_3009"):
            st.session_state.show_voice_settings = True
            st.rerun()
    
    # Voice settings panel
    if st.session_state.get('show_voice_settings', False):
        render_voice_settings_panel()

def handle_voice_input_advanced():
    """Handle advanced voice input with better error handling"""
    try:
        ai_agent = get_enhanced_ai_agent()
        
        with st.spinner("🎧 Listening for your voice..."):
            voice_text = ai_agent.voice_service.speech_to_text()
        
        if voice_text and not voice_text.startswith("❌"):
            st.success(f"🎤 Voice Input Received: {voice_text}")
            
            # Process voice input through chat system
            if st.session_state.current_page == "ai_chat":
                # Add to chat messages
                st.session_state.messages.append({
                    'role': 'user',
                    'content': voice_text,
                    'timestamp': datetime.now(),
                    'input_method': 'voice'
                })
                
                # Get AI response
                response_data = get_ai_response(voice_text)
                
                st.session_state.messages.append({
                    'role': 'assistant',
                    'content': response_data['response'],
                    'timestamp': datetime.now(),
                    'profile_integrated': response_data.get('profile_integrated', False)
                })
                
                # Optionally speak the response
                if st.session_state.get('auto_speak_responses', False):
                    ai_agent.voice_service.text_to_speech(response_data['response'])
                
                st.rerun()
            else:
                # Store voice input for use in current page
                st.session_state.last_voice_input = voice_text
        else:
            st.error(voice_text)  # Display error message
            
    except Exception as e:
        st.error(f"❌ Voice input failed: {e}")
        logger.error(f"Voice input error: {e}")

def test_text_to_speech():
    """Test text-to-speech functionality"""
    try:
        ai_agent = get_enhanced_ai_agent()
        test_message = "Hello! This is a test of the text-to-speech system for UEL AI Assistant."
        
        with st.spinner("🔊 Testing text-to-speech..."):
            success = ai_agent.voice_service.text_to_speech(test_message)
        
        if success:
            st.success("🔊 Text-to-speech test successful!")
        else:
            st.error("❌ Text-to-speech test failed")
            
    except Exception as e:
        st.error(f"❌ Text-to-speech error: {e}")

def render_voice_settings_panel():
    """Render voice settings configuration panel"""
    st.markdown("#### ⚙️ Voice Settings")
    
    with st.form("voice_settings"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.checkbox(
                "Auto-speak AI responses",
                value=st.session_state.get('auto_speak_responses', False),
                key="setting_auto_speak",
                help="Automatically speak AI responses aloud"
            )
            
            st.checkbox(
                "Voice input shortcuts",
                value=st.session_state.get('voice_shortcuts_enabled', True),
                key="setting_voice_shortcuts",
                help="Enable voice input shortcuts on all pages"
            )
        
        with col2:
            st.slider(
                "Speech Speed",
                min_value=0.5,
                max_value=2.0,
                value=st.session_state.get('speech_speed', 1.0),
                step=0.1,
                key="setting_speech_speed",
                help="Adjust text-to-speech speed"
            )
            
            st.slider(
                "Speech Volume",
                min_value=0.0,
                max_value=1.0,
                value=st.session_state.get('speech_volume', 0.8),
                step=0.1,
                key="setting_speech_volume",
                help="Adjust text-to-speech volume"
            )
        
        submitted = st.form_submit_button("💾 Save Voice Settings")
        if submitted:
            # Save voice settings
            st.session_state.auto_speak_responses = st.session_state.setting_auto_speak
            st.session_state.voice_shortcuts_enabled = st.session_state.setting_voice_shortcuts
            st.session_state.speech_speed = st.session_state.setting_speech_speed
            st.session_state.speech_volume = st.session_state.setting_speech_volume
            
            st.success("✅ Voice settings saved!")
    
    if st.button("❌ Close Settings", key="__close_settings_3010"):
        st.session_state.show_voice_settings = False
        st.rerun()

# =============================================================================
# EXPORT AND REPORTING SYSTEM (RESTORED)
# =============================================================================

def render_export_and_reporting_interface():
    """Render comprehensive export and reporting interface"""
    st.markdown("### 📤 Export & Reporting")
    
    # Export options
    export_col1, export_col2, export_col3 = st.columns(3)
    
    with export_col1:
        if st.button("📄 Export Profile", use_container_width=True, key="__export_profile_3011"):
            export_profile_data()
    
    with export_col2:
        if st.button("📊 Generate Report", use_container_width=True, key="__generate_report_3012"):
            generate_comprehensive_report()
    
    with export_col3:
        if st.button("📁 Download All Data", use_container_width=True, key="__download_all_data_3013"):
            download_all_user_data()
    
    # Report generation options
    st.markdown("#### 📋 Custom Report Generation")
    
    with st.form("report_generation"):
        report_type = st.selectbox(
            "Report Type",
            ["Profile Summary", "Course Recommendations", "Admission Analysis", "Document Status", "Activity Summary"]
        )
        
        report_format = st.selectbox(
            "Export Format",
            ["PDF", "Excel", "JSON", "CSV"]
        )
        
        include_charts = st.checkbox("Include Charts and Visualizations", value=True)
        include_raw_data = st.checkbox("Include Raw Data", value=False)
        
        if st.form_submit_button("📊 Generate Custom Report"):
            generate_custom_report(report_type, report_format, include_charts, include_raw_data)

def export_profile_data():
    """Export profile data in multiple formats"""
    try:
        profile = st.session_state.current_profile
        if not profile:
            st.error("No profile data to export")
            return
        
        # Prepare export data
        export_data = {
            "profile_info": profile,
            "export_timestamp": datetime.now().isoformat(),
            "export_version": "1.0",
            "verification_results": st.session_state.verification_results,
            "interaction_history": st.session_state.interaction_history[-50:],  # Last 50 interactions
            "recommendation_cache": st.session_state.recommendation_cache
        }
        
        # JSON export
        json_data = json.dumps(export_data, indent=2, default=str)
        
        st.download_button(
            label="📥 Download Profile (JSON)",
            data=json_data,
            file_name=f"uel_profile_{profile.get('first_name', 'user')}_{datetime.now().strftime('%Y%m%d')}.json",
            mime="application/json"
        )
        
        # CSV export for tabular data
        profile_df = pd.DataFrame([profile])
        csv_data = profile_df.to_csv(index=False)
        
        st.download_button(
            label="📊 Download Profile (CSV)",
            data=csv_data,
            file_name=f"uel_profile_{profile.get('first_name', 'user')}_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )
        
        st.success("✅ Profile export prepared successfully!")
        
    except Exception as e:
        st.error(f"❌ Export failed: {e}")
        logger.error(f"Profile export error: {e}")

def generate_comprehensive_report():
    """Generate comprehensive PDF report"""
    try:
        profile = st.session_state.current_profile
        if not profile:
            st.error("No profile data available for report generation")
            return
        
        with st.spinner("📊 Generating comprehensive report..."):
            # Simulate report generation
            time.sleep(3)
            
            # Create report data structure
            report_data = {
                "report_title": f"UEL AI Assistant Report - {profile.get('first_name', 'Student')} {profile.get('last_name', '')}",
                "generation_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "profile_summary": profile,
                "recommendations": get_course_recommendations()[:5],  # Top 5 recommendations
                "analytics": generate_analytics_data(),
                "document_status": st.session_state.verification_results,
                "next_steps": generate_next_steps_recommendations(profile)
            }
            
            # Create downloadable report content
            report_content = create_report_content(report_data)
            
            st.download_button(
                label="📥 Download Comprehensive Report",
                data=report_content,
                file_name=f"uel_comprehensive_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain"
            )
            
            st.success("✅ Comprehensive report generated successfully!")
    
    except Exception as e:
        st.error(f"❌ Report generation failed: {e}")
        logger.error(f"Report generation error: {e}")

def create_report_content(report_data: Dict) -> str:
    """Create formatted report content"""
    content = f"""
# {report_data['report_title']}

Generated on: {report_data['generation_date']}

## Profile Summary
Name: {report_data['profile_summary'].get('first_name', '')} {report_data['profile_summary'].get('last_name', '')}
Primary Interest: {report_data['profile_summary'].get('field_of_interest', 'Not specified')}
Academic Level: {report_data['profile_summary'].get('academic_level', 'Not specified')}
GPA: {report_data['profile_summary'].get('gpa', 'Not provided')}/4.0
IELTS Score: {report_data['profile_summary'].get('ielts_score', 'Not provided')}/9.0
Country: {report_data['profile_summary'].get('country', 'Not specified')}

## Course Recommendations
"""
    
    recommendations = report_data.get('recommendations', [])
    for i, rec in enumerate(recommendations, 1):
        content += f"""
{i}. {rec.get('course_name', 'Unknown Course')}
   Match Score: {rec.get('score', 0):.1%}
   Level: {rec.get('level', 'Unknown')}
   Department: {rec.get('department', 'Unknown')}
   
"""
    
    content += f"""
## Document Verification Status
Total Documents: {len(report_data.get('document_status', {}))}
Verified: {sum(1 for r in report_data.get('document_status', {}).values() if r.get('verification_status') == 'verified')}
Pending: {sum(1 for r in report_data.get('document_status', {}).values() if r.get('verification_status') == 'needs_review')}

## Next Steps Recommendations
"""
    
    next_steps = report_data.get('next_steps', [])
    for step in next_steps:
        content += f"• {step}\n"
    
    content += f"""

## Report Footer
This report was generated by UEL AI Assistant.
For questions or support, contact: admissions@uel.ac.uk

Report ID: {datetime.now().strftime('%Y%m%d_%H%M%S')}
"""
    
    return content

# =============================================================================
# ADVANCED SEARCH AND FILTERING SYSTEM (RESTORED)
# =============================================================================

def render_advanced_search_interface():
    """Render advanced search and filtering interface"""
    st.markdown("### 🔍 Advanced Search & Filters")
    
    # Search input with voice option
    col1, col2 = st.columns([4, 1])
    
    with col1:
        search_query = st.text_input(
            "🔍 Search courses, programs, or ask questions",
            placeholder="Search for computer science, business management, admission requirements...",
            key="advanced_search_query"
        )
    
    with col2:
        if st.button("🎤 Voice Search", key="__voice_search_3014"):
            handle_voice_search()
    
    # Advanced filtering options
    with st.expander("🔧 Advanced Filters", expanded=True):
        filter_col1, filter_col2, filter_col3, filter_col4 = st.columns(4)
        
        with filter_col1:
            level_filter = st.multiselect(
                "Academic Level",
                ["undergraduate", "postgraduate", "masters", "phd"],
                key="filter_level"
            )
            
            department_filter = st.multiselect(
                "Department",
                ["School of Computing", "Business School", "School of Engineering", "School of Psychology"],
                key="filter_department"
            )
        
        with filter_col2:
            fee_range = st.slider(
                "Fee Range (£)",
                min_value=0,
                max_value=30000,
                value=(0, 25000),
                step=1000,
                key="filter_fee_range"
            )
            
            duration_filter = st.multiselect(
                "Duration",
                ["1 year", "2 years", "3 years", "4 years"],
                key="filter_duration"
            )
        
        with filter_col3:
            gpa_requirement = st.slider(
                "Minimum GPA Requirement",
                min_value=2.0,
                max_value=4.0,
                value=2.0,
                step=0.1,
                key="filter_gpa"
            )
            
            ielts_requirement = st.slider(
                "Minimum IELTS Requirement",
                min_value=5.0,
                max_value=8.0,
                value=5.0,
                step=0.5,
                key="filter_ielts"
            )
        
        with filter_col4:
            trending_filter = st.checkbox(
                "Show only trending courses",
                key="filter_trending"
            )
            
            available_filter = st.checkbox(
                "Available for next intake",
                value=True,
                key="filter_available"
            )
    
    # Execute search and filtering
    if st.button("🔍 Apply Filters & Search", type="primary", use_container_width=True, key="__apply_filters___se_3015"):
        execute_advanced_search(search_query, {
            'level': level_filter,
            'department': department_filter,
            'fee_range': fee_range,
            'duration': duration_filter,
            'min_gpa': gpa_requirement,
            'min_ielts': ielts_requirement,
            'trending_only': trending_filter,
            'available_only': available_filter
        })

def execute_advanced_search(query: str, filters: Dict):
    """Execute advanced search with filters"""
    try:
        ai_agent = get_enhanced_ai_agent()
        courses_df = ai_agent.data_manager.courses_df
        
        if courses_df.empty:
            st.warning("No course data available for search")
            return
        
        # Apply filters
        filtered_df = apply_course_filters(courses_df, filters)
        
        # Apply text search if query provided
        if query:
            search_results = ai_agent.data_manager.intelligent_search(query)
            course_names = [result['data'].get('course_name', '') for result in search_results if result['type'] == 'course']
            
            if course_names:
                filtered_df = filtered_df[filtered_df['course_name'].isin(course_names)]
        
        # Display results
        display_search_results(filtered_df, query, filters)
        
    except Exception as e:
        st.error(f"❌ Search failed: {e}")
        logger.error(f"Advanced search error: {e}")

def apply_course_filters(courses_df: pd.DataFrame, filters: Dict) -> pd.DataFrame:
    """Apply filtering criteria to courses dataframe"""
    filtered_df = courses_df.copy()
    
    # Level filter
    if filters['level']:
        filtered_df = filtered_df[filtered_df['level'].isin(filters['level'])]
    
    # Department filter
    if filters['department']:
        filtered_df = filtered_df[filtered_df['department'].isin(filters['department'])]
    
    # Fee range filter
    fee_min, fee_max = filters['fee_range']
    if 'fees_international' in filtered_df.columns:
        filtered_df = filtered_df[
            (filtered_df['fees_international'] >= fee_min) & 
            (filtered_df['fees_international'] <= fee_max)
        ]
    
    # Duration filter
    if filters['duration']:
        filtered_df = filtered_df[filtered_df['duration'].isin(filters['duration'])]
    
    # GPA requirement filter
    if 'min_gpa' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['min_gpa'] <= filters['min_gpa']]
    
    # IELTS requirement filter
    if 'min_ielts' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['min_ielts'] <= filters['min_ielts']]
    
    # Trending filter
    if filters['trending_only'] and 'trending_score' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['trending_score'] >= 7.0]
    
    return filtered_df

def display_search_results(results_df: pd.DataFrame, query: str, filters: Dict):
    """Display search results with enhanced formatting"""
    if results_df.empty:
        st.warning("🔍 No courses found matching your criteria. Try adjusting your filters.")
        return
    
    st.success(f"🎯 Found {len(results_df)} courses matching your criteria")
    
    # Results summary
    col1, col2, col3 = st.columns(3)
    
    with col1:
        avg_fee = results_df['fees_international'].mean() if 'fees_international' in results_df.columns else 0
        st.metric("💰 Average Fee", f"£{avg_fee:,.0f}")
    
    with col2:
        if 'level' in results_df.columns:
            most_common_level = results_df['level'].mode().iloc[0] if not results_df['level'].mode().empty else 'N/A'
            st.metric("🎓 Most Common Level", most_common_level)
    
    with col3:
        if 'trending_score' in results_df.columns:
            avg_trending = results_df['trending_score'].mean()
            st.metric("🔥 Average Popularity", f"{avg_trending:.1f}/10")
    
    # Display results with comparison option
    st.markdown("### 📋 Search Results")
    
    for idx, course in results_df.iterrows():
        render_search_result_card(course, idx)

def render_search_result_card(course: pd.Series, index: int):
    """Render individual search result card with comparison option"""
    with st.container():
        # Course header
        col1, col2, col3 = st.columns([3, 1, 1])
        
        with col1:
            st.markdown(f"#### 🎓 {course.get('course_name', 'Unknown Course')}")
            st.caption(f"📍 {course.get('department', 'Unknown Department')} • ⏱️ {course.get('duration', 'Unknown')} • 📊 {course.get('level', 'Unknown').title()}")
        
        with col2:
            trending_score = course.get('trending_score', 5.0)
            st.metric("🔥 Popularity", f"{trending_score}/10")
        
        with col3:
            fee = course.get('fees_international', 0)
            st.metric("💰 Fee", f"£{fee:,}")
        
        # Course details
        description = course.get('description', 'No description available')
        st.markdown(f"**Description:** {description}")
        
        # Requirements and details
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            min_gpa = course.get('min_gpa', 'N/A')
            st.markdown(f"**Min GPA:** {min_gpa}")
        
        with col2:
            min_ielts = course.get('min_ielts', 'N/A')
            st.markdown(f"**Min IELTS:** {min_ielts}")
        
        with col3:
            keywords = course.get('keywords', 'N/A')
            if len(str(keywords)) > 50:
                keywords = str(keywords)[:50] + "..."
            st.markdown(f"**Keywords:** {keywords}")
        
        with col4:
            career_prospects = course.get('career_prospects', 'Various opportunities')
            if len(str(career_prospects)) > 50:
                career_prospects = str(career_prospects)[:50] + "..."
            st.markdown(f"**Careers:** {career_prospects}")
        
        # Action buttons
        button_col1, button_col2, button_col3, button_col4 = st.columns(4)
        
        with button_col1:
            if st.button(f"💬 Ask AI", key=f"ask_ai_search_{index}"):
                switch_to_chat_with_course_context(course.get('course_name'))
        
        with button_col2:
            if st.button(f"🔮 Check Chances", key=f"predict_search_{index}"):
                switch_to_prediction_with_course(course.get('course_name'))
        
        with button_col3:
            comparison_key = f"compare_search_{index}"
            if st.button(f"⚖️ Compare", key=comparison_key):
                add_to_comparison(course)
        
        with button_col4:
            if st.button(f"❤️ Save", key=f"save_search_{index}"):
                save_course_to_favorites(course.get('course_name'))
        
        st.markdown("---")

# =============================================================================
# COURSE COMPARISON SYSTEM (RESTORED)
# =============================================================================

def render_course_comparison_interface():
    """Render course comparison interface"""
    st.markdown("### ⚖️ Course Comparison")
    
    selected_courses = st.session_state.get('selected_courses_for_comparison', [])
    
    if len(selected_courses) < 2:
        st.info("🔍 Select at least 2 courses from search results to compare them.")
        return
    
    # Comparison table
    comparison_df = create_comparison_dataframe(selected_courses)
    
    if not comparison_df.empty:
        st.markdown(f"#### Comparing {len(selected_courses)} Courses")
        
        # Interactive comparison table
        st.dataframe(
            comparison_df,
            use_container_width=True,
            height=400
        )
        
        # Comparison charts
        col1, col2 = st.columns(2)
        
        with col1:
            # Fee comparison
            fig = px.bar(
                comparison_df,
                x='Course Name',
                y='International Fee (£)',
                title="Fee Comparison",
                color='International Fee (£)',
                color_continuous_scale='viridis'
            )
            # Fixed: Use update_layout instead of update_xaxis
            fig.update_layout(xaxis={'tickangle': 45})
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Requirements comparison
            fig = px.scatter(
                comparison_df,
                x='Min GPA',
                y='Min IELTS',
                size='International Fee (£)',
                color='Course Name',
                title="Requirements Comparison",
                hover_data=['Course Name', 'Department']
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # Comparison actions
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📊 Generate Comparison Report", key="__generate_compariso_3016"):
            generate_comparison_report(selected_courses)
    
    with col2:
        if st.button("🔄 Clear Comparison", key="__clear_comparison_3017"):
            st.session_state.selected_courses_for_comparison = []
            st.rerun()
    
    with col3:
        if st.button("❤️ Save Comparison", key="___save_comparison_3018"):
            save_course_comparison(selected_courses)

# =============================================================================
# UTILITY FUNCTIONS FOR ENHANCED FEATURES (RESTORED)
# =============================================================================

def generate_analytics_data() -> Dict:
    """Generate sample analytics data"""
    return {
        'active_users': random.randint(150, 250),
        'user_growth': random.randint(5, 15),
        'total_interactions': random.randint(800, 1200),
        'interaction_growth': random.randint(20, 50),
        'recommendations_generated': random.randint(300, 500),
        'recommendation_growth': random.randint(10, 25),
        'predictions_made': random.randint(100, 200),
        'prediction_growth': random.randint(15, 30),
        'documents_verified': random.randint(50, 100),
        'profile_updates': random.randint(200, 350)
    }

def generate_user_activity_timeline(profile: Dict) -> List[Dict]:
    """Generate user activity timeline data"""
    activities = []
    start_date = datetime.now() - timedelta(days=30)
    
    activity_types = [
        {"activity": "Profile Created", "category": "Profile", "duration": 1},
        {"activity": "AI Chat Session", "category": "Interaction", "duration": 1},
        {"activity": "Course Recommendations", "category": "Recommendations", "duration": 1},
        {"activity": "Admission Prediction", "category": "Predictions", "duration": 1},
        {"activity": "Document Upload", "category": "Documents", "duration": 1}
    ]
    
    for i, activity_type in enumerate(activity_types):
        activity_date = start_date + timedelta(days=i*3)
        activities.append({
            "activity": activity_type["activity"],
            "category": activity_type["category"],
            "start_date": activity_date,
            "end_date": activity_date + timedelta(hours=activity_type["duration"])
        })
    
    return activities

def categorize_interests(interests: List[str]) -> Dict[str, int]:
    """Categorize user interests"""
    categories = {
        "Technology": ["Artificial Intelligence", "Data Science", "Cybersecurity", "Web Development", "Software Engineering"],
        "Business": ["Business Management", "Finance", "Marketing", "Human Resources", "Entrepreneurship"],
        "Healthcare": ["Medicine", "Nursing", "Healthcare", "Biology", "Chemistry"],
        "Social Sciences": ["Psychology", "Counseling", "Social Work", "Education", "Research"],
        "Creative": ["Arts", "Creative Writing", "Design", "Media Studies", "Film Production"],
        "Legal/Political": ["Law", "International Relations", "Politics", "History", "Philosophy"]
    }
    
    interest_counts = {category: 0 for category in categories.keys()}
    
    for interest in interests:
        for category, category_interests in categories.items():
            if interest in category_interests:
                interest_counts[category] += 1
                break
    
    return interest_counts

def generate_prediction_analytics_data() -> Dict:
    """Generate prediction analytics data"""
    return {
        'model_accuracy': 0.85 + random.random() * 0.10,
        'total_predictions': random.randint(500, 800),
        'prediction_growth': random.randint(10, 25),
        'avg_response_time': 1.2 + random.random() * 0.8
    }

def get_system_performance_data() -> Dict:
    """Get system performance metrics"""
    return {
        'uptime': 0.95 + random.random() * 0.04,
        'avg_response_time': 1.0 + random.random() * 0.5,
        'response_time_delta': (random.random() - 0.5) * 0.2,
        'memory_usage': 60 + random.random() * 20,
        'cache_hit_rate': 0.80 + random.random() * 0.15
    }

def generate_next_steps_recommendations(profile: Dict) -> List[str]:
    """Generate personalized next steps recommendations"""
    recommendations = []
    
    completion = profile.get('profile_completion', 0)
    if completion < 80:
        recommendations.append("Complete your profile to get better recommendations")
    
    if not profile.get('career_goals'):
        recommendations.append("Add your career goals for more targeted guidance")
    
    if profile.get('gpa', 0) < 3.0:
        recommendations.append("Consider improving your GPA for better admission chances")
    
    if profile.get('ielts_score', 0) < 6.5:
        recommendations.append("Take IELTS test or improve your score")
    
    if not st.session_state.verification_results:
        recommendations.append("Upload and verify your academic documents")
    
    recommendations.extend([
        "Apply for courses that match your profile",
        "Contact admissions team for personalized guidance",
        "Explore scholarship opportunities",
        "Prepare for application interviews"
    ])
    
    return recommendations[:5]  # Return top 5 recommendations

# =============================================================================
# ADDITIONAL HELPER FUNCTIONS (RESTORED)
# =============================================================================

def save_profile_changes():
    """Save profile changes from editor"""
    try:
        # Collect all edited values
        updated_profile = st.session_state.current_profile.copy()
        
        # Basic info updates
        if 'edit_first_name' in st.session_state:
            updated_profile['first_name'] = st.session_state.edit_first_name
        if 'edit_last_name' in st.session_state:
            updated_profile['last_name'] = st.session_state.edit_last_name
        if 'edit_email' in st.session_state:
            updated_profile['email'] = st.session_state.edit_email
        
        # Academic info updates
        if 'edit_gpa' in st.session_state:
            updated_profile['gpa'] = st.session_state.edit_gpa
        if 'edit_ielts_score' in st.session_state:
            updated_profile['ielts_score'] = st.session_state.edit_ielts_score
        
        # Update profile completion
        profile_obj = UserProfile.from_dict(updated_profile)
        profile_obj.calculate_completion()
        
        # Save to session state
        st.session_state.current_profile = profile_obj.to_dict()
        
        # Save to AI system
        ai_agent = get_enhanced_ai_agent()
        ai_agent.profile_manager.save_profile(profile_obj)
        
        st.success("✅ Profile changes saved successfully!")
        
    except Exception as e:
        st.error(f"❌ Failed to save changes: {e}")
        logger.error(f"Profile save error: {e}")

def reset_profile_changes():
    """Reset profile changes to original values"""
    if st.session_state.get('profile_backup'):
        st.session_state.current_profile = st.session_state.profile_backup
        st.success("✅ Changes reset to original values")
    else:
        st.warning("No backup available to reset to")
    st.rerun()

def handle_voice_search():
    """Handle voice search input"""
    try:
        ai_agent = get_enhanced_ai_agent()
        
        with st.spinner("🎧 Listening for search query..."):
            voice_query = ai_agent.voice_service.speech_to_text()
        
        if voice_query and not voice_query.startswith("❌"):
            st.session_state.advanced_search_query = voice_query
            st.success(f"🎤 Voice search: {voice_query}")
            st.rerun()
        else:
            st.error(voice_query if voice_query else "Voice search failed")
            
    except Exception as e:
        st.error(f"❌ Voice search failed: {e}")

def switch_to_chat_with_course_context(course_name: str):
    """Switch to chat page with course context"""
    st.session_state.current_page = "ai_chat"
    st.session_state.course_context = course_name
    # Add initial message about the course
    initial_message = f"Tell me more about the {course_name} program at UEL."
    st.session_state.messages.append({
        'role': 'user',
        'content': initial_message,
        'timestamp': datetime.now(),
        'context': 'course_inquiry'
    })
    st.rerun()

def switch_to_prediction_with_course(course_name: str):
    """Switch to prediction page with pre-selected course"""
    st.session_state.current_page = "predictions"
    st.session_state.selected_course_for_prediction = course_name
    st.rerun()

def add_to_comparison(course: pd.Series):
    """Add course to comparison list"""
    if 'selected_courses_for_comparison' not in st.session_state:
        st.session_state.selected_courses_for_comparison = []
    
    course_data = course.to_dict()
    
    # Check if already in comparison
    existing_names = [c.get('course_name') for c in st.session_state.selected_courses_for_comparison]
    if course_data.get('course_name') not in existing_names:
        st.session_state.selected_courses_for_comparison.append(course_data)
        st.success(f"✅ Added {course_data.get('course_name')} to comparison")
        
        if len(st.session_state.selected_courses_for_comparison) >= 2:
            st.info("🎯 You can now view the course comparison!")
    else:
        st.warning("Course already in comparison list")

def save_course_to_favorites(course_name: str):
    """Save course to user favorites"""
    try:
        if st.session_state.current_profile:
            profile = st.session_state.current_profile
            preferred_courses = profile.get('preferred_courses', [])
            
            if isinstance(preferred_courses, str):
                preferred_courses = [preferred_courses] if preferred_courses else []
            elif not isinstance(preferred_courses, list):
                preferred_courses = []
            
            if course_name not in preferred_courses:
                preferred_courses.append(course_name)
                profile['preferred_courses'] = preferred_courses
                
                # Update session state
                st.session_state.current_profile = profile
                
                # Save to AI system
                ai_agent = get_enhanced_ai_agent()
                profile_obj = UserProfile.from_dict(profile)
                ai_agent.profile_manager.save_profile(profile_obj)
                
                st.success(f"❤️ {course_name} saved to favorites!")
            else:
                st.info("Course already in favorites")
        else:
            st.info("Create a profile to save favorite courses!")
            
    except Exception as e:
        st.error(f"Failed to save course: {e}")
        logger.error(f"Save course error: {e}")

def create_comparison_dataframe(courses: List[Dict]) -> pd.DataFrame:
    """Create comparison dataframe from selected courses"""
    try:
        comparison_data = []
        
        for course in courses:
            comparison_data.append({
                'Course Name': course.get('course_name', 'Unknown'),
                'Level': course.get('level', 'Unknown'),
                'Department': course.get('department', 'Unknown'),
                'Duration': course.get('duration', 'Unknown'),
                'International Fee (£)': course.get('fees_international', 0),
                'Domestic Fee (£)': course.get('fees_domestic', 0),
                'Min GPA': course.get('min_gpa', 0),
                'Min IELTS': course.get('min_ielts', 0),
                'Trending Score': course.get('trending_score', 0),
                'Description': course.get('description', 'No description')[:100] + "..."
            })
        
        return pd.DataFrame(comparison_data)
        
    except Exception as e:
        logger.error(f"Comparison dataframe creation error: {e}")
        return pd.DataFrame()

def generate_comparison_report(courses: List[Dict]):
    """Generate comparison report for selected courses"""
    try:
        comparison_df = create_comparison_dataframe(courses)
        
        if comparison_df.empty:
            st.error("No data available for comparison report")
            return
        
        # Generate report content
        report_content = f"""
# Course Comparison Report

Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Courses Being Compared
{', '.join([course.get('course_name', 'Unknown') for course in courses])}

## Comparison Summary

### Fee Analysis
- Highest Fee: £{comparison_df['International Fee (£)'].max():,} ({comparison_df.loc[comparison_df['International Fee (£)'].idxmax(), 'Course Name']})
- Lowest Fee: £{comparison_df['International Fee (£)'].min():,} ({comparison_df.loc[comparison_df['International Fee (£)'].idxmin(), 'Course Name']})
- Average Fee: £{comparison_df['International Fee (£)'].mean():,.0f}

### Academic Requirements
- Highest GPA Requirement: {comparison_df['Min GPA'].max():.1f} ({comparison_df.loc[comparison_df['Min GPA'].idxmax(), 'Course Name']})
- Lowest GPA Requirement: {comparison_df['Min GPA'].min():.1f} ({comparison_df.loc[comparison_df['Min GPA'].idxmin(), 'Course Name']})
- Highest IELTS Requirement: {comparison_df['Min IELTS'].max():.1f} ({comparison_df.loc[comparison_df['Min IELTS'].idxmax(), 'Course Name']})

### Popularity Rankings
"""
        
        # Add individual course details
        for course in courses:
            report_content += f"""
## {course.get('course_name', 'Unknown Course')}
- **Department:** {course.get('department', 'Unknown')}
- **Level:** {course.get('level', 'Unknown')}
- **Duration:** {course.get('duration', 'Unknown')}
- **International Fee:** £{course.get('fees_international', 0):,}
- **Min GPA:** {course.get('min_gpa', 0)}
- **Min IELTS:** {course.get('min_ielts', 0)}
- **Trending Score:** {course.get('trending_score', 0)}/10
- **Description:** {course.get('description', 'No description available')}

"""
        
        report_content += """
## Recommendation
Based on this comparison, consider factors such as:
- Your academic qualifications vs. requirements
- Budget considerations
- Career alignment with course content
- Course popularity and industry demand

For personalized advice, consult with UEL admissions team.

---
Report generated by UEL AI Assistant
"""
        
        # Download button
        st.download_button(
            label="📥 Download Comparison Report",
            data=report_content,
            file_name=f"course_comparison_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            mime="text/plain"
        )
        
        st.success("✅ Comparison report generated successfully!")
        
    except Exception as e:
        st.error(f"❌ Failed to generate comparison report: {e}")
        logger.error(f"Comparison report error: {e}")

def save_course_comparison(courses: List[Dict]):
    """Save course comparison for later reference"""
    try:
        comparison_data = {
            'comparison_id': f"comp_{int(time.time())}",
            'courses': courses,
            'created_date': datetime.now().isoformat(),
            'profile_id': st.session_state.current_profile.get('id') if st.session_state.current_profile else None
        }
        
        # Store in session state
        if 'saved_comparisons' not in st.session_state:
            st.session_state.saved_comparisons = []
        
        st.session_state.saved_comparisons.append(comparison_data)
        
        st.success("✅ Course comparison saved successfully!")
        
    except Exception as e:
        st.error(f"❌ Failed to save comparison: {e}")
        logger.error(f"Save comparison error: {e}")

def download_all_user_data():
    """Download all user data in a comprehensive package"""
    try:
        with st.spinner("📦 Preparing comprehensive data package..."):
            # Collect all user data
            all_data = {
                'profile': st.session_state.current_profile,
                'verification_results': st.session_state.verification_results,
                'interaction_history': st.session_state.interaction_history,
                'messages': st.session_state.messages,
                'recommendation_cache': st.session_state.recommendation_cache,
                'prediction_results': st.session_state.prediction_results,
                'feature_usage_stats': st.session_state.feature_usage_stats,
                'saved_comparisons': st.session_state.get('saved_comparisons', []),
                'export_metadata': {
                    'export_date': datetime.now().isoformat(),
                    'data_version': '1.0',
                    'profile_completion': st.session_state.current_profile.get('profile_completion', 0) if st.session_state.current_profile else 0
                }
            }
            
            # Create JSON export
            json_data = json.dumps(all_data, indent=2, default=str)
            
            st.download_button(
                label="📦 Download Complete Data Package (JSON)",
                data=json_data,
                file_name=f"uel_complete_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
            
            st.success("✅ Complete data package ready for download!")
            
    except Exception as e:
        st.error(f"❌ Failed to prepare data package: {e}")
        logger.error(f"Data package error: {e}")

def generate_custom_report(report_type: str, format_type: str, include_charts: bool, include_raw_data: bool):
    """Generate custom report based on user specifications"""
    try:
        with st.spinner(f"📊 Generating {report_type} report in {format_type} format..."):
            profile = st.session_state.current_profile
            
            if report_type == "Profile Summary":
                content = generate_profile_summary_report(profile, include_charts, include_raw_data)
            elif report_type == "Course Recommendations":
                content = generate_recommendations_report(include_charts, include_raw_data)
            elif report_type == "Admission Analysis":
                content = generate_admission_analysis_report(include_charts, include_raw_data)
            elif report_type == "Document Status":
                content = generate_document_status_report(include_charts, include_raw_data)
            elif report_type == "Activity Summary":
                content = generate_activity_summary_report(profile, include_charts, include_raw_data)
            else:
                content = "Report type not supported"
            
            # Format-specific processing
            if format_type == "JSON":
                content = json.dumps({"report": content, "metadata": {"type": report_type, "generated": datetime.now().isoformat()}}, indent=2)
                mime_type = "application/json"
                file_extension = "json"
            elif format_type == "CSV":
                # Convert to CSV if possible
                content = convert_report_to_csv(content, report_type)
                mime_type = "text/csv"
                file_extension = "csv"
            else:  # PDF or default text
                mime_type = "text/plain"
                file_extension = "txt"
            
            # Download button
            filename = f"uel_{report_type.lower().replace(' ', '_')}_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{file_extension}"
            
            st.download_button(
                label=f"📥 Download {report_type} Report ({format_type})",
                data=content,
                file_name=filename,
                mime=mime_type
            )
            
            st.success(f"✅ {report_type} report generated successfully!")
            
    except Exception as e:
        st.error(f"❌ Custom report generation failed: {e}")
        logger.error(f"Custom report error: {e}")

def generate_profile_summary_report(profile: Dict, include_charts: bool, include_raw_data: bool) -> str:
    """Generate profile summary report"""
    content = f"""
# Profile Summary Report

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Profile ID:** {profile.get('id', 'N/A')}

## Personal Information
- **Name:** {profile.get('first_name', '')} {profile.get('last_name', '')}
- **Email:** {profile.get('email', 'Not provided')}
- **Country:** {profile.get('country', 'Not specified')}
- **Nationality:** {profile.get('nationality', 'Not specified')}

## Academic Background
- **Current Level:** {profile.get('academic_level', 'Not specified')}
- **Field of Interest:** {profile.get('field_of_interest', 'Not specified')}
- **Institution:** {profile.get('current_institution', 'Not specified')}
- **GPA:** {profile.get('gpa', 'Not provided')}/4.0
- **IELTS Score:** {profile.get('ielts_score', 'Not provided')}/9.0

## Career Goals
{profile.get('career_goals', 'Not specified')}

## Areas of Interest
{', '.join(profile.get('interests', [])) if profile.get('interests') else 'Not specified'}

## Profile Statistics
- **Profile Completion:** {profile.get('profile_completion', 0):.1f}%
- **Interactions:** {profile.get('interaction_count', 0)}
- **Created:** {profile.get('created_date', 'Unknown')}
- **Last Updated:** {profile.get('updated_date', 'Unknown')}
"""
    
    if include_raw_data:
        content += f"\n## Raw Profile Data\n{json.dumps(profile, indent=2, default=str)}\n"
    
    return content

def generate_recommendations_report(include_charts: bool, include_raw_data: bool) -> str:
    """Generate course recommendations report"""
    recommendations = get_course_recommendations()
    
    content = f"""
# Course Recommendations Report

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Total Recommendations:** {len(recommendations)}

## Top Course Matches
"""
    
    for i, rec in enumerate(recommendations[:10], 1):  # Top 10
        content += f"""
### {i}. {rec.get('course_name', 'Unknown Course')}
- **Match Score:** {rec.get('score', 0):.1%}
- **Level:** {rec.get('level', 'Unknown')}
- **Department:** {rec.get('department', 'Unknown')}
- **Duration:** {rec.get('duration', 'Unknown')}
- **Fees:** {rec.get('fees', 'Unknown')}
- **Why Recommended:** {', '.join(rec.get('reasons', [])[:3])}

"""
    
    if include_raw_data:
        content += f"\n## Raw Recommendations Data\n{json.dumps(recommendations, indent=2, default=str)}\n"
    
    return content

def generate_admission_analysis_report(include_charts: bool, include_raw_data: bool) -> str:
    """Generate admission analysis report"""
    prediction_results = st.session_state.prediction_results
    
    content = f"""
# Admission Analysis Report

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Total Predictions:** {len(prediction_results)}

## Admission Chances Summary
"""
    
    if prediction_results:
        for course, result in prediction_results.items():
            prob = result.get('success_probability', 0)
            content += f"""
### {course}
- **Admission Probability:** {prob:.1%}
- **Confidence Level:** {result.get('confidence', 'Unknown')}
- **Key Factors:** {', '.join(result.get('insights', {}).get('strengths', [])[:3])}

"""
    else:
        content += "No prediction data available. Run admission predictions to generate analysis.\n"
    
    return content

def generate_document_status_report(include_charts: bool, include_raw_data: bool) -> str:
    """Generate document status report"""
    verification_results = st.session_state.verification_results
    
    content = f"""
# Document Verification Status Report

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Total Documents:** {len(verification_results)}

## Document Status Summary
"""
    
    if verification_results:
        verified_count = sum(1 for r in verification_results.values() if r.get('verification_status') == 'verified')
        pending_count = sum(1 for r in verification_results.values() if r.get('verification_status') == 'needs_review')
        rejected_count = sum(1 for r in verification_results.values() if r.get('verification_status') == 'rejected')
        
        content += f"""
- **Verified Documents:** {verified_count}
- **Pending Review:** {pending_count}
- **Rejected Documents:** {rejected_count}

## Individual Document Status
"""
        
        for doc_id, result in verification_results.items():
            content += f"""
### Document {doc_id[:8]}...
- **Type:** {result.get('document_type', 'Unknown')}
- **Status:** {result.get('verification_status', 'Unknown')}
- **Confidence:** {result.get('confidence_score', 0):.1%}
- **Issues:** {len(result.get('issues_found', []))} found

"""
    else:
        content += "No documents have been uploaded for verification.\n"
    
    return content

def generate_activity_summary_report(profile: Dict, include_charts: bool, include_raw_data: bool) -> str:
    """Generate activity summary report"""
    content = f"""
# Activity Summary Report

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**User:** {profile.get('first_name', 'Unknown')} {profile.get('last_name', '')}

## Activity Overview
- **Total Interactions:** {profile.get('interaction_count', 0)}
- **Profile Completion:** {profile.get('profile_completion', 0):.1f}%
- **Account Age:** {(datetime.now() - datetime.fromisoformat(profile.get('created_date', datetime.now().isoformat()))).days} days
- **Last Active:** {profile.get('last_active', 'Unknown')}

## Feature Usage
- **Favorite Features:** {', '.join(profile.get('favorite_features', [])) if profile.get('favorite_features') else 'None recorded'}
- **Chat Messages:** {len(st.session_state.messages)}
- **Documents Uploaded:** {len(st.session_state.verification_results)}
- **Courses Saved:** {len(profile.get('preferred_courses', [])) if profile.get('preferred_courses') else 0}

## Recent Activity
"""
    
    # Add recent messages
    recent_messages = st.session_state.messages[-5:] if st.session_state.messages else []
    for i, msg in enumerate(recent_messages, 1):
        if msg.get('role') == 'user':
            content += f"{i}. User: {msg.get('content', '')[:100]}...\n"
    
    return content

def convert_report_to_csv(content: str, report_type: str) -> str:
    """Convert report content to CSV format where possible"""
    try:
        if report_type == "Profile Summary":
            profile = st.session_state.current_profile
            df = pd.DataFrame([profile])
            return df.to_csv(index=False)
        elif report_type == "Course Recommendations":
            recommendations = get_course_recommendations()
            if recommendations:
                df = pd.DataFrame(recommendations)
                return df.to_csv(index=False)
        elif report_type == "Document Status":
            verification_results = st.session_state.verification_results
            if verification_results:
                docs_data = []
                for doc_id, result in verification_results.items():
                    docs_data.append({
                        'Document_ID': doc_id,
                        'Type': result.get('document_type', 'Unknown'),
                        'Status': result.get('verification_status', 'Unknown'),
                        'Confidence': result.get('confidence_score', 0),
                        'Issues_Count': len(result.get('issues_found', []))
                    })
                df = pd.DataFrame(docs_data)
                return df.to_csv(index=False)
        
        # Fallback: return original content
        return content
        
    except Exception as e:
        logger.error(f"CSV conversion error: {e}")
        return content

def show_delete_profile_confirmation():
    """Show profile deletion confirmation dialog"""
    st.warning("⚠️ **Delete Profile Confirmation**")
    st.markdown("This action will permanently delete your profile and all associated data. This cannot be undone.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("❌ Yes, Delete Profile", type="primary", key="__yes__delete_profil_3019"):
            delete_user_profile()
    
    with col2:
        if st.button("🔄 Cancel", key="__cancel_3020"):
            st.info("Profile deletion cancelled")

def delete_user_profile():
    """Delete user profile and all associated data"""
    try:
        # Clear session state
        profile_keys = [
            'current_profile', 'profile_active', 'messages', 'verification_results',
            'recommendation_cache', 'prediction_results', 'interaction_history'
        ]
        
        for key in profile_keys:
            if key in st.session_state:
                del st.session_state[key]
        
        # Reset to initial state
        initialize_comprehensive_session_state()
        
        st.success("✅ Profile deleted successfully")
        st.info("You will now be redirected to create a new profile")
        
        time.sleep(2)
        st.rerun()
        
    except Exception as e:
        st.error(f"❌ Failed to delete profile: {e}")
        logger.error(f"Profile deletion error: {e}")

# =============================================================================
# ENHANCED AI INTEGRATION FUNCTIONS (RESTORED)
# =============================================================================

def get_enhanced_ai_agent() -> 'UELAISystem':
    """Get or initialize enhanced AI agent with profile support"""
    if st.session_state.ai_agent is None:
        with st.spinner("🔄 Initializing Enhanced UEL AI System..."):
            try:
                st.session_state.ai_agent = UELAISystem()
                
                # Set current profile if available
                if st.session_state.get('current_profile'):
                    profile = UserProfile.from_dict(st.session_state.current_profile)
                    st.session_state.ai_agent.profile_manager.set_current_profile(profile)
                
                logger.info("✅ Enhanced AI Agent initialized successfully")
            except Exception as e:
                logger.error(f"❌ Failed to initialize Enhanced AI Agent: {e}")
                st.session_state.ai_agent = create_fallback_agent()
                st.error("⚠️ AI system running in limited mode. Some features may not be available.")
    
    return st.session_state.ai_agent

def get_ai_response(user_input: str) -> Dict:
    """Get AI response with profile integration"""
    try:
        ai_agent = get_enhanced_ai_agent()
        
        # Check if profile is active for enhanced responses
        profile_enhanced = st.session_state.get('profile_active', False)
        
        if profile_enhanced:
            current_profile = st.session_state.current_profile
            profile_obj = UserProfile.from_dict(current_profile)
            
            # Process message with profile context
            response_data = ai_agent.process_user_message(user_input, profile_obj)
            
            return {
                'response': response_data.get('ai_response', 'I apologize, but I could not generate a response.'),
                'requires_profile': False,
                'profile_integrated': True,
                'confidence': 0.85,
                'sentiment': response_data.get('sentiment', {}),
                'search_results': response_data.get('search_results', [])
            }
        else:
            # Provide general response without profile
            general_response = generate_general_response(user_input)
            return {
                'response': general_response,
                'requires_profile': False,
                'profile_integrated': False,
                'confidence': 0.7
            }
            
    except Exception as e:
        logger.error(f"Error getting AI response: {e}")
        return {
            'response': f"I apologize, but I'm experiencing technical difficulties. Please try again or contact support.",
            'requires_profile': False,
            'profile_integrated': False,
            'confidence': 0.5,
            'error': str(e)
        }

def generate_general_response(user_input: str) -> str:
    """Generate general responses for common queries"""
    user_input_lower = user_input.lower()
    
    if any(word in user_input_lower for word in ['program', 'course', 'study']):
        return "UEL offers a wide range of undergraduate and postgraduate programs across various disciplines including Business, Computer Science, Engineering, Psychology, and more. You can explore our full course catalog on the UEL website or speak with our admissions team for personalized guidance."
    
    elif any(word in user_input_lower for word in ['admission', 'apply', 'requirement']):
        return "UEL admission requirements vary by program but typically include academic transcripts, English proficiency test results (IELTS/TOEFL), and a personal statement. International students generally need IELTS 6.0-6.5 or equivalent. Contact admissions@uel.ac.uk for specific requirements."
    
    elif any(word in user_input_lower for word in ['fee', 'cost', 'tuition']):
        return "UEL tuition fees vary by program and student status. International undergraduate fees typically range from £13,000-£15,000 per year, while postgraduate fees range from £14,000-£16,000. Additional costs include accommodation and living expenses. Check the UEL website for current fees."
    
    elif any(word in user_input_lower for word in ['campus', 'facility', 'location']):
        return "UEL's main campus is located in East London, offering modern facilities including libraries, laboratories, sports centers, and student accommodation. The campus is well-connected to central London via public transport and provides a vibrant student community."
    
    elif any(word in user_input_lower for word in ['scholarship', 'financial', 'funding']):
        return "UEL offers various scholarships for international students, including academic excellence scholarships, country-specific awards, and need-based support. Visit the UEL scholarships page or contact the financial aid office for available opportunities."
    
    elif any(word in user_input_lower for word in ['accommodation', 'housing', 'residence']):
        return "UEL provides on-campus accommodation in modern residence halls with various room types and facilities. Off-campus housing options are also available. Apply early for the best accommodation choices and contact the accommodation office for assistance."
    
    else:
        return f"Thank you for your question about UEL. While I can provide general information, I'd be happy to give you more personalized guidance if you create a profile. For immediate assistance, please contact our admissions team at admissions@uel.ac.uk or call +44 20 8223 3000."

def get_course_recommendations() -> List[Dict]:
    """Get course recommendations using profile"""
    try:
        ai_agent = get_enhanced_ai_agent()
        
        if not st.session_state.get('profile_active') or not st.session_state.current_profile:
            logger.warning("Profile not active for course recommendations")
            return []
        
        profile_data = st.session_state.current_profile
        recommendations = ai_agent.course_recommender.recommend_courses(profile_data)
        
        logger.info(f"Generated {len(recommendations)} course recommendations")
        return recommendations
        
    except Exception as e:
        logger.error(f"Error getting course recommendations: {e}")
        return []

def predict_admission_success(course_name: str) -> Dict:
    """Predict admission success using profile"""
    try:
        ai_agent = get_enhanced_ai_agent()
        
        if not st.session_state.get('profile_active') or not st.session_state.current_profile:
            return {
                'error': 'Profile required for admission prediction',
                'success_probability': 0.5,
                'confidence': 0.0
            }
        
        profile_data = st.session_state.current_profile.copy()
        profile_data['course_applied'] = course_name
        
        prediction = ai_agent.predictive_engine.predict_admission_probability(profile_data)
        
        return {
            'success_probability': prediction.get('probability', 0.5),
            'confidence': 0.8 if prediction.get('confidence') == 'high' else 0.6 if prediction.get('confidence') == 'medium' else 0.4,
            'insights': {
                'strengths': prediction.get('factors', []),
                'areas_for_improvement': []
            },
            'recommendations': prediction.get('recommendations', [])
        }
        
    except Exception as e:
        logger.error(f"Error predicting admission success: {e}")
        return {
            'error': str(e),
            'success_probability': 0.5,
            'confidence': 0.0
        }

def create_fallback_agent():
    """Create fallback agent if needed"""
    class FallbackAgent:
        def __init__(self):
            self.profile_manager = None
            self.data_manager = self.create_fallback_data_manager()
            self.course_recommender = self.create_fallback_recommender()
            self.predictive_engine = self.create_fallback_predictor()
            self.document_verifier = self.create_fallback_verifier()
        
        def create_fallback_data_manager(self):
            class FallbackDataManager:
                def __init__(self):
                    self.courses_df = pd.DataFrame({
                        'course_name': ['Business Management', 'Computer Science', 'Psychology', 'Engineering', 'Law', 'Medicine'],
                        'level': ['undergraduate', 'undergraduate', 'undergraduate', 'undergraduate', 'postgraduate', 'postgraduate'],
                        'department': ['Business School', 'School of Computing', 'School of Psychology', 'School of Engineering', 'School of Law', 'School of Medicine'],
                        'description': [
                            'Comprehensive business management program focusing on leadership and strategy',
                            'Cutting-edge computer science program with AI and software engineering focus',
                            'In-depth psychology program covering cognitive and behavioral studies',
                            'Modern engineering program with practical applications',
                            'Comprehensive law program covering various legal disciplines',
                            'Medical program with clinical and research components'
                        ],
                        'duration': ['3 years', '3 years', '3 years', '4 years', '2 years', '5 years'],
                        'fees_international': [14000, 15000, 13500, 16000, 18000, 25000],
                        'fees_domestic': [9250, 9250, 9250, 9250, 12000, 15000],
                        'min_gpa': [3.0, 3.2, 2.8, 3.3, 3.5, 3.8],
                        'min_ielts': [6.0, 6.5, 6.0, 6.5, 7.0, 7.5],
                        'trending_score': [7.5, 9.2, 6.8, 8.1, 7.8, 9.5],
                        'keywords': [
                            'business, management, leadership, strategy, finance',
                            'programming, AI, software, technology, innovation',
                            'behavior, cognition, research, therapy, mental health',
                            'design, construction, innovation, problem-solving',
                            'legal, justice, advocacy, research, policy',
                            'healthcare, research, clinical, patient care'
                        ],
                        'career_prospects': [
                            'Management roles, consulting, entrepreneurship',
                            'Software developer, AI engineer, tech consultant',
                            'Therapist, researcher, counselor, academic',
                            'Engineer, project manager, technical consultant',
                            'Lawyer, legal advisor, policy maker',
                            'Doctor, researcher, healthcare administrator'
                        ]
                    })
                
                def intelligent_search(self, query):
                    # Simple search simulation
                    results = []
                    for _, course in self.courses_df.iterrows():
                        if query.lower() in course['course_name'].lower() or query.lower() in course['keywords'].lower():
                            results.append({
                                'type': 'course',
                                'data': course.to_dict(),
                                'relevance_score': 0.8
                            })
                    return results[:5]  # Return top 5 results
            
            return FallbackDataManager()
        
        def create_fallback_recommender(self):
            class FallbackRecommender:
                def recommend_courses(self, profile_data):
                    field_interest = profile_data.get('field_of_interest', '').lower()
                    gpa = profile_data.get('gpa', 3.0)
                    
                    # Simple matching logic
                    recommendations = []
                    
                    if 'computer' in field_interest or 'technology' in field_interest:
                        recommendations.append({
                            'course_name': 'Computer Science BSc',
                            'score': 0.95,
                            'level': 'undergraduate',
                            'department': 'School of Computing',
                            'description': 'Cutting-edge computer science program with AI and software engineering focus',
                            'duration': '3 years',
                            'fees': '£15,000',
                            'reasons': [
                                'Perfect match for your computer science interests',
                                'Strong industry connections',
                                'Modern curriculum with AI focus',
                                'Excellent career prospects in tech'
                            ]
                        })
                    
                    if 'business' in field_interest or 'management' in field_interest:
                        recommendations.append({
                            'course_name': 'Business Management BSc',
                            'score': 0.90,
                            'level': 'undergraduate',
                            'department': 'Business School',
                            'description': 'Comprehensive business management program focusing on leadership and strategy',
                            'duration': '3 years',
                            'fees': '£14,000',
                            'reasons': [
                                'Aligns with your business interests',
                                'Strong graduate employment rate',
                                'Industry-relevant curriculum',
                                'Leadership development focus'
                            ]
                        })
                    
                    # Add general recommendations based on GPA
                    if gpa >= 3.5:
                        recommendations.append({
                            'course_name': 'Engineering MEng',
                            'score': 0.75,
                            'level': 'undergraduate',
                            'department': 'School of Engineering',
                            'description': 'Modern engineering program with practical applications',
                            'duration': '4 years',
                            'fees': '£16,000',
                            'reasons': [
                                'Your high GPA qualifies you for this competitive program',
                                'Excellent career prospects',
                                'Hands-on learning approach',
                                'Strong industry partnerships'
                            ]
                        })
                    
                    # Default recommendation if none match
                    if not recommendations:
                        recommendations.append({
                            'course_name': 'Liberal Arts BA',
                            'score': 0.60,
                            'level': 'undergraduate',
                            'department': 'School of Arts',
                            'description': 'Broad-based liberal arts program allowing exploration of multiple disciplines',
                            'duration': '3 years',
                            'fees': '£13,000',
                            'reasons': [
                                'Flexible program structure',
                                'Opportunity to explore different subjects',
                                'Develops critical thinking skills',
                                'Good foundation for various careers'
                            ]
                        })
                    
                    return recommendations[:5]  # Return top 5
            
            return FallbackRecommender()
        
        def create_fallback_predictor(self):
            class FallbackPredictor:
                def predict_admission_probability(self, profile_data):
                    gpa = profile_data.get('gpa', 3.0)
                    ielts = profile_data.get('ielts_score', 6.5)
                    work_exp = profile_data.get('work_experience_years', 0)
                    
                    # Simple prediction logic
                    base_prob = 0.5
                    
                    # GPA factor
                    if gpa >= 3.7:
                        base_prob += 0.25
                    elif gpa >= 3.3:
                        base_prob += 0.15
                    elif gpa >= 3.0:
                        base_prob += 0.05
                    else:
                        base_prob -= 0.10
                    
                    # IELTS factor
                    if ielts >= 7.0:
                        base_prob += 0.15
                    elif ielts >= 6.5:
                        base_prob += 0.10
                    elif ielts >= 6.0:
                        base_prob += 0.05
                    else:
                        base_prob -= 0.05
                    
                    # Work experience factor
                    if work_exp >= 2:
                        base_prob += 0.10
                    elif work_exp >= 1:
                        base_prob += 0.05
                    
                    # Cap probability
                    probability = min(max(base_prob, 0.1), 0.95)
                    
                    # Determine confidence and factors
                    confidence = 'high' if probability > 0.7 else 'medium' if probability > 0.4 else 'low'
                    
                    factors = []
                    if gpa >= 3.5:
                        factors.append('Strong academic performance')
                    if ielts >= 6.5:
                        factors.append('Good English proficiency')
                    if work_exp > 0:
                        factors.append('Relevant work experience')
                    
                    recommendations = []
                    if gpa < 3.0:
                        recommendations.append('Consider improving academic performance')
                    if ielts < 6.5:
                        recommendations.append('Consider retaking IELTS for higher score')
                    if not work_exp:
                        recommendations.append('Gain relevant work or volunteer experience')
                    
                    return {
                        'probability': probability,
                        'confidence': confidence,
                        'factors': factors,
                        'recommendations': recommendations
                    }
            
            return FallbackPredictor()
        
        def create_fallback_verifier(self):
            class FallbackVerifier:
                def verify_document(self, document_data, doc_type):
                    # Simulate document verification
                    return {
                        'document_id': f"doc_{int(time.time())}",
                        'document_type': doc_type,
                        'verification_status': 'verified',
                        'confidence_score': 0.85,
                        'issues_found': [],
                        'recommendations': ['Document appears to be authentic'],
                        'verified_fields': {
                            'basic_info': {'verified': True, 'confidence': 0.9, 'value': 'Present'},
                            'signatures': {'verified': True, 'confidence': 0.8, 'value': 'Valid'},
                            'dates': {'verified': True, 'confidence': 0.95, 'value': 'Current'}
                        }
                    }
            
            return FallbackVerifier()
        
        def process_user_message(self, user_input, profile=None):
            return {
                'ai_response': "I'm running in basic mode. I can provide general UEL information and basic assistance. For full functionality, please ensure all system components are properly installed.",
                'sentiment': {'sentiment': 'neutral', 'polarity': 0.0},
                'search_results': []
            }
        
        def get_system_status(self):
            return {
                'system_ready': False,
                'ollama_available': False,
                'ml_ready': False,
                'data_loaded': True,
                'voice_available': False
            }
    
    return FallbackAgent()

def get_feature_status() -> Dict:
    """Get comprehensive feature status"""
    try:
        ai_agent = get_enhanced_ai_agent()
        status = ai_agent.get_system_status()
        
        return {
            'llm_integration': status.get('ollama_available', False),
            'ml_predictions': status.get('ml_ready', False),
            'course_recommendations': status.get('data_loaded', True),
            'document_verification': True,
            'voice_services': status.get('voice_available', False),
            'analytics': True,
            'export_functionality': True,
            'search_system': True,
            'comparison_tools': True
        }
        
    except Exception as e:
        logger.error(f"Error getting feature status: {e}")
        return {
            'llm_integration': False,
            'ml_predictions': False,
            'course_recommendations': True,
            'document_verification': True,
            'voice_services': False,
            'analytics': True,
            'export_functionality': True,
            'search_system': True,
            'comparison_tools': True
        }

# =============================================================================
# ENHANCED SIDEBAR WITH ALL FEATURES (RESTORED)
# =============================================================================

def render_enhanced_sidebar():
    """Render sidebar with complete feature set"""
    with st.sidebar:
        # Profile section (always at top)
        render_sidebar_profile_section()
        
        st.markdown("---")
        
        # Navigation with all features
        render_complete_sidebar_navigation()
        
        st.markdown("---")
        
        # System status with all features
        render_enhanced_system_status()
        
        st.markdown("---")
        
        # Quick actions with all features
        render_enhanced_quick_actions()

def render_complete_sidebar_navigation():
    """Render complete navigation with all features"""
    st.markdown("### 📋 Navigation")
    
    # Define all available pages
    pages = [
        ("🏠 Dashboard", "dashboard", False),
        ("💬 AI Chat", "ai_chat", False),
        ("🎯 Recommendations", "recommendations", True),
        ("🔮 Predictions", "predictions", True),
        ("🎤 Interview Prep", "interview_preparation", True),
        ("📄 Documents", "documents", False),
        ("📊 Analytics", "analytics", False),
        ("🔍 Search", "search", False),
        ("⚖️ Compare", "comparison", False),
        ("📤 Export", "export", False),
        ("🎤 Voice", "voice", False)
    ]
    
    profile_active = st.session_state.get('profile_active', False)
    
    for page_name, page_key, requires_profile in pages:
        accessible = not requires_profile or profile_active
        
        if accessible:
            if st.button(page_name, key=f"nav_{page_key}", use_container_width=True):
                st.session_state.current_page = page_key
                st.rerun()
        else:
            st.markdown(f"""
            <div style="
                padding: 0.5rem 1rem; 
                border-radius: 8px; 
                background: rgba(156, 163, 175, 0.1); 
                border: 1px solid #9ca3af; 
                color: #6b7280; 
                text-align: center; 
                margin-bottom: 0.25rem;
                opacity: 0.6;
            ">
                {page_name} 🔒
            </div>
            """, unsafe_allow_html=True)

def render_enhanced_system_status():
    """Render enhanced system status with all features"""
    st.markdown("### ⚙️ System Status")
    
    try:
        feature_status = get_feature_status()
        
        # All features status
        all_features = [
            ("🤖 AI Chat", feature_status.get('llm_integration', True)),
            ("🧠 ML Models", feature_status.get('ml_predictions', True)),
            ("🎯 Recommendations", feature_status.get('course_recommendations', True)),
            ("📄 Document AI", feature_status.get('document_verification', True)),
            ("🎤 Voice Services", feature_status.get('voice_services', False)),
            ("📊 Analytics", feature_status.get('analytics', True)),
            ("📤 Export Tools", feature_status.get('export_functionality', True)),
            ("🔍 Search System", feature_status.get('search_system', True))
        ]
        
        for feature_name, status in all_features:
            color = "#4ECDC4" if status else "#FF8A80"
            icon = "✅" if status else "❌"
            st.markdown(f"""
            <div style="display: flex; justify-content: space-between; align-items: center; padding: 0.5rem; margin: 0.3rem 0; background: rgba(255,255,255,0.1); border-radius: 8px;">
                <span style="color: rgba(255,255,255,0.9); font-size: 0.9rem;">{feature_name}</span>
                <span style="color: {color}; font-size: 1rem;">{icon}</span>
            </div>
            """, unsafe_allow_html=True)
        
    except Exception as e:
        st.error(f"Status unavailable: {e}")

def render_enhanced_quick_actions():
    """Render enhanced quick actions with all features"""
    st.markdown("### 🚀 Quick Actions")
    
    profile_active = st.session_state.get('profile_active', False)
    
    if profile_active:
        # Profile-enabled quick actions
        if st.button("🎯 Get Recommendations", use_container_width=True, key="__get_recommendation_3021"):
            st.session_state.current_page = "recommendations"
            st.rerun()
        if st.button("🎤 Interview Practice", use_container_width=True, key="__interview_practice_quick"):
            st.session_state.current_page = "interview_preparation"
            st.rerun()
        
        if st.button("🔮 Check Admission Chances", use_container_width=True, key="__check_admission_ch_3022"):
            st.session_state.current_page = "predictions"
            st.rerun()
        
        if st.button("💬 Ask AI Anything", use_container_width=True, key="__ask_ai_anything_3023"):
            st.session_state.current_page = "ai_chat"
            st.rerun()
        
        if st.button("📄 Upload Documents", use_container_width=True, key="__upload_documents_3024"):
            st.session_state.current_page = "documents"
            st.rerun()
        
        if st.button("🔍 Advanced Search", use_container_width=True, key="__advanced_search_3025"):
            st.session_state.current_page = "search"
            st.rerun()
        
        if st.button("📊 View Analytics", use_container_width=True, key="__view_analytics_3026"):
            st.session_state.current_page = "analytics"
            st.rerun()
    else:
        # General actions available to all
        if st.button("💬 Chat with AI", use_container_width=True, key="__chat_with_ai_3027"):
            st.session_state.current_page = "ai_chat"
            st.rerun()
        
        if st.button("🔍 Browse Courses", use_container_width=True, key="__browse_courses_3028"):
            st.session_state.current_page = "search"
            st.rerun()
        
        if st.button("📊 View Analytics", use_container_width=True, key="__view_analytics_3029"):
            st.session_state.current_page = "analytics"
            st.rerun()
        
        # Profile-required actions
        st.markdown("""
        <div style="background: rgba(239, 68, 68, 0.1); padding: 1rem; border-radius: 8px; border: 1px solid #ef4444; text-align: center;">
            <div style="font-size: 0.9rem; color: #ef4444; margin-bottom: 0.5rem;">🔒 Profile Required</div>
            <div style="font-size: 0.8rem; color: var(--text-secondary);">Create your profile to access personalized features</div>
        </div>
        """, unsafe_allow_html=True)

# =============================================================================
# COMPLETE PROFILE SETUP WIZARD (RESTORED WITH PASSWORD)
# =============================================================================

def render_profile_setup_wizard():
    """Render comprehensive profile setup wizard with password protection"""
    st.markdown("""
    <div class="page-header">
        <div class="header-icon">👤</div>
        <div>
            <h1 style="margin: 0; background: var(--uel-gradient); -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-weight: 800; font-size: 2.5rem;">Create Your Profile</h1>
            <p style="margin: 0; color: var(--text-secondary); font-size: 1.2rem; font-weight: 500;">Let's personalize your UEL AI experience</p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Check if editing existing profile
    if st.session_state.get('show_profile_editor', False) and st.session_state.current_profile:
        render_comprehensive_profile_editor()
        return
    
    # Progress indicator
    total_steps = 5  # Added password step
    current_step = st.session_state.get('profile_setup_step', 1)
    
    st.markdown("### 📋 Setup Progress")
    progress_col1, progress_col2, progress_col3, progress_col4, progress_col5 = st.columns(5)
    
    steps = [
        ("Basic Info", 1),
        ("Academic Background", 2), 
        ("Interests & Goals", 3),
        ("Security Setup", 4),
        ("Review & Complete", 5)
    ]
    
    for i, ((step_name, step_num), col) in enumerate(zip(steps, [progress_col1, progress_col2, progress_col3, progress_col4, progress_col5])):
        with col:
            if step_num < current_step:
                status_class = "completed"
                icon = "✅"
            elif step_num == current_step:
                status_class = "current"
                icon = "🔄"
            else:
                status_class = "pending"
                icon = "⏳"
            
            st.markdown(f"""
            <div class="profile-progress-step {status_class}">
                <div style="margin-right: 0.5rem; font-size: 1.2rem;">{icon}</div>
                <div>
                    <div style="font-weight: bold; font-size: 0.9rem;">{step_name}</div>
                    <div style="font-size: 0.8rem; opacity: 0.7;">Step {step_num}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    # Progress bar
    progress_percentage = ((current_step - 1) / (total_steps - 1)) * 100
    st.progress(progress_percentage / 100)
    
    st.markdown("---")
    
    # Render current step
    if current_step == 1:
        render_basic_info_step()
    elif current_step == 2:
        render_academic_background_step()
    elif current_step == 3:
        render_interests_goals_step()
    elif current_step == 4:
        render_security_setup_step()
    elif current_step == 5:
        render_review_complete_step()

def render_basic_info_step():
    """Render basic information step"""
    st.markdown("### 👤 Basic Information")
    st.markdown("*Tell us a bit about yourself*")
    
    with st.form("basic_info_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            first_name = st.text_input(
                "First Name *", 
                value=st.session_state.get('temp_first_name', ''),
                placeholder="John",
                help="Your first name"
            )
            last_name = st.text_input(
                "Last Name *", 
                value=st.session_state.get('temp_last_name', ''),
                placeholder="Smith", 
                help="Your last name"
            )
            email = st.text_input(
                "Email Address", 
                value=st.session_state.get('temp_email', ''),
                placeholder="john.smith@email.com",
                help="Your email address (optional but recommended)"
            )
        
        with col2:
            country = st.selectbox(
                "Country of Residence *",
                ["", "United Kingdom", "United States", "India", "China", "Nigeria", 
                 "Pakistan", "Canada", "Germany", "France", "Australia", "Brazil", "Other"],
                index=0 if not st.session_state.get('temp_country') else 
                ["", "United Kingdom", "United States", "India", "China", "Nigeria", 
                 "Pakistan", "Canada", "Germany", "France", "Australia", "Brazil", "Other"].index(st.session_state.get('temp_country', '')),
                help="Your country of residence"
            )
            nationality = st.text_input(
                "Nationality", 
                value=st.session_state.get('temp_nationality', ''),
                placeholder="British, American, etc.",
                help="Your nationality"
            )
            date_of_birth = st.date_input(
                "Date of Birth (optional)",
                value=st.session_state.get('temp_date_of_birth'),
                min_value=datetime(1950, 1, 1),
                max_value=datetime.now(),
                help="Your date of birth"
            )
        
        # Navigation
        col_back, col_next = st.columns([1, 3])
        
        with col_next:
            next_step = st.form_submit_button("Continue to Academic Background →", type="primary", use_container_width=True)
        
        if next_step and first_name and last_name and country:
            # Save temporary data
            st.session_state.temp_first_name = first_name
            st.session_state.temp_last_name = last_name
            st.session_state.temp_email = email
            st.session_state.temp_country = country
            st.session_state.temp_nationality = nationality
            st.session_state.temp_date_of_birth = date_of_birth
            
            # Move to next step
            st.session_state.profile_setup_step = 2
            st.rerun()
        
        elif next_step:
            st.error("❌ Please fill in all required fields marked with *")

def render_academic_background_step():
    """Render academic background step"""
    st.markdown("### 🎓 Academic Background")
    st.markdown("*Help us understand your educational journey*")
    
    with st.form("academic_background_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            academic_level = st.selectbox(
                "Current Academic Level *",
                ["", "High School", "High School Graduate", "Undergraduate Student", 
                 "Bachelor's Graduate", "Postgraduate Student", "Master's Graduate", 
                 "PhD Student", "PhD Graduate", "Working Professional"],
                index=0 if not st.session_state.get('temp_academic_level') else
                ["", "High School", "High School Graduate", "Undergraduate Student", 
                 "Bachelor's Graduate", "Postgraduate Student", "Master's Graduate", 
                 "PhD Student", "PhD Graduate", "Working Professional"].index(st.session_state.get('temp_academic_level', '')),
                help="Your current education level"
            )
            
            current_institution = st.text_input(
                "Current/Previous Institution",
                value=st.session_state.get('temp_current_institution', ''),
                placeholder="e.g., Oxford University, Local High School",
                help="Name of your current or most recent educational institution"
            )
            
            current_major = st.text_input(
                "Current/Previous Major/Subject",
                value=st.session_state.get('temp_current_major', ''),
                placeholder="e.g., Computer Science, Business Studies",
                help="Your field of study"
            )
        
        with col2:
            gpa = st.number_input(
                "GPA/Grade Average (0.0-4.0)",
                min_value=0.0,
                max_value=4.0,
                value=st.session_state.get('temp_gpa', 3.0),
                step=0.1,
                help="Your cumulative GPA on a 4.0 scale (or equivalent)"
            )
            
            ielts_score = st.number_input(
                "IELTS Score (if taken)",
                min_value=0.0,
                max_value=9.0,
                value=st.session_state.get('temp_ielts_score', 6.5),
                step=0.5,
                help="Your IELTS English proficiency score"
            )
            
            graduation_year = st.number_input(
                "Expected/Actual Graduation Year",
                min_value=2020,
                max_value=2030,
                value=st.session_state.get('temp_graduation_year', datetime.now().year),
                help="When you graduated or expect to graduate"
            )
        
        # Navigation
        col_back, col_next = st.columns([1, 3])
        
        with col_back:
            back_step = st.form_submit_button("← Back", use_container_width=True)
        
        with col_next:
            next_step = st.form_submit_button("Continue to Interests & Goals →", type="primary", use_container_width=True)
        
        if back_step:
            st.session_state.profile_setup_step = 1
            st.rerun()
        
        if next_step and academic_level:
            # Save temporary data
            st.session_state.temp_academic_level = academic_level
            st.session_state.temp_current_institution = current_institution
            st.session_state.temp_current_major = current_major
            st.session_state.temp_gpa = gpa
            st.session_state.temp_ielts_score = ielts_score
            st.session_state.temp_graduation_year = graduation_year
            
            # Move to next step
            st.session_state.profile_setup_step = 3
            st.rerun()
        
        elif next_step:
            st.error("❌ Please fill in all required fields marked with *")

def render_interests_goals_step():
    """Render interests and goals step"""
    st.markdown("### 🎯 Interests & Career Goals")
    st.markdown("*Help us understand what you're passionate about*")
    
    with st.form("interests_goals_form"):
        # Primary field of interest
        field_of_interest = st.text_input(
            "Primary Field of Interest *",
            value=st.session_state.get('temp_field_of_interest', ''),
            placeholder="e.g., Computer Science, Business Management, Psychology",
            help="What subject area interests you most for your studies?"
        )
        
        # Multiple interests
        st.markdown("**Additional Areas of Interest**")
        interests = st.multiselect(
            "Select all that apply:",
            [
                "Artificial Intelligence", "Data Science", "Cybersecurity", "Web Development", "Software Engineering",
                "Business Management", "Finance", "Marketing", "Human Resources", "Entrepreneurship",
                "Psychology", "Counseling", "Social Work", "Education", "Research",
                "Engineering", "Architecture", "Design", "Arts", "Creative Writing",
                "Medicine", "Nursing", "Healthcare", "Biology", "Chemistry",
                "Law", "International Relations", "Politics", "History", "Philosophy",
                "Media Studies", "Communications", "Journalism", "Film Production"
            ],
            default=st.session_state.get('temp_interests', []),
            help="Select any additional areas that interest you"
        )
        
        col1, col2 = st.columns(2)
        
        with col1:
            career_goals = st.text_area(
                "Career Goals & Aspirations",
                value=st.session_state.get('temp_career_goals', ''),
                placeholder="Describe your career aspirations and what you hope to achieve...",
                height=100,
                help="Tell us about your professional goals"
            )
            
            target_industry = st.selectbox(
                "Target Industry",
                ["", "Technology", "Finance", "Healthcare", "Education", "Government",
                 "Non-Profit", "Media & Entertainment", "Sports", "Retail", "Manufacturing",
                 "Consulting", "Startups", "Research & Development", "Other"],
                index=0 if not st.session_state.get('temp_target_industry') else
                ["", "Technology", "Finance", "Healthcare", "Education", "Government",
                 "Non-Profit", "Media & Entertainment", "Sports", "Retail", "Manufacturing",
                 "Consulting", "Startups", "Research & Development", "Other"].index(st.session_state.get('temp_target_industry', '')),
                help="Which industry would you like to work in?"
            )
        
        with col2:
            work_experience_years = st.number_input(
                "Work Experience (years)",
                min_value=0,
                max_value=30,
                value=st.session_state.get('temp_work_experience_years', 0),
                help="Years of relevant work experience"
            )
            
            preferred_study_mode = st.selectbox(
                "Preferred Study Mode",
                ["Full-time", "Part-time", "Online", "Flexible/Hybrid"],
                index=0 if not st.session_state.get('temp_preferred_study_mode') else
                ["Full-time", "Part-time", "Online", "Flexible/Hybrid"].index(st.session_state.get('temp_preferred_study_mode', 'Full-time')),
                help="How would you prefer to study?"
            )
            
            budget_range = st.selectbox(
                "Budget Range (annual)",
                ["Not sure", "Under £10,000", "£10,000 - £15,000", "£15,000 - £20,000", "Over £20,000"],
                index=0 if not st.session_state.get('temp_budget_range') else
                ["Not sure", "Under £10,000", "£10,000 - £15,000", "£15,000 - £20,000", "Over £20,000"].index(st.session_state.get('temp_budget_range', 'Not sure')),
                help="What's your budget for tuition fees?"
            )
        
        # Navigation
        col_back, col_next = st.columns([1, 3])
        
        with col_back:
            back_step = st.form_submit_button("← Back", use_container_width=True)
        
        with col_next:
            next_step = st.form_submit_button("Continue to Security Setup →", type="primary", use_container_width=True)
        
        if back_step:
            st.session_state.profile_setup_step = 2
            st.rerun()
        
        if next_step and field_of_interest:
            # Save temporary data
            st.session_state.temp_field_of_interest = field_of_interest
            st.session_state.temp_interests = interests
            st.session_state.temp_career_goals = career_goals
            st.session_state.temp_target_industry = target_industry
            st.session_state.temp_work_experience_years = work_experience_years
            st.session_state.temp_preferred_study_mode = preferred_study_mode
            st.session_state.temp_budget_range = budget_range
            
            # Move to next step
            st.session_state.profile_setup_step = 4
            st.rerun()
        
        elif next_step:
            st.error("❌ Please specify your primary field of interest")

def render_security_setup_step():
    """Render security setup step with password"""
    st.markdown("### 🔐 Security Setup")
    st.markdown("*Secure your profile with a password*")
    
    st.info("💡 Your profile will be automatically saved and you can log back in using your password.")
    
    with st.form("security_setup_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            password = st.text_input(
                "Create Password *",
                type="password",
                help="Choose a secure password (minimum 6 characters)",
                placeholder="Enter your password"
            )
            
            confirm_password = st.text_input(
                "Confirm Password *",
                type="password",
                help="Re-enter your password to confirm",
                placeholder="Confirm your password"
            )
        
        with col2:
            st.markdown("#### 🛡️ Password Requirements")
            st.markdown("""
            - Minimum 6 characters
            - Keep it secure and memorable
            - Used to protect your profile data
            - Required for future logins
            """)
            
            save_preference = st.radio(
                "Data Persistence",
                ["Save my profile for future sessions", "Use profile only for this session"],
                index=0,
                help="Choose how long to keep your profile"
            )
        
        # Navigation
        col_back, col_next = st.columns([1, 3])
        
        with col_back:
            back_step = st.form_submit_button("← Back", use_container_width=True)
        
        with col_next:
            next_step = st.form_submit_button("Continue to Review →", type="primary", use_container_width=True)
        
        if back_step:
            st.session_state.profile_setup_step = 3
            st.rerun()
        
        if next_step:
            if not password:
                st.error("❌ Please create a password")
            elif len(password) < 6:
                st.error("❌ Password must be at least 6 characters long")
            elif password != confirm_password:
                st.error("❌ Passwords do not match")
            else:
                # Save security settings
                st.session_state.temp_password = password
                st.session_state.temp_save_preference = save_preference
                
                # Move to next step
                st.session_state.profile_setup_step = 5
                st.rerun()

def render_review_complete_step():
    """Render review and complete step"""
    st.markdown("### ✅ Review & Complete Your Profile")
    st.markdown("*Please review your information before completing setup*")
    
    # Collect all temporary data
    profile_data = {
        'id': f"profile_{int(time.time())}",
        'first_name': st.session_state.get('temp_first_name', ''),
        'last_name': st.session_state.get('temp_last_name', ''),
        'email': st.session_state.get('temp_email', ''),
        'country': st.session_state.get('temp_country', ''),
        'nationality': st.session_state.get('temp_nationality', ''),
        'date_of_birth': st.session_state.get('temp_date_of_birth', ''),
        'academic_level': st.session_state.get('temp_academic_level', ''),
        'current_institution': st.session_state.get('temp_current_institution', ''),
        'current_major': st.session_state.get('temp_current_major', ''),
        'gpa': st.session_state.get('temp_gpa', 0.0),
        'ielts_score': st.session_state.get('temp_ielts_score', 0.0),
        'graduation_year': st.session_state.get('temp_graduation_year', 0),
        'field_of_interest': st.session_state.get('temp_field_of_interest', ''),
        'interests': st.session_state.get('temp_interests', []),
        'career_goals': st.session_state.get('temp_career_goals', ''),
        'target_industry': st.session_state.get('temp_target_industry', ''),
        'work_experience_years': st.session_state.get('temp_work_experience_years', 0),
        'preferred_study_mode': st.session_state.get('temp_preferred_study_mode', ''),
        'budget_range': st.session_state.get('temp_budget_range', ''),
        'profile_completion': 0.0,  # Initialize completion
        'created_date': datetime.now().isoformat(),
        'interaction_count': 0
    }
    
    # Calculate profile completion
    required_fields = ['first_name', 'last_name', 'country', 'academic_level', 'field_of_interest']
    completed_fields = sum(1 for field in required_fields if profile_data[field])
    optional_fields = ['email', 'career_goals', 'interests', 'current_institution']
    completed_optional = sum(1 for field in optional_fields if profile_data.get(field))
    
    profile_data['profile_completion'] = ((completed_fields / len(required_fields)) * 70) + ((completed_optional / len(optional_fields)) * 30)

    # Display review in columns
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 👤 Basic Information")
        st.markdown(f"""
        <div class="enhanced-card" style="border-left: 4px solid var(--uel-primary);">
            <p><strong>Name:</strong> {profile_data['first_name']} {profile_data['last_name']}</p>
            <p><strong>Email:</strong> {profile_data['email'] or 'Not provided'}</p>
            <p><strong>Country:</strong> {profile_data['country']}</p>
            <p><strong>Nationality:</strong> {profile_data['nationality'] or 'Not specified'}</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("#### 🎯 Interests & Goals")
        interests_text = ', '.join(profile_data['interests']) if profile_data['interests'] else 'None specified'
        st.markdown(f"""
        <div class="enhanced-card" style="border-left: 4px solid var(--uel-secondary);">
            <p><strong>Primary Interest:</strong> {profile_data['field_of_interest']}</p>
            <p><strong>Additional Interests:</strong> {interests_text}</p>
            <p><strong>Career Goals:</strong> {profile_data['career_goals'][:100]}{'...' if len(profile_data['career_goals']) > 100 else ''}</p>
            <p><strong>Target Industry:</strong> {profile_data['target_industry'] or 'Not specified'}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("#### 🎓 Academic Background")
        st.markdown(f"""
        <div class="enhanced-card" style="border-left: 4px solid var(--uel-success);">
            <p><strong>Academic Level:</strong> {profile_data['academic_level']}</p>
            <p><strong>Institution:</strong> {profile_data['current_institution'] or 'Not specified'}</p>
            <p><strong>Major/Subject:</strong> {profile_data['current_major'] or 'Not specified'}</p>
            <p><strong>GPA:</strong> {profile_data['gpa']}/4.0</p>
            <p><strong>IELTS Score:</strong> {profile_data['ielts_score']}/9.0</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("#### 💼 Preferences")
        st.markdown(f"""
        <div class="enhanced-card" style="border-left: 4px solid var(--uel-warning);">
            <p><strong>Work Experience:</strong> {profile_data['work_experience_years']} years</p>
            <p><strong>Study Mode:</strong> {profile_data['preferred_study_mode']}</p>
            <p><strong>Budget Range:</strong> {profile_data['budget_range']}</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Profile completion display
    completion_percentage = profile_data['profile_completion']
    st.markdown(f"""
    <div class="enhanced-card" style="background: linear-gradient(135deg, rgba(16, 185, 129, 0.1), rgba(16, 185, 129, 0.05)); text-align: center;">
        <h3 style="margin-top: 0; color: var(--uel-success);">📊 Profile Completion: {completion_percentage:.0f}%</h3>
        <div style="background: #e2e8f0; border-radius: 10px; height: 20px; margin: 1rem 0;">
            <div style="background: var(--uel-success); height: 100%; border-radius: 10px; width: {completion_percentage}%; transition: width 0.3s ease;"></div>
        </div>
        <p style="margin-bottom: 0;">Your profile is ready! You can always edit it later.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Terms and conditions
    st.markdown("---")
    accept_terms = st.checkbox(
        "I agree to the terms and conditions and privacy policy",
        help="By checking this box, you agree to our terms of service and privacy policy"
    )
    
    # Navigation
    col_back, col_complete = st.columns([1, 3])
    
    with col_back:
        if st.button("← Back to Edit", use_container_width=True, key="__back_to_edit_3030"):
            st.session_state.profile_setup_step = 4
            st.rerun()
    
    with col_complete:
        if st.button("🎉 Create My Profile", type="primary", use_container_width=True, disabled=not accept_terms, key="__create_my_profile_3031"):
            if accept_terms:
                try:
                    # Validate required fields
                    required_fields = ['first_name', 'last_name', 'country', 'academic_level', 'field_of_interest']
                    missing_fields = [field for field in required_fields if not profile_data[field]]
                    if missing_fields:
                        st.error(f"❌ Please complete required fields: {', '.join(missing_fields)}")
                        return
                    
                    password = st.session_state.get('temp_password', '')
                    save_preference = st.session_state.get('temp_save_preference', '')
                    
                    # Save profile with password protection
                    if save_preference == "Save my profile for future sessions":
                        if save_profile_to_storage(profile_data, password):
                            st.success("✅ Profile saved with password protection!")
                        else:
                            st.warning("⚠️ Profile created but saving failed. Profile will be active for this session only.")
                    else:
                        st.info("📝 Profile created for this session only.")
                    
                    # Update session state
                    st.session_state.current_profile = profile_data
                    st.session_state.profile_active = True
                    st.session_state.current_page = 'dashboard'
                    st.session_state.onboarding_completed = True
                    
                    # Clear temporary data
                    temp_keys = [key for key in st.session_state.keys() if key.startswith('temp_')]
                    for key in temp_keys:
                        del st.session_state[key]
                    
                    st.success("🎉 Profile created successfully! Welcome to UEL AI Assistant!")
                    st.balloons()
                    time.sleep(2)
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"❌ Error creating profile: {str(e)}")
                    logger.error(f"Profile creation error: {str(e)}", exc_info=True)
            else:
                st.error("❌ Please accept the terms and conditions to continue")

# =============================================================================
# ENHANCED CHAT INTERFACE (RESTORED)
# =============================================================================

def render_enhanced_chat_page():
    """Render enhanced chat page with complete functionality"""
    # Allow flexible access - enhanced with profile
    profile_available = check_profile_access("AI Chat", flexible=True)
    
    st.markdown("""
    <div class="page-header">
        <div class="header-icon">💬</div>
        <div>
            <h1 style="margin: 0; background: var(--uel-gradient); -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-weight: 800; font-size: 2.5rem;">AI Chat Assistant</h1>
            <p style="margin: 0; color: var(--text-secondary); font-size: 1.2rem; font-weight: 500;">Your intelligent guide to UEL - personalized with your profile</p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    profile = st.session_state.current_profile
    
    # Enhanced personalized welcome
    if profile:
        st.markdown(f"""
        <div class="enhanced-card" style="background: var(--uel-gradient); color: white; text-align: center;">
            <h3 style="margin-top: 0;">👋 Hello {profile.get('first_name', 'Student')}!</h3>
            <p style="margin-bottom: 0;">I'm your complete AI assistant, ready to help with your {profile.get('field_of_interest', 'academic')} journey at UEL. I know your background and can provide personalized guidance!</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="enhanced-card" style="background: linear-gradient(135deg, rgba(59, 130, 246, 0.1), rgba(59, 130, 246, 0.05)); border-left: 4px solid #3b82f6;">
            <h3 style="margin-top: 0; color: #3b82f6;">🤖 AI Assistant Ready</h3>
            <p style="margin-bottom: 0;">I can help with general UEL information. Create a profile for personalized advice!</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Initialize chat if needed
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    # Enhanced personalized quick actions
    render_chat_quick_actions(profile)
    
    # Chat interface
    st.markdown("---")
    st.markdown("### 💬 Conversation")
    
    # Enhanced chat controls
    chat_control_col1, chat_control_col2, chat_control_col3 = st.columns([2, 1, 1])
    
    with chat_control_col2:
        if st.button("🎤 Voice Input", use_container_width=True, key="__voice_input_3032"):
            handle_voice_input_advanced()
    
    with chat_control_col3:
        if st.button("🗑️ Clear Chat", use_container_width=True, key="___clear_chat_3033"):
            st.session_state.messages = []
            st.rerun()
    
    # Display messages with enhanced formatting
    for message in st.session_state.messages:
        render_enhanced_chat_message(message)
    
    # Enhanced input area
    render_chat_input_interface()

def render_chat_quick_actions(profile: Optional[Dict]):
    """Render chat quick actions based on profile"""
    st.markdown("### 🚀 Quick Questions")
    
    if profile:
        # Personalized suggestions
        field = profile.get('field_of_interest', '').lower()
        level = profile.get('academic_level', '').lower()
        
        suggestions = []
        if 'computer' in field or 'technology' in field or 'data' in field:
            suggestions.extend([
                f"What computer science programs at UEL match my {profile.get('gpa', 3.0)} GPA?",
                f"How can I improve my chances for tech programs with my {profile.get('ielts_score', 6.5)} IELTS score?",
                "What programming skills should I develop before starting?",
                "Tell me about internship opportunities in tech at UEL"
            ])
        elif 'business' in field or 'management' in field:
            suggestions.extend([
                f"What business programs fit my {level} background?",
                "How does UEL's business school compare to others?",
                "What are the career services for business students?",
                "Tell me about the MBA program requirements"
            ])
        elif 'engineering' in field:
            suggestions.extend([
                "What engineering specializations does UEL offer?",
                "How do my grades compare to typical engineering admits?",
                "What are the lab facilities like for engineering students?",
                "Tell me about the professional accreditation"
            ])
        else:
            suggestions.extend([
                f"What {field} programs are available at UEL?",
                f"How competitive are admissions for {field} programs?",
                f"What career paths are available in {field}?",
                f"What support is available for {level} students?"
            ])
        
        # Add contextual suggestions
        if profile.get('gpa', 0) < 3.0:
            suggestions.append("How can I strengthen my application with a lower GPA?")
        if profile.get('ielts_score', 0) < 6.5:
            suggestions.append("What English proficiency support is available?")
        if profile.get('work_experience_years', 0) > 0:
            suggestions.append("How does my work experience help my application?")
    else:
        # General suggestions
        suggestions = [
            "What undergraduate programs does UEL offer?",
            "Tell me about UEL's admissions requirements",  
            "What are the tuition fees for international students?",
            "How do I apply to UEL?",
            "What is campus life like at UEL?",
            "What support services are available?",
            "Tell me about UEL's facilities",
            "How can I contact the admissions team?"
        ]
    
    # Display enhanced suggestion buttons
    suggestion_cols = st.columns(2)
    for i, suggestion in enumerate(suggestions[:8]):  # Show up to 8 suggestions
        with suggestion_cols[i % 2]:
            if st.button(suggestion, key=f"suggestion_{i}", use_container_width=True):
                handle_chat_message(suggestion)

def render_enhanced_chat_message(message):
    """Render enhanced chat message with better styling"""
    role = message['role']
    content = message['content']
    timestamp = message.get('timestamp', datetime.now())
    
    if role == 'user':
        profile = st.session_state.current_profile
        user_name = profile.get('first_name', 'You') if profile else 'You'
        input_method = message.get('input_method', 'text')
        method_icon = "🎤" if input_method == 'voice' else "💬"
        
        st.markdown(f"""
        <div style="text-align: right; margin: 1rem 0;">
            <div style="display: inline-block; background: var(--uel-gradient); color: white; padding: 1rem 1.5rem; border-radius: 20px 20px 5px 20px; max-width: 70%;">
                <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 0.5rem;">
                    <span style="font-weight: bold;">{user_name}</span>
                    <span style="font-size: 0.8rem; opacity: 0.8;">{method_icon}</span>
                </div>
                <div>{content}</div>
                <div style="font-size: 0.8rem; opacity: 0.8; margin-top: 0.5rem;">{timestamp.strftime('%H:%M')}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    else:  # assistant
        profile_integrated = message.get('profile_integrated', False)
        confidence = message.get('confidence', 0.8)
        
        profile_indicator = "🎯" if profile_integrated else "🤖"
        personalized_text = "Personalized Response" if profile_integrated else "General Response"
        
        st.markdown(f"""
        <div style="margin: 1rem 0;">
            <div style="display: inline-block; background: white; border: 2px solid var(--uel-primary); padding: 1rem 1.5rem; border-radius: 20px 20px 20px 5px; max-width: 70%;">
                <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 0.5rem;">
                    <div style="display: flex; align-items: center;">
                        <span style="margin-right: 0.5rem;">{profile_indicator}</span>
                        <span style="font-weight: bold; color: var(--uel-primary);">UEL AI Assistant</span>
                    </div>
                    <span style="font-size: 0.8rem; color: var(--uel-success); background: rgba(16, 185, 129, 0.1); padding: 0.2rem 0.5rem; border-radius: 10px;">{personalized_text}</span>
                </div>
                <div style="color: var(--text-primary);">{content}</div>
                <div style="font-size: 0.8rem; color: var(--text-light); margin-top: 0.5rem; display: flex; justify-content: space-between;">
                    <span>{timestamp.strftime('%H:%M')}</span>
                    <span>Confidence: {confidence:.0%}</span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

def render_chat_input_interface():
    """Render enhanced chat input interface"""
    with st.form("chat_form", clear_on_submit=True):
        profile = st.session_state.current_profile
        placeholder_text = f"Ask me anything about UEL or your {profile.get('field_of_interest', 'academic')} interests..." if profile else "Ask me anything about UEL..."
        
        user_input = st.text_area(
            "Your message:",
            placeholder=placeholder_text,
            height=100,
            help="Type your message or use voice input above"
        )
        
        col1, col2 = st.columns([4, 1])
        with col1:
            send_button = st.form_submit_button("Send 📤", type="primary", use_container_width=True)
        with col2:
            if st.form_submit_button("🔊 Speak", use_container_width=True):
                if st.session_state.messages and st.session_state.messages[-1].get('role') == 'assistant':
                    last_response = st.session_state.messages[-1].get('content', '')
                    if last_response:
                        test_text_to_speech_with_message(last_response)
        
        if send_button and user_input:
            handle_chat_message(user_input)

def handle_chat_message(user_input: str):
    """Handle chat message with complete profile integration"""
    try:
        # Add user message
        st.session_state.messages.append({
            'role': 'user',
            'content': user_input,
            'timestamp': datetime.now(),
            'input_method': 'text'
        })
        
        # Get AI response with full context
        response_data = get_ai_response(user_input)
        
        # Add assistant response
        st.session_state.messages.append({
            'role': 'assistant',
            'content': response_data['response'],
            'timestamp': datetime.now(),
            'profile_integrated': response_data.get('profile_integrated', False),
            'confidence': response_data.get('confidence', 0.8),
            'requires_profile': response_data.get('requires_profile', False)
        })
        
        # Update interaction count
        if st.session_state.current_profile:
            profile = st.session_state.current_profile
            profile['interaction_count'] = profile.get('interaction_count', 0) + 1
            st.session_state.current_profile = profile
        
        st.rerun()
        
    except Exception as e:
        st.error(f"Error processing message: {e}")
        logger.error(f"Chat message error: {e}")

def test_text_to_speech_with_message(message: str):
    """Test text-to-speech with specific message"""
    try:
        ai_agent = get_enhanced_ai_agent()
        
        with st.spinner("🔊 Speaking response..."):
            success = ai_agent.voice_service.text_to_speech(message)
        
        if success:
            st.success("🔊 Response spoken successfully!")
        else:
            st.error("❌ Text-to-speech failed")
            
    except Exception as e:
        st.error(f"❌ Speech error: {e}")

# =============================================================================
# ENHANCED RECOMMENDATIONS PAGE (RESTORED)
# =============================================================================

def render_enhanced_recommendations_page():
    """Render enhanced recommendations page with complete functionality"""
    # Require profile for full recommendations
    if not check_profile_access("Course Recommendations", flexible=False):
        return
    
    st.markdown("""
    <div class="page-header">
        <div class="header-icon">🎯</div>
        <div>
            <h1 style="margin: 0; background: var(--uel-gradient); -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-weight: 800; font-size: 2.5rem;">Your Complete Course Recommendations</h1>
            <p style="margin: 0; color: var(--text-secondary); font-size: 1.2rem; font-weight: 500;">AI-matched courses with advanced filtering and comparison</p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    profile = st.session_state.current_profile
    
    # Enhanced profile summary for recommendations
    st.markdown(f"""
    <div class="enhanced-card" style="background: linear-gradient(135deg, rgba(100, 181, 246, 0.1), rgba(100, 181, 246, 0.05)); border-left: 6px solid var(--uel-info);">
        <h3 style="margin-top: 0; color: var(--uel-info);">👤 Recommendations Based On Your Complete Profile</h3>
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 1rem; margin-top: 1rem;">
            <div><strong>Primary Interest:</strong> {profile.get('field_of_interest', 'Not specified')}</div>
            <div><strong>Academic Level:</strong> {profile.get('academic_level', 'Not specified')}</div>
            <div><strong>Target Industry:</strong> {profile.get('target_industry', 'Open to all')}</div>
            <div><strong>Study Mode:</strong> {profile.get('preferred_study_mode', 'Flexible')}</div>
            <div><strong>Budget Range:</strong> {profile.get('budget_range', 'Not specified')}</div>
            <div><strong>Profile Strength:</strong> {'Strong' if profile.get('gpa', 0) >= 3.5 else 'Good' if profile.get('gpa', 0) >= 3.0 else 'Developing'}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Advanced recommendation controls
    with st.expander("🔧 Advanced Recommendation Settings", expanded=False):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            match_threshold = st.slider("Minimum Match Score", 0.0, 1.0, 0.3, 0.1, help="Filter courses by minimum match score")
            include_reach = st.checkbox("Include 'reach' schools", value=True, help="Include courses with higher requirements")
        
        with col2:
            sort_by = st.selectbox("Sort by", ["Match Score", "Fees (Low to High)", "Fees (High to Low)", "Popularity"], help="How to order recommendations")
            max_results = st.number_input("Max Results", 5, 20, 10, help="Maximum number of recommendations to show")
        
        with col3:
            level_override = st.selectbox("Override Level Filter", ["Auto", "undergraduate", "postgraduate", "masters"], help="Override automatic level filtering")
            show_analytics = st.checkbox("Show match analytics", value=True, help="Display detailed matching information")
    
    # Get enhanced recommendations
    with st.spinner("🤖 Generating your complete personalized course recommendations..."):
        recommendations = get_course_recommendations()
        
        # Apply advanced filtering
        if match_threshold > 0:
            recommendations = [r for r in recommendations if r.get('score', 0) >= match_threshold]
        
        # Apply sorting
        if sort_by == "Match Score":
            recommendations.sort(key=lambda x: x.get('score', 0), reverse=True)
        elif sort_by == "Fees (Low to High)":
            recommendations.sort(key=lambda x: x.get('fees_international', 999999))
        elif sort_by == "Fees (High to Low)":
            recommendations.sort(key=lambda x: x.get('fees_international', 0), reverse=True)
        elif sort_by == "Popularity":
            recommendations.sort(key=lambda x: x.get('trending_score', 0), reverse=True)
        
        # Limit results
        recommendations = recommendations[:max_results]
    
    if recommendations:
        st.success(f"✅ Found {len(recommendations)} personalized matches for you!")
        
        # Recommendation overview analytics
        if show_analytics:
            render_recommendation_analytics(recommendations)
        
        # Display enhanced recommendations
        for i, rec in enumerate(recommendations):
            render_enhanced_recommendation_card(rec, i)
    else:
        st.warning("⚠️ No recommendations found matching your criteria. Try adjusting the filters above.")

def render_recommendation_analytics(recommendations: List[Dict]):
    """Render recommendation analytics overview"""
    st.markdown("### 📊 Recommendation Analytics")
    
    # Calculate analytics
    scores = [r.get('score', 0) for r in recommendations]
    avg_score = np.mean(scores) if scores else 0
    
    fees = [r.get('fees_international', 0) for r in recommendations if isinstance(r.get('fees_international'), (int, float))]
    avg_fee = np.mean(fees) if fees else 0
    
    levels = [r.get('level', 'Unknown') for r in recommendations]
    level_distribution = pd.Series(levels).value_counts()
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Average Match Score", f"{avg_score:.1%}")
    
    with col2:
        st.metric("Average Fee", f"£{avg_fee:,.0f}")
    
    with col3:
        most_common_level = level_distribution.index[0] if not level_distribution.empty else 'N/A'
        st.metric("Most Common Level", most_common_level)
    
    with col4:
        high_match_count = sum(1 for r in recommendations if r.get('score', 0) >= 0.8)
        st.metric("Excellent Matches", high_match_count)

def render_enhanced_recommendation_card(recommendation: Dict, index: int):
    """Render enhanced recommendation card with complete functionality"""
    with st.expander(
        f"{recommendation.get('match_quality', '✅ Match')} - {recommendation.get('course_name', 'Unknown Course')} ({recommendation.get('level', 'Unknown')})",
        expanded=(index < 3)  # Show first 3 expanded
    ):
        # Main course information
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown(f"""
            <h4 style="margin: 0 0 1rem 0; color: var(--uel-primary);">{recommendation.get('course_name', 'Unknown Course')}</h4>
            <p><strong>Department:</strong> {recommendation.get('department', 'Not specified')}</p>
            <p><strong>Duration:</strong> {recommendation.get('duration', 'Not specified')}</p>
            <p><strong>Study Mode:</strong> Full-time, Part-time options available</p>
            <p><strong>Description:</strong> {recommendation.get('description', 'No description available')}</p>
            """, unsafe_allow_html=True)
            
            # Enhanced match reasons
            if recommendation.get('reasons'):
                st.markdown("**🎯 Why This Course is Perfect for You:**")
                for reason in recommendation.get('reasons', [])[:4]:  # Show up to 4 reasons
                    st.markdown(f"• {reason}")
        
        with col2:
            # Enhanced score display
            score = recommendation.get('score', 0)
            st.markdown(f"""
            <div style="text-align: center; background: var(--uel-primary); color: white; padding: 1rem; border-radius: 16px;">
                <h2 style="margin: 0; font-size: 2rem;">{score:.0%}</h2>
                <p style="margin: 0;">Match Score</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Additional metrics
            st.metric("🏆 Popularity", f"{recommendation.get('trending_score', 5)}/10")
            st.metric("💰 International Fee", recommendation.get('fees', 'Contact for details'))
        
        # Requirements and compatibility
        st.markdown("#### 📋 Requirements & Your Compatibility")
        
        req_col1, req_col2, req_col3 = st.columns(3)
        
        with req_col1:
            min_gpa = recommendation.get('min_gpa', 0)
            your_gpa = st.session_state.current_profile.get('gpa', 0)
            gpa_status = "✅ Meets" if your_gpa >= min_gpa else "⚠️ Below" if your_gpa >= min_gpa - 0.3 else "❌ Does not meet"
            st.markdown(f"**Min GPA:** {min_gpa} | **Your GPA:** {your_gpa} | {gpa_status}")
        
        with req_col2:
            min_ielts = recommendation.get('min_ielts', 0)
            your_ielts = st.session_state.current_profile.get('ielts_score', 0)
            ielts_status = "✅ Meets" if your_ielts >= min_ielts else "⚠️ Close" if your_ielts >= min_ielts - 0.5 else "❌ Does not meet"
            st.markdown(f"**Min IELTS:** {min_ielts} | **Your IELTS:** {your_ielts} | {ielts_status}")
        
        with req_col3:
            career_fit = "🎯 Excellent" if recommendation.get('score', 0) >= 0.8 else "👍 Good" if recommendation.get('score', 0) >= 0.6 else "✅ Suitable"
            st.markdown(f"**Career Alignment:** {career_fit}")
        
        # Enhanced action buttons
        action_col1, action_col2, action_col3, action_col4, action_col5 = st.columns(5)
        
        with action_col1:
            if st.button(f"💬 Ask AI", key=f"ask_ai_{index}", use_container_width=True):
                switch_to_chat_with_course_context(recommendation.get('course_name'))
        
        with action_col2:
            if st.button(f"🔮 Predict Chances", key=f"predict_{index}", use_container_width=True):
                switch_to_prediction_with_course(recommendation.get('course_name'))
        
        with action_col3:
            if st.button(f"⚖️ Add to Compare", key=f"compare_{index}", use_container_width=True):
                add_course_to_comparison(recommendation)
        
        with action_col4:
            if st.button(f"❤️ Save to Favorites", key=f"save_{index}", use_container_width=True):
                save_course_to_favorites(recommendation.get('course_name'))
        
        with action_col5:
            if st.button(f"📤 Export Details", key=f"export_{index}", use_container_width=True):
                export_course_details(recommendation)

def add_course_to_comparison(course: Dict):
    """Add course to comparison with enhanced feedback"""
    if 'selected_courses_for_comparison' not in st.session_state:
        st.session_state.selected_courses_for_comparison = []
    
    # Check if already in comparison
    existing_names = [c.get('course_name') for c in st.session_state.selected_courses_for_comparison]
    if course.get('course_name') not in existing_names:
        st.session_state.selected_courses_for_comparison.append(course)
        st.success(f"✅ Added {course.get('course_name')} to comparison ({len(st.session_state.selected_courses_for_comparison)} courses total)")
        
        if len(st.session_state.selected_courses_for_comparison) >= 2:
            st.info("🎯 You can now view the course comparison in the Compare section!")
    else:
        st.warning("Course already in comparison list")

def export_course_details(course: Dict):
    """Export individual course details"""
    try:
        course_report = f"""
# Course Details Report: {course.get('course_name', 'Unknown Course')}

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Course Information
- **Name:** {course.get('course_name', 'Unknown')}
- **Level:** {course.get('level', 'Unknown')}
- **Department:** {course.get('department', 'Unknown')}
- **Duration:** {course.get('duration', 'Unknown')}

## Fees & Requirements
- **International Fee:** {course.get('fees', 'Contact for details')}
- **Minimum GPA:** {course.get('min_gpa', 'Not specified')}
- **Minimum IELTS:** {course.get('min_ielts', 'Not specified')}

## Course Description
{course.get('description', 'No description available')}

## Career Prospects
{course.get('career_prospects', 'Various opportunities available')}

## Your Match Analysis
- **Match Score:** {course.get('score', 0):.1%}
- **Match Quality:** {course.get('match_quality', 'Good Match')}

### Why This Course Matches You:
"""
        
        for reason in course.get('reasons', []):
            course_report += f"• {reason}\n"
        
        course_report += f"""

## Next Steps
1. Contact admissions: admissions@uel.ac.uk
2. Prepare required documents
3. Submit application before deadline
4. Consider visiting campus or attending virtual sessions

---
Report generated by UEL AI Assistant
Your personalized university guidance system
"""
        
        st.download_button(
            label="📥 Download Course Details",
            data=course_report,
            file_name=f"uel_course_{course.get('course_name', 'unknown').replace(' ', '_').lower()}_{datetime.now().strftime('%Y%m%d')}.txt",
            mime="text/plain"
        )
        
        st.success("✅ Course details exported successfully!")
        
    except Exception as e:
        st.error(f"❌ Export failed: {e}")

# =============================================================================
# ENHANCED PREDICTIONS PAGE (RESTORED)
# =============================================================================

def render_enhanced_predictions_page():
    """Render enhanced predictions page with complete functionality"""
    # Require profile for predictions
    if not check_profile_access("Admission Predictions", flexible=False):
        return
    
    st.markdown("""
    <div class="page-header">
        <div class="header-icon">🔮</div>
        <div>
            <h1 style="margin: 0; background: var(--uel-gradient); -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-weight: 800; font-size: 2.5rem;">Complete Admission Prediction Center</h1>
            <p style="margin: 0; color: var(--text-secondary); font-size: 1.2rem; font-weight: 500;">AI analysis with comprehensive insights and improvement recommendations</p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    profile = st.session_state.current_profile
    
    # Enhanced profile data display
    st.markdown(f"""
    <div class="enhanced-card" style="background: linear-gradient(135deg, rgba(16, 185, 129, 0.1), rgba(16, 185, 129, 0.05)); border-left: 6px solid var(--uel-success);">
        <h3 style="margin-top: 0; color: var(--uel-success);">📊 Your Complete Academic Profile</h3>
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 1rem;">
            <div><strong>GPA:</strong> {profile.get('gpa', 0.0)}/4.0</div>
            <div><strong>IELTS:</strong> {profile.get('ielts_score', 0.0)}/9.0</div>
            <div><strong>Experience:</strong> {profile.get('work_experience_years', 0)} years</div>
            <div><strong>Level:</strong> {profile.get('academic_level', 'Unknown')}</div>
            <div><strong>Field:</strong> {profile.get('field_of_interest', 'Not specified')}</div>
            <div><strong>Country:</strong> {profile.get('country', 'Not specified')}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Course selection with enhanced interface
    st.markdown("### 🎯 Select Target Course for Prediction")
    
    try:
        ai_agent = get_enhanced_ai_agent()
        courses_data = ai_agent.data_manager.courses_df
        
        if courses_data.empty:
            st.error("❌ Course data not available.")
            return
        
        # Enhanced course filtering and selection
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Filter courses based on profile interests and level
            field_interest = profile.get('field_of_interest', '').lower()
            user_level = profile.get('academic_level', '').lower()
            
            # Create enhanced course options with recommendations
            course_options = []
            for _, course in courses_data.iterrows():
                course_name = course.get('course_name', 'Unknown')
                level = course.get('level', 'Unknown')
                fees = course.get('fees_international', 0)
                min_gpa = course.get('min_gpa', 0)
                
                # Smart highlighting
                highlight = ""
                if field_interest in course_name.lower():
                    highlight = "🎯 "  # Perfect match
                elif any(word in course_name.lower() for word in field_interest.split()):
                    highlight = "⭐ "  # Good match
                
                # Difficulty indicator
                difficulty = ""
                your_gpa = profile.get('gpa', 3.0)
                if min_gpa > your_gpa + 0.5:
                    difficulty = " (Reach)"
                elif min_gpa <= your_gpa - 0.3:
                    difficulty = " (Safety)"
                
                display_name = f"{highlight}{course_name} ({level}) - £{fees:,}{difficulty}"
                course_options.append((display_name, course_name))
            
            # Course selection
            if course_options:
                # Pre-select if coming from recommendations
                default_course = st.session_state.get('selected_course_for_prediction')
                default_index = 0
                
                if default_course:
                    for i, (_, course_name) in enumerate(course_options):
                        if course_name == default_course:
                            default_index = i
                            break
                
                course_selector = st.selectbox(
                    "Choose your target course:",
                    range(len(course_options)),
                    index=default_index,
                    format_func=lambda x: course_options[x][0] if x < len(course_options) else "",
                    help="🎯 = Perfect match for your interests, ⭐ = Good match, (Reach) = Above your stats, (Safety) = Below your stats"
                )
                
                selected_course = course_options[course_selector][1] if course_selector < len(course_options) else ""
            else:
                st.error("No courses available for prediction")
                return
        
        with col2:
            # Prediction options
            st.markdown("#### 🔧 Prediction Options")
            
            include_improvement_tips = st.checkbox("Include improvement recommendations", value=True)
            compare_with_similar = st.checkbox("Compare with similar applicants", value=True)
            detailed_analysis = st.checkbox("Detailed factor analysis", value=True)
        
        # Enhanced prediction button
        if st.button("🔮 Get Complete Admission Analysis", type="primary", use_container_width=True, key="__get_complete_admis_3034"):
            if selected_course:
                with st.spinner("🤖 AI is conducting comprehensive analysis of your admission chances..."):
                    prediction_result = predict_admission_success(selected_course)
                    
                    # Store prediction for later reference
                    st.session_state.prediction_results[selected_course] = prediction_result
                
                if 'error' not in prediction_result:
                    display_enhanced_prediction_result(
                        prediction_result, 
                        selected_course, 
                        profile,
                        include_improvement_tips,
                        compare_with_similar,
                        detailed_analysis
                    )
                else:
                    st.error(f"❌ Prediction error: {prediction_result.get('error')}")
            else:
                st.error("❌ Please select a course first")
        
        # Quick course comparison for predictions
        if len(st.session_state.prediction_results) > 1:
            st.markdown("### 📊 Compare Your Prediction Results")
            if st.button("📊 View Prediction Comparison", key="__view_prediction_co_3035"):
                render_prediction_comparison()
        
        # Clear selection option
        if st.session_state.get('selected_course_for_prediction'):
            if st.button("🔄 Clear Course Selection", key="__clear_course_selec_3036"):
                del st.session_state.selected_course_for_prediction
                st.rerun()
    
    except Exception as e:
        st.error(f"❌ Error loading courses: {e}")
        logger.error(f"Course loading error: {e}")

def display_enhanced_prediction_result(prediction_result: Dict, course_name: str, profile: Dict, 
                                     include_tips: bool, compare_similar: bool, detailed_analysis: bool):
    """Display enhanced prediction results with complete analysis"""
    success_prob = prediction_result.get('success_probability', 0.5)
    confidence = prediction_result.get('confidence', 0.7)
    
    # Enhanced status determination
    if success_prob >= 0.85:
        color = "#10b981"
        status = "Excellent"
        icon = "🎉"
        message = "Outstanding chances! You're a strong candidate."
    elif success_prob >= 0.7:
        color = "#3b82f6"
        status = "Very Good"
        icon = "🌟"
        message = "Strong profile with great potential!"
    elif success_prob >= 0.55:
        color = "#8b5cf6"
        status = "Good"
        icon = "👍"
        message = "Solid chances with some improvements possible."
    elif success_prob >= 0.4:
        color = "#f59e0b"
        status = "Moderate"
        icon = "⚠️"
        message = "Room for improvement to strengthen your application."
    else:
        color = "#ef4444"
        status = "Challenging"
        icon = "📚"
        message = "Significant preparation needed for competitive application."
    
    # Enhanced main prediction display
    st.markdown(f"""
    <div style="background: {color}; color: white; padding: 3rem; border-radius: 32px; text-align: center; margin: 2rem 0; box-shadow: 0 10px 30px rgba(0,0,0,0.2);">
        <div style="font-size: 4rem; margin-bottom: 1rem;">{icon}</div>
        <h1 style="margin: 0; font-size: 3.5rem; font-weight: 800;">{success_prob*100:.1f}%</h1>
        <h2 style="margin: 1rem 0; font-size: 2rem;">{status} Admission Chances</h2>
        <h3 style="margin: 0.5rem 0; font-size: 1.4rem;">for {course_name}</h3>
        <p style="font-size: 1rem; opacity: 0.9; margin-top: 1rem;">AI Analysis Confidence: {confidence:.0%}</p>
        <p style="font-size: 1.1rem; opacity: 0.9; margin-top: 0.5rem; font-style: italic;">{message}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Comprehensive analysis tabs
    if detailed_analysis:
        analysis_tab1, analysis_tab2, analysis_tab3, analysis_tab4 = st.tabs([
            "💪 Strengths", 
            "📈 Areas to Improve", 
            "💡 Action Plan", 
            "📊 Detailed Analysis"
        ])
        
        with analysis_tab1:
            render_strengths_analysis(prediction_result, profile)
        
        with analysis_tab2:
            render_improvement_areas(prediction_result, profile)
        
        with analysis_tab3:
            if include_tips:
                render_action_plan(prediction_result, profile, course_name)
        
        with analysis_tab4:
            render_detailed_factor_analysis(prediction_result, profile)
    
    # Enhanced quick actions
    st.markdown("### 🚀 What's Next?")
    
    action_col1, action_col2, action_col3, action_col4 = st.columns(4)
    
    with action_col1:
        if st.button("🎯 Find Similar Courses", use_container_width=True, key="__find_similar_cours_3037"):
            st.session_state.current_page = "recommendations"
            st.rerun()
    
    with action_col2:
        if st.button("💬 Discuss with AI", use_container_width=True, key="__discuss_with_ai_3038"):
            st.session_state.current_page = "ai_chat"
            # Pre-populate with prediction context
            context_message = f"I just got my admission prediction for {course_name}. My chances are {success_prob:.1%}. Can you help me understand what I should do next?"
            st.session_state.messages.append({
                'role': 'user',
                'content': context_message,
                'timestamp': datetime.now(),
                'context': 'prediction_followup'
            })
            st.rerun()
    
    with action_col3:
        if st.button("📈 Improve Profile", use_container_width=True, key="__improve_profile_3039"):
            st.session_state.show_profile_editor = True
            st.rerun()
    
    with action_col4:
        if st.button("📊 Export Analysis", use_container_width=True, key="__export_analysis_3040"):
            export_prediction_analysis(prediction_result, course_name, profile)

def render_strengths_analysis(prediction_result: Dict, profile: Dict):
    """Render detailed strengths analysis"""
    st.markdown("### 💪 Your Application Strengths")
    
    insights = prediction_result.get('insights', {})
    strengths = insights.get('strengths', [])
    
    # Add profile-based strength analysis
    profile_strengths = []
    
    gpa = profile.get('gpa', 0)
    if gpa >= 3.7:
        profile_strengths.append("🎓 Exceptional academic performance (GPA 3.7+)")
    elif gpa >= 3.3:
        profile_strengths.append("📚 Strong academic performance")
    
    ielts = profile.get('ielts_score', 0)
    if ielts >= 7.5:
        profile_strengths.append("🗣️ Outstanding English proficiency")
    elif ielts >= 6.5:
        profile_strengths.append("✅ Good English proficiency")
    
    work_exp = profile.get('work_experience_years', 0)
    if work_exp >= 3:
        profile_strengths.append("💼 Valuable professional experience")
    elif work_exp >= 1:
        profile_strengths.append("👔 Relevant work experience")
    
    interests = profile.get('interests', [])
    if len(interests) >= 3:
        profile_strengths.append("🎯 Diverse academic interests")
    
    if profile.get('career_goals'):
        profile_strengths.append("🎯 Clear career direction and goals")
    
    # Combine and display all strengths
    all_strengths = strengths + profile_strengths
    
    if all_strengths:
        for strength in all_strengths:
            st.success(f"✅ {strength}")
    else:
        st.info("💡 Keep building your academic profile to develop clear strengths. Consider improving your GPA, gaining work experience, or taking English proficiency tests.")

def render_improvement_areas(prediction_result: Dict, profile: Dict):
    """Render improvement areas analysis"""
    st.markdown("### 📈 Areas for Improvement")
    
    insights = prediction_result.get('insights', {})
    improvements = insights.get('areas_for_improvement', [])
    
    # Add profile-based improvement suggestions
    profile_improvements = []
    
    gpa = profile.get('gpa', 0)
    if gpa < 3.0:
        profile_improvements.append("📚 Improve academic performance - aim for GPA 3.0+")
    elif gpa < 3.5:
        profile_improvements.append("🎯 Consider retaking courses to boost GPA to 3.5+")
    
    ielts = profile.get('ielts_score', 0)
    if ielts < 6.5:
        profile_improvements.append("📖 Improve English proficiency - aim for IELTS 6.5+")
    elif ielts < 7.0:
        profile_improvements.append("🗣️ Consider retaking IELTS for higher score (7.0+)")
    
    if not profile.get('work_experience_years', 0):
        profile_improvements.append("💼 Gain relevant work experience or internships")
    
    if not profile.get('career_goals'):
        profile_improvements.append("🎯 Develop and articulate clear career goals")
    
    if len(profile.get('interests', [])) < 2:
        profile_improvements.append("🎨 Explore and develop broader academic interests")
    
    # Combine and display improvements
    all_improvements = improvements + profile_improvements
    
    if all_improvements:
        for improvement in all_improvements:
            st.warning(f"📌 {improvement}")
    else:
        st.success("🌟 Your profile looks well-balanced! Focus on maintaining your current performance and preparing strong application materials.")

def render_action_plan(prediction_result: Dict, profile: Dict, course_name: str):
    """Render personalized action plan"""
    st.markdown("### 💡 Your Personalized Action Plan")
    
    recommendations = prediction_result.get('recommendations', [])
    success_prob = prediction_result.get('success_probability', 0.5)
    
    # Generate timeline-based action plan
    immediate_actions = []
    if not st.session_state.verification_results:
        immediate_actions.append("📄 Upload and verify your academic documents")
    
    immediate_actions.extend([
        "📝 Draft your personal statement specifically for this program",
        "📧 Contact the admissions team for specific guidance",
        "🔍 Research the course curriculum and faculty"
    ])
    
    # Short-term actions (next 2-3 months)
    short_term_actions = []
    
    if profile.get('ielts_score', 0) < 6.5:
        short_term_actions.append("📚 Prepare for and retake IELTS test")
    
    if success_prob < 0.6:
        short_term_actions.append("🎓 Consider taking additional coursework to strengthen application")
        short_term_actions.append("💼 Gain relevant work experience or volunteer in the field")
    
    short_term_actions.extend([
        "🏫 Attend virtual or in-person campus visits",
        "👥 Connect with current students or alumni",
        "📚 Complete any prerequisite courses if required"
    ])
    
    # Long-term actions (next 6-12 months)
    long_term_actions = [
        "📅 Submit application before early deadline for better chances",
        "🎯 Apply to multiple programs to increase options",
        "💰 Research and apply for scholarships",
        "🏠 Investigate accommodation options if accepted"
    ]
    
    # Display action plan
    timeline_col1, timeline_col2, timeline_col3 = st.columns(3)
    
    with timeline_col1:
        st.markdown("#### 🚨 Immediate (Next 2 weeks)")
        for action in immediate_actions:
            st.markdown(f"• {action}")
    
    with timeline_col2:
        st.markdown("#### ⏰ Short-term (Next 2-3 months)")
        for action in short_term_actions:
            st.markdown(f"• {action}")
    
    with timeline_col3:
        st.markdown("#### 🎯 Long-term (Next 6-12 months)")
        for action in long_term_actions:
            st.markdown(f"• {action}")
    
    # AI-generated recommendations
    if recommendations:
        st.markdown("#### 🤖 AI-Generated Recommendations")
        for rec in recommendations:
            st.markdown(f"• {rec}")

def render_detailed_factor_analysis(prediction_result: Dict, profile: Dict):
    """Render detailed factor analysis"""
    st.markdown("### 📊 Detailed Factor Analysis")
    
    # Create factor importance visualization
    factors = {
        'GPA': profile.get('gpa', 0) / 4.0,
        'IELTS Score': profile.get('ielts_score', 0) / 9.0,
        'Work Experience': min(profile.get('work_experience_years', 0) / 5.0, 1.0),
        'Field Match': 0.9 if profile.get('field_of_interest') else 0.3,
        'Profile Completeness': profile.get('profile_completion', 0) / 100
    }
    
    # Display factors
    for factor, score in factors.items():
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            st.markdown(f"**{factor}**")
        
        with col2:
            st.progress(score)
        
        with col3:
            color = "#10b981" if score >= 0.8 else "#f59e0b" if score >= 0.6 else "#ef4444"
            status = "Strong" if score >= 0.8 else "Good" if score >= 0.6 else "Needs Work"
            st.markdown(f"<span style='color: {color}; font-weight: bold;'>{status}</span>", unsafe_allow_html=True)

def render_prediction_comparison():
    """Render prediction comparison interface"""
    st.markdown("### 📊 Your Prediction Results Comparison")
    
    prediction_results = st.session_state.prediction_results
    
    if len(prediction_results) < 2:
        st.info("Make predictions for at least 2 courses to compare results.")
        return
    
    # Create comparison data
    comparison_data = []
    for course, result in prediction_results.items():
        comparison_data.append({
            'Course': course,
            'Success Probability': result.get('success_probability', 0),
            'Confidence': result.get('confidence', 0.5),
            'Status': 'Excellent' if result.get('success_probability', 0) >= 0.8 else 
                     'Good' if result.get('success_probability', 0) >= 0.6 else 
                     'Moderate'
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    
    # Display comparison table
    st.dataframe(comparison_df, use_container_width=True)
    
    # Visualization
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.bar(
            comparison_df,
            x='Course',
            y='Success Probability',
            title="Admission Probability Comparison",
            color='Success Probability',
            color_continuous_scale='RdYlGn'
        )
        fig.update_layout(xaxis={'tickangle': 45})
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.pie(
            comparison_df,
            names='Status',
            title="Prediction Status Distribution"
        )
        st.plotly_chart(fig, use_container_width=True)

def export_prediction_analysis(prediction_result: Dict, course_name: str, profile: Dict):
    """Export prediction analysis report"""
    try:
        success_prob = prediction_result.get('success_probability', 0.5)
        confidence = prediction_result.get('confidence', 0.7)
        
        analysis_report = f"""
# Admission Prediction Analysis Report

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Course:** {course_name}
**Student:** {profile.get('first_name', '')} {profile.get('last_name', '')}

## Prediction Summary
- **Admission Probability:** {success_prob:.1%}
- **Analysis Confidence:** {confidence:.1%}
- **Recommendation:** {'Highly Recommended' if success_prob >= 0.8 else 'Recommended' if success_prob >= 0.6 else 'Consider with Preparation'}

## Student Profile Overview
- **GPA:** {profile.get('gpa', 0.0)}/4.0
- **IELTS Score:** {profile.get('ielts_score', 0.0)}/9.0
- **Work Experience:** {profile.get('work_experience_years', 0)} years
- **Academic Level:** {profile.get('academic_level', 'Not specified')}
- **Field of Interest:** {profile.get('field_of_interest', 'Not specified')}

## Strengths
"""
        
        # Add strengths
        insights = prediction_result.get('insights', {})
        strengths = insights.get('strengths', [])
        for strength in strengths:
            analysis_report += f"• {strength}\n"
        
        analysis_report += """
## Areas for Improvement
"""
        
        # Add improvements
        improvements = insights.get('areas_for_improvement', [])
        for improvement in improvements:
            analysis_report += f"• {improvement}\n"
        
        analysis_report += """
## Recommendations
"""
        
        # Add recommendations
        recommendations = prediction_result.get('recommendations', [])
        for rec in recommendations:
            analysis_report += f"• {rec}\n"
        
        analysis_report += f"""

## Next Steps
1. Contact UEL admissions team: admissions@uel.ac.uk
2. Prepare required application documents
3. Consider the improvement areas mentioned above
4. Apply before the deadline for best chances

---
Analysis generated by UEL AI Assistant
Report ID: {datetime.now().strftime('%Y%m%d_%H%M%S')}
"""
        
        st.download_button(
            label="📥 Download Prediction Analysis",
            data=analysis_report,
            file_name=f"uel_prediction_analysis_{course_name.replace(' ', '_').lower()}_{datetime.now().strftime('%Y%m%d')}.txt",
            mime="text/plain"
        )
        
        st.success("✅ Prediction analysis exported successfully!")
        
    except Exception as e:
        st.error(f"❌ Export failed: {e}")

# =============================================================================
# ENHANCED DASHBOARD (RESTORED)
# =============================================================================

def render_enhanced_dashboard():
    """Render enhanced dashboard with complete functionality"""
    st.markdown("""
    <div class="page-header">
        <div class="header-icon">🏠</div>
        <div>
            <h1 style="margin: 0; background: var(--uel-gradient); -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-weight: 800; font-size: 2.5rem;">Your Personal Dashboard</h1>
            <p style="margin: 0; color: var(--text-secondary); font-size: 1.2rem; font-weight: 500;">Welcome to your complete UEL AI experience</p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    profile = st.session_state.current_profile
    
    if profile:
        render_personalized_dashboard(profile)
    else:
        render_welcome_dashboard()

def render_personalized_dashboard(profile: Dict):
    """Render personalized dashboard for logged-in users"""
    # Personalized welcome with enhanced stats
    st.markdown(f"""
    <div class="enhanced-card" style="background: var(--uel-gradient); color: white; text-align: center;">
        <h2 style="margin-top: 0; font-size: 2.5rem;">👋 Welcome back, {profile.get('first_name', 'Student')}!</h2>
        <p style="font-size: 1.3rem; margin: 1rem 0;">Ready to continue your {profile.get('field_of_interest', 'academic')} journey?</p>
        <div style="margin-top: 2rem;">
            <div style="display: inline-block; padding: 0.8rem 1.5rem; background: rgba(255,255,255,0.2); border-radius: 25px; margin: 0.5rem; font-weight: 600;">
                🎯 Profile: {profile.get('profile_completion', 0):.0f}% Complete
            </div>
            <div style="display: inline-block; padding: 0.8rem 1.5rem; background: rgba(255,255,255,0.2); border-radius: 25px; margin: 0.5rem; font-weight: 600;">
                🎓 {profile.get('academic_level', 'Student').title()}
            </div>
            <div style="display: inline-block; padding: 0.8rem 1.5rem; background: rgba(255,255,255,0.2); border-radius: 25px; margin: 0.5rem; font-weight: 600;">
                💬 {len(st.session_state.messages)} Conversations
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Enhanced activity overview
    st.markdown("## 📊 Your Comprehensive Activity Overview")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("Profile Completion", f"{profile.get('profile_completion', 0):.0f}%")
    
    with col2:
        interaction_count = profile.get('interaction_count', 0)
        st.metric("AI Interactions", interaction_count)
    
    with col3:
        interests_count = len(profile.get('interests', []))
        st.metric("Interest Areas", interests_count)
    
    with col4:
        docs_count = len(st.session_state.verification_results)
        st.metric("Documents Uploaded", docs_count)
    
    with col5:
        days_active = (datetime.now() - datetime.fromisoformat(profile.get('created_date', datetime.now().isoformat()))).days
        st.metric("Days Active", days_active)
    
    # Comprehensive feature grid
    render_personalized_feature_grid(profile)
    
    # Profile improvement suggestions (enhanced)
    completion = profile.get('profile_completion', 0)
    if completion < 100:
        render_profile_improvement_suggestions(profile, completion)

def render_welcome_dashboard():
    """Render welcome dashboard for new users"""
    st.markdown("""
    <div class="enhanced-card" style="background: linear-gradient(135deg, rgba(59, 130, 246, 0.1), rgba(59, 130, 246, 0.05)); text-align: center;">
        <h2 style="margin-top: 0; color: #3b82f6;">🎓 Welcome to UEL AI Assistant</h2>
        <p>Your intelligent guide to University of East London programs and admissions</p>
    </div>
    """, unsafe_allow_html=True)
    
    # General information
    st.markdown("### 🏛️ About University of East London")
    st.markdown("""
    The University of East London (UEL) is a public university located in London, England. 
    We offer a wide range of undergraduate and postgraduate programs with a focus on 
    practical, career-oriented education.
    """)
    
    # Available features overview
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="enhanced-card" style="text-align: center;">
            <h3 style="color: var(--uel-primary);">💬 AI Chat</h3>
            <p>Get instant answers about UEL programs, admissions, and campus life</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="enhanced-card" style="text-align: center;">
            <h3 style="color: var(--uel-secondary);">🔍 Course Search</h3>
            <p>Browse our extensive catalog of programs and find what interests you</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="enhanced-card" style="text-align: center;">
            <h3 style="color: var(--uel-success);">📊 Analytics</h3>
            <p>View program statistics, trends, and university information</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Encourage profile creation
    st.markdown("### 🎯 Get Personalized Experience")
    st.info("Create a profile to unlock personalized course recommendations, admission predictions, and tailored guidance!")
    
    # General quick actions
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("💬 Chat with AI", use_container_width=True, key="__chat_with_ai_3041"):
            st.session_state.current_page = "ai_chat"
            st.rerun()
    
    with col2:
        if st.button("🔍 Browse Courses", use_container_width=True, key="__browse_courses_3042"):
            st.session_state.current_page = "search"
            st.rerun()
    
    with col3:
        if st.button("📝 Create Profile", use_container_width=True, key="__create_profile_3043"):
            st.session_state.current_page = "profile_setup"
            st.rerun()
    
    with col4:
        if st.button("📊 View Analytics", use_container_width=True, key="__view_analytics_3044"):
            st.session_state.current_page = "analytics"
            st.rerun()

def render_personalized_feature_grid(profile: Dict):
    """Render personalized feature grid"""
    st.markdown("## 🚀 All Features Available to You")
    
    feature_grid_col1, feature_grid_col2, feature_grid_col3 = st.columns(3)
    feature_grid3_col1, feature_grid3_col2, feature_grid3_col3 = st.columns(3)
    
    with feature_grid_col1:
        st.markdown(f"""
        <div class="enhanced-card" style="text-align: center; border-left: 4px solid var(--uel-primary); min-height: 200px;">
            <h3 style="color: var(--uel-primary); margin-top: 0;">🎯 Smart Recommendations</h3>
            <p>Get AI-matched courses for {profile.get('field_of_interest', 'your interests')}</p>
            <p><strong>Latest:</strong> {len(get_course_recommendations())} matches found</p>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("Get My Recommendations", type="primary", use_container_width=True, key="get_my_recommendatio_3045"):
            st.session_state.current_page = "recommendations"
            st.rerun()
    
    with feature_grid3_col1:
        st.markdown(f"""
        <div class="enhanced-card" style="text-align: center; border-left: 4px solid #8b5cf6; min-height: 180px;">
            <h3 style="color: #8b5cf6; margin-top: 0;">🎤 Interview Preparation</h3>
            <p>AI-powered mock interviews and practice sessions</p>
            <p><strong>Features:</strong> Voice practice, AI feedback, analytics</p>
        </div>
        """, unsafe_allow_html=True)
    
        if st.button("Start Interview Prep", use_container_width=True, key="start_interview_prep_dashboard"):
            st.session_state.current_page = "interview_preparation"
            st.rerun()
    


    with feature_grid_col2:
        st.markdown(f"""
        <div class="enhanced-card" style="text-align: center; border-left: 4px solid var(--uel-secondary); min-height: 200px;">
            <h3 style="color: var(--uel-secondary); margin-top: 0;">🔮 Admission Predictions</h3>
            <p>Check your chances with your current profile</p>
            <p><strong>Profile Strength:</strong> {'Strong' if profile.get('gpa', 0) >= 3.5 else 'Good' if profile.get('gpa', 0) >= 3.0 else 'Developing'}</p>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("Check My Chances", type="primary", use_container_width=True, key="check_my_chances_3046"):
            st.session_state.current_page = "predictions"
            st.rerun()
    
    with feature_grid_col3:
        st.markdown(f"""
        <div class="enhanced-card" style="text-align: center; border-left: 4px solid var(--uel-success); min-height: 200px;">
            <h3 style="color: var(--uel-success); margin-top: 0;">💬 AI Assistant</h3>
            <p>Ask personalized questions about your journey</p>
            <p><strong>Conversations:</strong> {len(st.session_state.messages)} messages</p>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("Start AI Chat", type="primary", use_container_width=True, key="start_ai_chat_3047"):
            st.session_state.current_page = "ai_chat"
            st.rerun()
    
    # Second row of features
    feature_grid2_col1, feature_grid2_col2, feature_grid2_col3 = st.columns(3)
    
    with feature_grid2_col1:
        docs_count = len(st.session_state.verification_results)
        st.markdown(f"""
        <div class="enhanced-card" style="text-align: center; border-left: 4px solid var(--uel-info); min-height: 180px;">
            <h3 style="color: var(--uel-info); margin-top: 0;">📄 Document Center</h3>
            <p>Upload and verify your academic documents</p>
            <p><strong>Status:</strong> {docs_count} documents uploaded</p>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("Manage Documents", use_container_width=True, key="manage_documents_3048"):
            st.session_state.current_page = "documents"
            st.rerun()
    
    with feature_grid2_col2:
        st.markdown(f"""
        <div class="enhanced-card" style="text-align: center; border-left: 4px solid var(--uel-warning); min-height: 180px;">
            <h3 style="color: var(--uel-warning); margin-top: 0;">🔍 Advanced Search</h3>
            <p>Find courses with powerful filters and AI search</p>
            <p><strong>Features:</strong> Voice search, comparison tools</p>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("Advanced Search", use_container_width=True, key="advanced_search_3049"):
            st.session_state.current_page = "search"
            st.rerun()
    
    with feature_grid2_col3:
        st.markdown(f"""
        <div class="enhanced-card" style="text-align: center; border-left: 4px solid var(--uel-accent); min-height: 180px;">
            <h3 style="color: var(--uel-accent); margin-top: 0;">📊 Analytics & Export</h3>
            <p>View insights and export your data</p>
            <p><strong>Available:</strong> Reports, charts, data export</p>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("View Analytics", use_container_width=True, key="view_analytics_3050"):
            st.session_state.current_page = "analytics"
            st.rerun()

def render_profile_improvement_suggestions(profile: Dict, completion: float):
    """Render profile improvement suggestions"""
    st.markdown("## 📈 Complete Your Profile for Better Results")
    
    suggestions = []
    if not profile.get('career_goals'):
        suggestions.append("Add your career goals for targeted recommendations")
    if not profile.get('interests'):
        suggestions.append("Add more interest areas for diverse matches")
    if not profile.get('email'):
        suggestions.append("Add your email address for updates")
    if profile.get('gpa', 0) == 0:
        suggestions.append("Update your GPA for accurate predictions")
    if profile.get('ielts_score', 0) == 0:
        suggestions.append("Add IELTS score for requirement matching")
    
    if suggestions:
        st.info(f"🎯 **Quick Wins:** {', '.join(suggestions[:3])}")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("📝 Complete Profile Now", use_container_width=True, key="__complete_profile_n_3051"):
                st.session_state.show_profile_editor = True
                st.rerun()
        with col2:
            if st.button("📚 View Profile Tips", use_container_width=True, key="__view_profile_tips_3052"):
                st.session_state.show_profile_completion_tips = True
                st.rerun()

# =============================================================================
# ENHANCED SETUP AND MAIN FUNCTION (RESTORED)
# =============================================================================

def setup_enhanced_streamlit():
    """Configure Streamlit with enhanced styling and complete features"""
    st.set_page_config(
        page_title="🎓 UEL AI Assistant - Complete System",
        page_icon="🎓",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Complete enhanced CSS with all features
    st.markdown("""
    <style>
        /* Enhanced CSS for complete feature set */
        :root {
            --uel-primary: #00B8A9;
            --uel-secondary: #16213E;
            --uel-accent: #E63946;
            --uel-success: #4ECDC4;
            --uel-warning: #FFE66D;
            --uel-info: #64B5F6;
            --uel-gradient: linear-gradient(135deg, #00B8A9, #16213E);
            --uel-card-gradient: linear-gradient(135deg, rgba(255,255,255,0.9), rgba(255,255,255,0.95));
            --text-primary: #2D3748;
            --text-secondary: #4A5568;
            --text-light: #718096;
            --shadow-light: 0 2px 4px rgba(0,0,0,0.1);
            --shadow-medium: 0 4px 6px rgba(0,0,0,0.15);
            --shadow-strong: 0 10px 15px rgba(0,0,0,0.2);
        }
        
        .main {
            padding-top: 2rem;
        }
        
        .enhanced-card {
            background: var(--uel-card-gradient);
            border-radius: 16px;
            padding: 1.5rem;
            box-shadow: var(--shadow-medium);
            border: 1px solid rgba(0, 184, 169, 0.1);
            margin: 1rem 0;
            transition: all 0.3s ease;
        }
        
        .enhanced-card:hover {
            transform: translateY(-2px);
            box-shadow: var(--shadow-strong);
        }
        
        .page-header {
            display: flex;
            align-items: center;
            gap: 1rem;
            margin-bottom: 2rem;
            padding: 2rem;
            background: var(--uel-gradient);
            border-radius: 20px;
            color: white;
            box-shadow: var(--shadow-strong);
        }
        
        .header-icon {
            font-size: 3rem;
            opacity: 0.9;
        }
        
        .profile-progress-step {
            display: flex;
            align-items: center;
            padding: 1rem;
            margin: 0.5rem 0;
            border-radius: 16px;
            border: 2px solid transparent;
            transition: all 0.3s ease;
        }
        
        .profile-progress-step.completed {
            background: rgba(16, 185, 129, 0.1);
            border-color: #10b981;
        }
        
        .profile-progress-step.current {
            background: rgba(0, 184, 169, 0.1);
            border-color: var(--uel-primary);
        }
        
        .profile-progress-step.pending {
            background: rgba(156, 163, 175, 0.1);
            border-color: #9ca3af;
        }
        
        /* Enhanced tab styling */
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
        }
        
        .stTabs [data-baseweb="tab"] {
            height: 50px;
            padding-left: 20px;
            padding-right: 20px;
            border-radius: 10px;
            background: rgba(0, 184, 169, 0.1);
            transition: all 0.3s ease;
        }
        
        .stTabs [data-baseweb="tab"]:hover {
            background: rgba(0, 184, 169, 0.2);
        }
        
        .stTabs [data-baseweb="tab-highlight"] {
            background: var(--uel-primary);
        }
        
        /* Enhanced button styling */
        .stButton > button {
            border-radius: 8px;
            transition: all 0.3s ease;
        }
        
        .stButton > button:hover {
            transform: translateY(-1px);
            box-shadow: var(--shadow-light);
        }
        
        /* Enhanced form styling */
        .stForm {
            background: var(--uel-card-gradient);
            padding: 1.5rem;
            border-radius: 12px;
            border: 1px solid rgba(0, 184, 169, 0.1);
        }
        
        /* Enhanced expander styling */
        .streamlit-expanderHeader {
            background: rgba(0, 184, 169, 0.1);
            border-radius: 8px;
        }
        
        /* Enhanced dataframe styling */
        .stDataFrame {
            border-radius: 8px;
            overflow: hidden;
        }
    </style>
    """, unsafe_allow_html=True)




def render_research_evaluation_interface():
    """A+ Feature: Research evaluation interface for academic validation"""
    st.markdown("""
    <div class="page-header">
        <div class="header-icon">🔬</div>
        <div>
            <h1 style="margin: 0; background: var(--uel-gradient); -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-weight: 800; font-size: 2.5rem;">Research Evaluation Center</h1>
            <p style="margin: 0; color: var(--text-secondary); font-size: 1.2rem; font-weight: 500;">Academic validation and experimental analysis</p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    if not st.session_state.get('system_ready', False):
        st.error("❌ System not ready for research evaluation")
        return
    
    ai_agent = get_enhanced_ai_agent()
    
    # Check if research evaluator is available
    if not hasattr(ai_agent, 'research_evaluator'):
        st.warning("⚠️ Research evaluation framework not available. Please ensure advanced system is initialized.")
        return
    
    # Research evaluation controls
    st.markdown("### 🧪 Experimental Controls")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        num_test_profiles = st.number_input("Number of Test Profiles", 10, 100, 50, help="Number of synthetic profiles to generate for testing")
        include_bias_analysis = st.checkbox("Include Bias Analysis", value=True, help="Analyze bias across demographics")
    
    with col2:
        test_baseline_methods = st.multiselect(
            "Baseline Methods to Test",
            ["random", "popularity", "content_based", "collaborative"],
            default=["random", "popularity"],
            help="Select baseline methods for comparison"
        )
        include_statistical_tests = st.checkbox("Statistical Significance Testing", value=True, help="Perform statistical tests")
    
    with col3:
        export_results = st.checkbox("Export Results", value=True, help="Export evaluation results")
        detailed_analysis = st.checkbox("Detailed Analysis", value=True, help="Generate detailed analysis report")
    
    # Run comprehensive evaluation
    if st.button("🚀 Run Comprehensive Evaluation", type="primary", use_container_width=True, key="__run_comprehensive__3053"):
        with st.spinner("🔬 Conducting comprehensive academic evaluation..."):
            try:
                # Generate synthetic test profiles
                test_profiles = generate_test_profiles(num_test_profiles)
                
                # Run evaluation
                evaluation_results = ai_agent.research_evaluator.conduct_comprehensive_evaluation(test_profiles)
                
                # Store results
                st.session_state.research_evaluation_results = evaluation_results
                
                st.success(f"✅ Evaluation completed with {num_test_profiles} test profiles!")
                
            except Exception as e:
                st.error(f"❌ Evaluation failed: {e}")
                return
    
    # Display evaluation results
    if st.session_state.get('research_evaluation_results'):
        render_evaluation_results(st.session_state.research_evaluation_results)

def render_evaluation_results(results: Dict):
    """Display comprehensive evaluation results"""
    st.markdown("### 📊 Evaluation Results")
    
    # Key metrics overview
    rec_eval = results.get('recommendation_evaluation', {})
    pred_eval = results.get('prediction_evaluation', {})
    baseline_comp = results.get('baseline_comparison', {})
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        precision_5 = rec_eval.get('precision_at_k', {}).get('p@5', 0)
        st.metric("Precision@5", f"{precision_5:.3f}", help="Precision at top 5 recommendations")
    
    with col2:
        auc_score = pred_eval.get('auc_roc', 0)
        st.metric("AUC-ROC", f"{auc_score:.3f}", help="Area under ROC curve for predictions")
    
    with col3:
        diversity = rec_eval.get('diversity_scores', [])
        avg_diversity = np.mean(diversity) if diversity else 0
        st.metric("Avg Diversity", f"{avg_diversity:.3f}", help="Average recommendation diversity")
    
    with col4:
        coverage = rec_eval.get('catalog_coverage', 0)
        st.metric("Catalog Coverage", f"{coverage:.3f}", help="Fraction of courses recommended")
    
    # Detailed results in tabs
    result_tab1, result_tab2, result_tab3, result_tab4 = st.tabs([
        "📈 Recommendation Metrics", 
        "🔮 Prediction Analysis", 
        "⚖️ Baseline Comparison", 
        "🔍 Bias Analysis"
    ])
    
    with result_tab1:
        render_recommendation_metrics(rec_eval)
    
    with result_tab2:
        render_prediction_metrics(pred_eval)
    
    with result_tab3:
        render_baseline_comparison(baseline_comp)
    
    with result_tab4:
        bias_analysis = results.get('bias_analysis', {})
        render_bias_analysis(bias_analysis)
    
    # Export functionality
    if st.button("📥 Download Research Report", key="__download_research__3054"):
        ai_agent = get_enhanced_ai_agent()
        report = ai_agent.research_evaluator.generate_research_report()
        
        st.download_button(
            label="📄 Download Complete Report",
            data=report,
            file_name=f"uel_research_evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            mime="text/plain"
        )

def render_recommendation_metrics(rec_eval: Dict):
    """Display recommendation evaluation metrics"""
    st.markdown("#### 📊 Recommendation System Performance")
    
    precision_data = rec_eval.get('precision_at_k', {})
    recall_data = rec_eval.get('recall_at_k', {})
    
    if precision_data and recall_data:
        # Create precision/recall chart
        k_values = [1, 3, 5, 10]
        precision_values = [precision_data.get(f'p@{k}', 0) for k in k_values]
        recall_values = [recall_data.get(f'r@{k}', 0) for k in k_values]
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.line(
                x=k_values, 
                y=precision_values,
                title="Precision@K",
                labels={'x': 'K (Top K Recommendations)', 'y': 'Precision'}
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.line(
                x=k_values,
                y=recall_values,
                title="Recall@K", 
                labels={'x': 'K (Top K Recommendations)', 'y': 'Recall'}
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # Additional metrics
    st.markdown("#### 📈 Additional Quality Metrics")
    
    metrics_df = pd.DataFrame([
        {"Metric": "NDCG", "Value": rec_eval.get('avg_ndcg', 0), "Description": "Normalized Discounted Cumulative Gain"},
        {"Metric": "Catalog Coverage", "Value": rec_eval.get('catalog_coverage', 0), "Description": "Fraction of catalog items recommended"},
        {"Metric": "Avg Diversity", "Value": np.mean(rec_eval.get('diversity_scores', [])), "Description": "Average recommendation diversity"}
    ])
    
    st.dataframe(metrics_df, use_container_width=True)

def render_prediction_metrics(pred_eval: Dict):
    """Display prediction evaluation metrics"""
    st.markdown("#### 🔮 Prediction System Performance")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Regression metrics
        metrics_data = {
            'MSE': pred_eval.get('mse', 0),
            'MAE': pred_eval.get('mae', 0),
            'Calibration': pred_eval.get('calibration_score', 0)
        }
        
        fig = px.bar(
            x=list(metrics_data.keys()),
            y=list(metrics_data.values()),
            title="Prediction Quality Metrics"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Classification metrics
        auc_score = pred_eval.get('auc_roc', 0)
        
        # Create a simple gauge chart for AUC
        fig = px.bar(
            x=['AUC-ROC'],
            y=[auc_score],
            title="Classification Performance",
            range_y=[0, 1]
        )
        st.plotly_chart(fig, use_container_width=True)

def render_baseline_comparison(baseline_comp: Dict):
    """Display baseline comparison results"""
    st.markdown("#### ⚖️ Comparison with Baseline Methods")
    
    statistical_comps = baseline_comp.get('statistical_comparisons', {})
    
    if statistical_comps:
        comparison_data = []
        for method, stats in statistical_comps.items():
            comparison_data.append({
                'Method': method,
                'Baseline Score': stats.get('baseline_mean', 0),
                'Our System Score': stats.get('ensemble_mean', 0),
                'Improvement %': stats.get('improvement_percentage', 0),
                'Significant': '✅ Yes' if stats.get('significant', False) else '❌ No'
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        st.dataframe(comparison_df, use_container_width=True)
        
        # Visualization
        fig = px.bar(
            comparison_df,
            x='Method',
            y=['Baseline Score', 'Our System Score'],
            title="Performance Comparison vs Baselines",
            barmode='group'
        )
        st.plotly_chart(fig, use_container_width=True)

def render_bias_analysis(bias_analysis: Dict):
    """Display bias analysis results"""
    st.markdown("#### 🔍 Bias Analysis")
    
    if not bias_analysis:
        st.info("No bias analysis data available")
        return
    
    # Country bias analysis
    country_bias = bias_analysis.get('country_bias', {})
    if country_bias:
        st.markdown("##### 🌍 Geographic Bias Analysis")
        
        country_data = []
        for country, data in country_bias.items():
            country_data.append({
                'Country': country,
                'Sample Size': data.get('sample_size', 0),
                'Avg Recommendation Score': data.get('avg_recommendation_score', 0),
                'Score Std Dev': data.get('score_std', 0)
            })
        
        country_df = pd.DataFrame(country_data)
        st.dataframe(country_df, use_container_width=True)
        
        # Visualization
        if len(country_data) > 1:
            fig = px.box(
                country_df,
                x='Country',
                y='Avg Recommendation Score',
                title="Recommendation Score Distribution by Country"
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # Fairness metrics
    fairness_metrics = bias_analysis.get('fairness_metrics', {})
    if fairness_metrics:
        st.markdown("##### ⚖️ Fairness Metrics")
        
        fairness_df = pd.DataFrame([
            {"Metric": "Demographic Parity", "Value": fairness_metrics.get('demographic_parity', 0), "Interpretation": "Lower is better (less bias)"},
            {"Metric": "Equalized Odds", "Value": fairness_metrics.get('equalized_odds', 0), "Interpretation": "Closer to 0 is better"},
            {"Metric": "Calibration", "Value": fairness_metrics.get('calibration_across_groups', 0), "Interpretation": "Closer to 1 is better"}
        ])
        
        st.dataframe(fairness_df, use_container_width=True)

def generate_test_profiles(num_profiles: int) -> List[Dict]:
    """Generate synthetic test profiles for evaluation"""
    test_profiles = []
    
    fields_of_interest = ["Computer Science", "Business Management", "Data Science", "Engineering", "Psychology", "Medicine"]
    countries = ["United Kingdom", "India", "China", "Nigeria", "Pakistan", "USA", "Canada"]
    academic_levels = ["undergraduate", "postgraduate", "masters"]
    
    for i in range(num_profiles):
        profile = {
            'id': f"test_user_{i}",
            'first_name': f"TestUser{i}",
            'last_name': "Research",
            'field_of_interest': random.choice(fields_of_interest),
            'country': random.choice(countries),
            'academic_level': random.choice(academic_levels),
            'gpa': round(random.uniform(2.5, 4.0), 2),
            'ielts_score': round(random.uniform(5.5, 8.5), 1),
            'work_experience_years': random.randint(0, 8),
            'interests': random.sample(["Technology", "Business", "Research", "Healthcare"], k=random.randint(1, 3)),
            'career_goals': f"Career in {random.choice(fields_of_interest)}",
            'profile_completion': random.uniform(60, 100),
            'created_date': datetime.now().isoformat()
        }
        test_profiles.append(profile)
    
    return test_profiles



def main():
    """Enhanced main application with complete error handling and all features"""
    try:
        # Setup enhanced Streamlit
        setup_enhanced_streamlit()
        
        # Initialize comprehensive session state
        initialize_comprehensive_session_state()
        
        # Handle profile login dialog if shown
        if st.session_state.get('show_profile_login', False):
            render_profile_login_dialog()
            return
        
        # Get AI agent with error handling
        try:
            ai_agent = get_enhanced_ai_agent()
        except Exception as e:
            st.error(f"⚠️ AI system initialization error: {e}")
            st.info("Running in basic mode. Some features may be limited.")
        
        # Render profile status bar
        render_profile_status_bar()
        
        # Render enhanced sidebar
        render_enhanced_sidebar()
        
        # Handle modal dialogs and overlays
        handle_modal_dialogs()
        
        # Determine current page and route
        page = st.session_state.current_page
        
        try:
            if page == "profile_setup" or (not st.session_state.get('profile_active') and page in ["recommendations", "predictions"]):
                render_profile_setup_wizard()
            elif page == "dashboard":
                render_enhanced_dashboard()
            elif page == "ai_chat":
                render_enhanced_chat_page()
            elif page == "recommendations":
                render_enhanced_recommendations_page()
            elif page == "predictions":
                render_enhanced_predictions_page()
            elif page == "documents":
                render_advanced_document_verification()
            elif page == "interview_preparation":
                render_interview_preparation_center()  # ADD THIS LINE
            elif page == "analytics":
                render_comprehensive_analytics_dashboard()
            elif page == "search":
                render_advanced_search_interface()
            elif page == "comparison":
                render_course_comparison_interface()
            elif page == "export":
                render_export_and_reporting_interface()
            elif page == "voice":
                render_voice_services_interface()
            else:
                # Default to dashboard
                st.session_state.current_page = "dashboard"
                st.rerun()
        
        except Exception as page_error:
            st.error(f"❌ Page rendering error: {page_error}")
            logger.error(f"Page rendering error: {page_error}")
            
            # Error recovery options
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("🏠 Return to Dashboard", type="primary", use_container_width=True, key="__return_to_dashboar_3055"):
                    st.session_state.current_page = "dashboard"
                    st.rerun()
            
            with col2:
                if st.button("🔄 Refresh System", use_container_width=True, key="__refresh_system_3056"):
                    # Clear AI agent to force re-initialization
                    st.session_state.ai_agent = None
                    st.rerun()
            
            with col3:
                if st.button("📞 Contact Support", use_container_width=True, key="__contact_support_3057"):
                    st.info("📧 Contact: admissions@uel.ac.uk for technical support")
    
    except Exception as main_error:
        st.error(f"❌ Application error: {main_error}")
        logger.error(f"Main application error: {main_error}")
        
        st.info("Please refresh the page or contact support if the issue persists.")
        
        if st.button("🔄 Refresh Application", key="__refresh_applicatio_3058"):
            st.rerun()

def handle_modal_dialogs():
    """Handle all modal dialogs and overlays"""
    # Profile editor modal
    if st.session_state.get('show_profile_editor', False):
        render_comprehensive_profile_editor()
        
        if st.button("❌ Close Profile Editor", key="__close_profile_edit_3059"):
            st.session_state.show_profile_editor = False
            st.rerun()
    
    # Document verifier modal
    if st.session_state.get('show_document_verifier', False):
        render_advanced_document_verification()
        
        if st.button("❌ Close Document Verifier", key="__close_document_ver_3060"):
            st.session_state.show_document_verifier = False
            st.rerun()
    
    # Export dialog
    if st.session_state.get('show_export_dialog', False):
        render_export_and_reporting_interface()
        
        if st.button("❌ Close Export Dialog", key="__close_export_dialo_3061"):
            st.session_state.show_export_dialog = False
            st.rerun()

if __name__ == "__main__":
    main()