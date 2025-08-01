# Enhanced unified_uel_ai_system.py - Profile-Driven Integration

import random
import pandas as pd
import numpy as np
import streamlit as st
import re
import json
import requests
from datetime import datetime, timedelta
import os
from typing import Dict, List, Optional, Tuple, Any, Callable
import sqlite3
import pickle
import hashlib
import logging
import time
import psutil
import threading
import queue
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
from functools import wraps
from pathlib import Path
import asyncio
import aiohttp

import logging

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Create console handler with a higher log level
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)

# Create formatter and add it to the handler
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)

# Add the handler to the logger
logger.addHandler(ch)


# Try to import optional libraries
try:
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logging.warning("Scikit-learn not available. ML features will be limited.")

try:
    from textblob import TextBlob
    TEXTBLOB_AVAILABLE = True
except ImportError:
    TEXTBLOB_AVAILABLE = False
    logging.warning("TextBlob not available. Sentiment analysis will be limited.")

try:
    import speech_recognition as sr
    import pyttsx3
    VOICE_AVAILABLE = True
except ImportError:
    VOICE_AVAILABLE = False
    logging.warning("Voice libraries not available. Voice features will be disabled.")

try:
    import plotly.express as px
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    logging.warning("Plotly not available. Advanced charts will be limited.")

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Create a module-level logger function to avoid scope issues
def get_logger(name: str = __name__):
    """Get a logger instance"""
    return logging.getLogger(name)

# =============================================================================
# ENHANCED PROFILE MANAGEMENT SYSTEM
# =============================================================================

@dataclass
class UserProfile:
    """Enhanced user profile with comprehensive data"""
    # Basic Information
    id: str
    first_name: str
    last_name: str
    email: str = ""
    phone: str = ""
    date_of_birth: str = ""
    
    # Location & Demographics
    country: str = ""
    nationality: str = ""
    city: str = ""
    postal_code: str = ""
    
    # Academic Background
    academic_level: str = ""  # current education level
    field_of_interest: str = ""  # primary field
    current_institution: str = ""
    current_major: str = ""
    gpa: float = 0.0
    graduation_year: int = 0
    
    # English Proficiency
    ielts_score: float = 0.0
    toefl_score: int = 0
    other_english_cert: str = ""
    
    # Professional Background
    work_experience_years: int = 0
    current_job_title: str = ""
    target_industry: str = ""
    professional_skills: List[str] = field(default_factory=list)
    
    # Interests & Preferences
    interests: List[str] = field(default_factory=list)
    career_goals: str = ""
    preferred_study_mode: str = ""  # full-time, part-time, online
    preferred_start_date: str = ""
    budget_range: str = ""
    
    # Application History
    previous_applications: List[str] = field(default_factory=list)
    rejected_courses: List[str] = field(default_factory=list)
    preferred_courses: List[str] = field(default_factory=list)
    
    # System Data
    created_date: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_date: str = field(default_factory=lambda: datetime.now().isoformat())
    last_active: str = field(default_factory=lambda: datetime.now().isoformat())
    profile_completion: float = 0.0
    
    # AI Interaction History
    interaction_count: int = 0
    favorite_features: List[str] = field(default_factory=list)
    ai_preferences: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        """Convert profile to dictionary"""
        return {
            'id': self.id,
            'first_name': self.first_name,
            'last_name': self.last_name,
            'email': self.email,
            'phone': self.phone,
            'date_of_birth': self.date_of_birth,
            'country': self.country,
            'nationality': self.nationality,
            'city': self.city,
            'postal_code': self.postal_code,
            'academic_level': self.academic_level,
            'field_of_interest': self.field_of_interest,
            'current_institution': self.current_institution,
            'current_major': self.current_major,
            'gpa': self.gpa,
            'graduation_year': self.graduation_year,
            'ielts_score': self.ielts_score,
            'toefl_score': self.toefl_score,
            'other_english_cert': self.other_english_cert,
            'work_experience_years': self.work_experience_years,
            'current_job_title': self.current_job_title,
            'target_industry': self.target_industry,
            'professional_skills': self.professional_skills,
            'interests': self.interests,
            'career_goals': self.career_goals,
            'preferred_study_mode': self.preferred_study_mode,
            'preferred_start_date': self.preferred_start_date,
            'budget_range': self.budget_range,
            'previous_applications': self.previous_applications,
            'rejected_courses': self.rejected_courses,
            'preferred_courses': self.preferred_courses,
            'created_date': self.created_date,
            'updated_date': self.updated_date,
            'last_active': self.last_active,
            'profile_completion': self.profile_completion,
            'interaction_count': self.interaction_count,
            'favorite_features': self.favorite_features,
            'ai_preferences': self.ai_preferences
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'UserProfile':
        """Create profile from dictionary"""
        return cls(**data)
    
    def calculate_completion(self) -> float:
        """Calculate profile completion percentage"""
        fields_to_check = [
            'first_name', 'last_name', 'email', 'country', 'nationality',
            'academic_level', 'field_of_interest', 'gpa', 'ielts_score',
            'career_goals', 'interests'
        ]
        
        completed = 0
        for field in fields_to_check:
            value = getattr(self, field, None)
            if value:
                if isinstance(value, list) and len(value) > 0:
                    completed += 1
                elif isinstance(value, (str, int, float)) and value:
                    completed += 1
        
        self.profile_completion = (completed / len(fields_to_check)) * 100
        return self.profile_completion
    
    def update_activity(self):
        """Update last active timestamp"""
        self.last_active = datetime.now().isoformat()
        self.updated_date = datetime.now().isoformat()
    
    def add_interaction(self, feature_name: str):
        """Track feature usage"""
        self.interaction_count += 1
        self.update_activity()
        
        if feature_name not in self.favorite_features:
            self.favorite_features.append(feature_name)
        
        # Keep only top 5 favorite features
        if len(self.favorite_features) > 5:
            self.favorite_features = self.favorite_features[-5:]

class ProfileManager:
    """Enhanced profile management with persistence and validation"""
    
    def __init__(self, db_manager=None):
        self.db_manager = db_manager
        self.current_profile: Optional[UserProfile] = None
        self.profile_cache: Dict[str, UserProfile] = {}
        self.logger = logging.getLogger(__name__)
    
    def create_profile(self, profile_data: Dict) -> UserProfile:
        """Create new user profile with validation"""
        try:
            # Generate unique ID if not provided
            if 'id' not in profile_data:
                profile_data['id'] = self._generate_profile_id()
            
            # Validate required fields
            required_fields = ['first_name', 'last_name', 'field_of_interest']
            for field in required_fields:
                if not profile_data.get(field):
                    raise ValueError(f"Required field missing: {field}")
            
            # Create profile
            profile = UserProfile(**profile_data)
            profile.calculate_completion()
            
            # Save to database
            self.save_profile(profile)
            
            # Set as current profile
            self.current_profile = profile
            self.profile_cache[profile.id] = profile
            
            self.logger.info(f"Created profile for {profile.first_name} {profile.last_name}")
            return profile
            
        except Exception as e:
            self.logger.error(f"Error creating profile: {e}")
            raise
    
    def update_profile(self, profile_id: str, updates: Dict) -> UserProfile:
        """Update existing profile"""
        try:
            profile = self.get_profile(profile_id)
            if not profile:
                raise ValueError(f"Profile not found: {profile_id}")
            
            # Update fields
            for key, value in updates.items():
                if hasattr(profile, key):
                    setattr(profile, key, value)
            
            profile.calculate_completion()
            profile.update_activity()
            
            # Save changes
            self.save_profile(profile)
            
            self.logger.info(f"Updated profile {profile_id}")
            return profile
            
        except Exception as e:
            self.logger.error(f"Error updating profile: {e}")
            raise
    
    def get_profile(self, profile_id: str) -> Optional[UserProfile]:
        """Get profile by ID"""
        try:
            # Check cache first
            if profile_id in self.profile_cache:
                return self.profile_cache[profile_id]
            
            # Load from database
            if self.db_manager:
                profile_data = self._load_profile_from_db(profile_id)
                if profile_data:
                    profile = UserProfile.from_dict(profile_data)
                    self.profile_cache[profile_id] = profile
                    return profile
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error getting profile: {e}")
            return None
    
    def set_current_profile(self, profile: UserProfile):
        """Set the current active profile"""
        self.current_profile = profile
        profile.update_activity()
        self.save_profile(profile)
        
        # Store in session state for Streamlit
        if 'st' in globals():
            st.session_state.current_profile = profile
            st.session_state.profile_active = True
    
    def get_current_profile(self) -> Optional[UserProfile]:
        """Get current active profile"""
        return self.current_profile
    
    def save_profile(self, profile: UserProfile):
        """Save profile to database"""
        try:
            if self.db_manager:
                self._save_profile_to_db(profile)
            
            # Update cache
            self.profile_cache[profile.id] = profile
            
        except Exception as e:
            self.logger.error(f"Error saving profile: {e}")
    
    def _generate_profile_id(self) -> str:
        """Generate unique profile ID"""
        timestamp = int(time.time() * 1000)
        random_part = random.randint(1000, 9999)
        return f"UEL_{timestamp}_{random_part}"
    
    def _load_profile_from_db(self, profile_id: str) -> Optional[Dict]:
        """Load profile from database"""
        try:
            conn = self.db_manager.get_connection()
            cursor = conn.cursor()
            
            cursor.execute("SELECT data FROM student_profiles WHERE id = ?", (profile_id,))
            result = cursor.fetchone()
            
            if result:
                return json.loads(result[0])
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error loading profile from DB: {e}")
            return None
    
    def _save_profile_to_db(self, profile: UserProfile):
        """Save profile to database"""
        try:
            conn = self.db_manager.get_connection()
            cursor = conn.cursor()
            
            # Create table if not exists
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS student_profiles (
                    id TEXT PRIMARY KEY,
                    data TEXT NOT NULL,
                    created_date TEXT,
                    updated_date TEXT
                )
            """)
            
            # Save profile
            profile_json = json.dumps(profile.to_dict())
            cursor.execute("""
                INSERT OR REPLACE INTO student_profiles 
                (id, data, created_date, updated_date) 
                VALUES (?, ?, ?, ?)
            """, (
                profile.id, 
                profile_json, 
                profile.created_date, 
                profile.updated_date
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"Error saving profile to DB: {e}")

# =============================================================================
# CONFIGURATION AND ENUMS
# =============================================================================

class AIModelType(Enum):
    """Available AI model types"""
    DEEPSEEK_CODER = "deepseek-coder"
    DEEPSEEK_LATEST = "deepseek:latest"
    LLAMA2 = "llama2"
    CODELLAMA = "codellama"
    MISTRAL = "mistral"
    PHI = "phi"

class DocumentType(Enum):
    """Supported document types for verification"""
    TRANSCRIPT = "transcript"
    IELTS_CERTIFICATE = "ielts_certificate"
    PASSPORT = "passport"
    PERSONAL_STATEMENT = "personal_statement"
    REFERENCE_LETTER = "reference"

@dataclass
class SystemConfig:
    """Main system configuration"""
    # Database
    database_path: str = "uel_ai_system.db"
    data_directory: str = "/Users/muhammadahmed/Desktop/uel-enhanced-ai-assistant/data"
    
    # AI Models
    ollama_host: str = "http://localhost:11434"
    default_model: str = AIModelType.DEEPSEEK_LATEST.value
    llm_temperature: float = 0.7
    max_tokens: int = 1000
    
    # ML Settings
    enable_ml_predictions: bool = True
    enable_sentiment_analysis: bool = True
    enable_document_verification: bool = True
    
    # Performance
    cache_enabled: bool = True
    cache_ttl: int = 3600
    max_cache_size: int = 1000
    
    # Security
    session_timeout_minutes: int = 60
    max_file_size_mb: int = 10
    allowed_file_types: List[str] = field(default_factory=lambda: ['pdf', 'jpg', 'jpeg', 'png', 'doc', 'docx'])
    
    # University Info
    university_name: str = "University of East London"
    university_short_name: str = "UEL"
    admissions_email: str = "admissions@uel.ac.uk"
    admissions_phone: str = "+44 20 8223 3000"

# Global configuration
config = SystemConfig()

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def format_currency(amount: float, currency: str = "GBP") -> str:
    """Format currency amount"""
    symbol_map = {"GBP": "£", "USD": "$", "EUR": "€"}
    symbol = symbol_map.get(currency, currency)
    return f"{symbol}{amount:,.0f}"

def format_date(date_str: str) -> str:
    """Format date string"""
    try:
        if isinstance(date_str, str):
            dt = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
            return dt.strftime("%b %d, %Y")
    except:
        pass
    return date_str

def format_duration(seconds: float) -> str:
    """Format duration in human-readable format"""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}m"
    else:
        return f"{seconds/3600:.1f}h"

def get_status_color(status: str) -> str:
    """Get color for status badges"""
    color_map = {
        'inquiry': '#3b82f6',
        'application_started': '#f59e0b',
        'under_review': '#6366f1',
        'accepted': '#10b981',
        'rejected': '#ef4444',
        'submitted': '#8b5cf6',
        'draft': '#6b7280',
        'verified': '#10b981',
        'pending': '#f59e0b'
    }
    return color_map.get(status, '#6b7280')

def get_level_color(level: str) -> str:
    """Get color for academic level badges"""
    color_map = {
        'undergraduate': '#3b82f6',
        'postgraduate': '#8b5cf6',
        'masters': '#10b981',
        'phd': '#ef4444'
    }
    return color_map.get(level.lower(), '#6b7280')

def safe_get(dictionary: Dict, key: str, default: Any = None) -> Any:
    """Safely get value from dictionary"""
    try:
        return dictionary.get(key, default) if dictionary else default
    except:
        return default

def generate_sample_data():
    """Generate comprehensive sample data for demonstration"""
    import random
    
    # Sample students
    first_names = ["John", "Emma", "Michael", "Sophia", "William", "Olivia", "James", "Ava"]
    last_names = ["Smith", "Johnson", "Williams", "Jones", "Brown", "Davis", "Miller", "Wilson"]
    countries = ["United Kingdom", "United States", "India", "China", "Nigeria", "Pakistan", "Canada"]
    fields = ["Computer Science", "Business Management", "Engineering", "Psychology", "Medicine", "Law"]
    
    students = []
    for i in range(20):
        first = random.choice(first_names)
        last = random.choice(last_names)
        students.append({
            'id': i + 1,
            'first_name': first,
            'last_name': last,
            'email': f"{first.lower()}.{last.lower()}{random.randint(1, 999)}@example.com",
            'country': random.choice(countries),
            'nationality': random.choice(countries),
            'field_of_interest': random.choice(fields),
            'academic_level': random.choice(['undergraduate', 'postgraduate', 'masters']),
            'status': random.choice(['inquiry', 'application_started', 'under_review', 'accepted']),
            'phone': f"+44 {random.randint(1000000000, 9999999999)}",
            'documents': []
        })
    
    # Sample courses
    course_names = ["Computer Science", "Business Management", "Data Science", "Engineering", "Psychology"]
    departments = ["School of Computing", "Business School", "School of Engineering", "School of Psychology"]
    
    courses = []
    for i, name in enumerate(course_names):
        courses.append({
            'id': i + 1,
            'course_name': name,
            'course_code': f"UEL{random.randint(1000, 9999)}",
            'department': random.choice(departments),
            'level': random.choice(['undergraduate', 'postgraduate', 'masters']),
            'duration': random.choice(['1 year', '2 years', '3 years']),
            'description': f"Comprehensive {name.lower()} program at UEL with excellent career prospects.",
            'fees': {
                'domestic': random.randint(9000, 12000),
                'international': random.randint(13000, 18000)
            }
        })
    
    return {
        'students': students,
        'courses': courses,
        'applications': [],
        'analytics': {
            'total_students': len(students),
            'active_applications': random.randint(15, 25),
            'total_courses': len(courses)
        }
    }

# =============================================================================
# DATABASE MANAGER
# =============================================================================

class DatabaseManager:
    """Enhanced database management"""
    
    def __init__(self, db_path: str = None):
        self.db_path = db_path or config.database_path
        self.init_db()
    
    def get_connection(self):
        """Get database connection"""
        return sqlite3.connect(self.db_path)
    
    def init_db(self):
        """Initialize database with all required tables"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        # Students table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS students (
            id TEXT PRIMARY KEY,
            data TEXT NOT NULL
        )
        ''')
        
        # Courses table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS courses (
            id TEXT PRIMARY KEY,
            data TEXT NOT NULL
        )
        ''')
        
        # Applications table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS applications (
            id TEXT PRIMARY KEY,
            student_id TEXT NOT NULL,
            course_id TEXT NOT NULL,
            data TEXT NOT NULL,
            FOREIGN KEY (student_id) REFERENCES students (id),
            FOREIGN KEY (course_id) REFERENCES courses (id)
        )
        ''')
        
        # Conversations table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS conversations (
            id TEXT PRIMARY KEY,
            student_id TEXT,
            data TEXT NOT NULL,
            FOREIGN KEY (student_id) REFERENCES students (id)
        )
        ''')
        
        # Analytics table for tracking
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS analytics (
            id TEXT PRIMARY KEY,
            event_type TEXT NOT NULL,
            data TEXT NOT NULL,
            timestamp TEXT NOT NULL
        )
        ''')
        
        # Create indexes
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_applications_student ON applications (student_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_applications_course ON applications (course_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_conversations_student ON conversations (student_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_analytics_type ON analytics (event_type)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_analytics_timestamp ON analytics (timestamp)')
        
        conn.commit()
        conn.close()

# =============================================================================
# DATA MANAGER
# =============================================================================

class DataManager:
    """Enhanced data management with robust CSV integration"""
    
    def __init__(self, data_dir: str = None):
        """Initialize data manager with real CSV files"""
        self.data_dir = data_dir or config.data_directory
        self.db_manager = DatabaseManager()
        self.logger = get_logger(f"{__name__}.DataManager")  # Add logger instance
        
        # Initialize empty DataFrames
        self.applications_df = pd.DataFrame()
        self.courses_df = pd.DataFrame()
        self.faqs_df = pd.DataFrame()
        self.counseling_df = pd.DataFrame()
        
        # Initialize ML components only if available
        if SKLEARN_AVAILABLE:
            self.vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
            self.all_text_vectors = None
        else:
            self.vectorizer = None
            self.all_text_vectors = None
        
        self.combined_data = []
        
        # Ensure data directory exists
        if not os.path.exists(self.data_dir):
            self.logger.warning(f"Data directory {self.data_dir} not found. Creating it...")
            try:
                os.makedirs(self.data_dir, exist_ok=True)
            except Exception as e:
                self.logger.error(f"Could not create data directory: {e}")
                self.data_dir = "."  # Fallback to current directory
        
        # Load data from CSV files
        self.load_all_data()
        
        # Create search index if possible
        if SKLEARN_AVAILABLE:
            try:
                self._create_search_index()
            except Exception as e:
                self.logger.warning(f"Could not create search index: {e}")
    
    def load_all_data(self):
        """Load all data from CSV files with robust error handling"""
        try:
            # Load CSV data with improved error handling
            self._load_csv_data_robust()
            self.logger.info("✅ Data loading completed successfully")
            
        except Exception as e:
            self.logger.error(f"Error loading data: {e}")
            # Create minimal sample data if loading fails
            self._create_minimal_sample_data()
    
    def _load_csv_data_robust(self):
        """Load CSV data files with robust error handling and flexible schemas"""
        csv_files = {
            'applications.csv': 'applications_df',
            'courses.csv': 'courses_df', 
            'faqs.csv': 'faqs_df',
            'counseling_slots.csv': 'counseling_df',
            'counseling.csv': 'counseling_df'  # Alternative name
        }
    
        for filename, df_name in csv_files.items():
            csv_path = os.path.join(self.data_dir, filename)
        
            try:
                if os.path.exists(csv_path):
                    # Try to read CSV with different encodings
                    encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
                    df = None
                    
                    for encoding in encodings:
                        try:
                            df = pd.read_csv(csv_path, encoding=encoding)
                            self.logger.info(f"✅ Loaded {filename} with {encoding} encoding")
                            break
                        except UnicodeDecodeError:
                            continue
                        except Exception as e:
                            self.logger.warning(f"Error reading {filename} with {encoding}: {e}")
                            continue
                    
                    if df is None:
                        self.logger.error(f"❌ Could not read {filename} with any encoding")
                        continue
                    
                    # Clean column names
                    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
                    
                    # Store the dataframe
                    setattr(self, df_name, df)
                    self.logger.info(f"✅ Loaded {len(df)} records from {filename}")
                
                    # Special handling for courses.csv
                    if filename == 'courses.csv':
                        self._process_courses_data(df)
                    
                    # Special handling for applications.csv
                    elif filename == 'applications.csv':
                        self._process_applications_data(df)
                    
                    # Special handling for faqs.csv
                    elif filename == 'faqs.csv':
                        self._process_faqs_data(df)
                        
                else:
                    self.logger.warning(f"⚠️ {filename} not found at {csv_path}")
                    setattr(self, df_name, pd.DataFrame())
                
            except Exception as e:
                self.logger.error(f"❌ Error loading {filename}: {e}")
                setattr(self, df_name, pd.DataFrame())
    
    def _process_courses_data(self, df):
        """Process and standardize courses data"""
        try:
            # Ensure required columns exist with fallbacks
            required_columns = {
                'course_name': 'Unknown Course',
                'level': 'undergraduate',
                'description': 'No description available',
                'department': 'General Studies',
                'duration': '1 year',
                'fees_domestic': 9250,
                'fees_international': 15000,
                'min_gpa': 2.5,
                'min_ielts': 6.0,
                'trending_score': 5.0,
                'keywords': '',
                'career_prospects': 'Various career opportunities available'
            }
            
            for col, default_val in required_columns.items():
                if col not in df.columns:
                    df[col] = default_val
                    self.logger.info(f"Added default column '{col}' to courses data")
                else:
                    # Fill missing values
                    df[col] = df[col].fillna(default_val)
            
            # Ensure numeric columns are properly typed
            numeric_columns = ['fees_domestic', 'fees_international', 'min_gpa', 'min_ielts', 'trending_score']
            for col in numeric_columns:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(required_columns[col])
            
            self.courses_df = df
            self.logger.info(f"✅ Processed courses data - {len(df)} courses ready")
            
        except Exception as e:
            self.logger.error(f"Error processing courses data: {e}")
    
    def _process_applications_data(self, df):
        """Process and standardize applications data"""
        try:
            # Ensure required columns exist with fallbacks
            required_columns = {
                'name': 'Unknown Student',
                'applicant_name': 'Unknown Student', 
                'first_name': 'John',
                'last_name': 'Doe',
                'course_applied': 'General Studies',
                'status': 'under_review',
                'gpa': 3.0,
                'ielts_score': 6.5,
                'nationality': 'UK',
                'work_experience_years': 0,
                'application_date': datetime.now().strftime('%Y-%m-%d'),
                'current_education': 'undergraduate'
            }
            
            # Use existing columns or create with defaults
            for col, default_val in required_columns.items():
                if col not in df.columns:
                    # Try to map from similar column names
                    similar_cols = [c for c in df.columns if col.split('_')[0] in c]
                    if similar_cols:
                        df[col] = df[similar_cols[0]]
                        self.logger.info(f"Mapped '{similar_cols[0]}' to '{col}' in applications data")
                    else:
                        df[col] = default_val
                        self.logger.info(f"Added default column '{col}' to applications data")
                else:
                    # Fill missing values
                    df[col] = df[col].fillna(default_val)
            
            # Ensure numeric columns are properly typed
            numeric_columns = ['gpa', 'ielts_score', 'work_experience_years']
            for col in numeric_columns:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(required_columns[col])
            
            self.applications_df = df
            self.logger.info(f"✅ Processed applications data - {len(df)} applications ready")
            
        except Exception as e:
            self.logger.error(f"Error processing applications data: {e}")
    
    def _process_faqs_data(self, df):
        """Process and standardize FAQs data"""
        try:
            # Ensure required columns exist
            if 'question' not in df.columns:
                # Try to find question-like columns
                question_cols = [c for c in df.columns if 'question' in c.lower() or 'q' in c.lower()]
                if question_cols:
                    df['question'] = df[question_cols[0]]
                else:
                    self.logger.warning("No question column found in FAQs data")
                    return
            
            if 'answer' not in df.columns:
                # Try to find answer-like columns
                answer_cols = [c for c in df.columns if 'answer' in c.lower() or 'a' in c.lower()]
                if answer_cols:
                    df['answer'] = df[answer_cols[0]]
                else:
                    self.logger.warning("No answer column found in FAQs data")
                    return
            
            # Remove rows with missing questions or answers
            df = df.dropna(subset=['question', 'answer'])
            
            self.faqs_df = df
            self.logger.info(f"✅ Processed FAQs data - {len(df)} FAQs ready")
            
        except Exception as e:
            self.logger.error(f"Error processing FAQs data: {e}")
    
    def _create_minimal_sample_data(self):
        """Create minimal sample data if CSV loading fails"""
        self.logger.info("Creating minimal sample data as fallback...")
        
        # Sample courses
        self.courses_df = pd.DataFrame([
            {
                'course_name': 'Computer Science BSc',
                'level': 'undergraduate',
                'description': 'Comprehensive computer science program with focus on programming and software development.',
                'department': 'School of Computing',
                'duration': '3 years',
                'fees_domestic': 9250,
                'fees_international': 15000,
                'min_gpa': 3.0,
                'min_ielts': 6.0,
                'trending_score': 8.5,
                'keywords': 'programming, software, algorithms, data structures',
                'career_prospects': 'Software developer, systems analyst, data scientist'
            },
            {
                'course_name': 'Business Management BA',
                'level': 'undergraduate', 
                'description': 'Business administration and management program with practical focus.',
                'department': 'Business School',
                'duration': '3 years',
                'fees_domestic': 9250,
                'fees_international': 14000,
                'min_gpa': 2.8,
                'min_ielts': 6.0,
                'trending_score': 7.0,
                'keywords': 'business, management, leadership, strategy',
                'career_prospects': 'Manager, consultant, entrepreneur'
            },
            {
                'course_name': 'Data Science MSc',
                'level': 'masters',
                'description': 'Advanced data science program with machine learning and analytics.',
                'department': 'School of Computing',
                'duration': '1 year',
                'fees_domestic': 12000,
                'fees_international': 18000,
                'min_gpa': 3.5,
                'min_ielts': 6.5,
                'trending_score': 9.0,
                'keywords': 'data science, machine learning, analytics, python',
                'career_prospects': 'Data scientist, ML engineer, business analyst'
            }
        ])
        
        # Sample applications
        self.applications_df = pd.DataFrame([
            {
                'name': 'John Smith',
                'course_applied': 'Computer Science BSc',
                'status': 'accepted',
                'gpa': 3.8,
                'ielts_score': 7.0,
                'nationality': 'UK',
                'work_experience_years': 2,
                'application_date': '2024-01-15',
                'current_education': 'undergraduate'
            }
        ])
        
        # Sample FAQs
        self.faqs_df = pd.DataFrame([
            {
                'question': 'What are the entry requirements?',
                'answer': 'Entry requirements vary by course. Generally we require IELTS 6.0-6.5 and relevant academic qualifications.'
            },
            {
                'question': 'When is the application deadline?',
                'answer': 'Main deadline is August 1st for September intake. January intake deadline is November 1st.'
            }
        ])
        
        self.logger.info("✅ Minimal sample data created")
    
    def _create_search_index(self):
        """Create search index from loaded data"""
        if not SKLEARN_AVAILABLE:
            self.logger.warning("Scikit-learn not available. Search functionality disabled.")
            return
        
        self.combined_data = []
        
        # Index courses from loaded data
        if not self.courses_df.empty:
            self.logger.info(f"Indexing {len(self.courses_df)} courses for search...")
            
            for _, course in self.courses_df.iterrows():
                # Build searchable text from available columns
                text_parts = []
                
                # Add course name
                if 'course_name' in course and pd.notna(course['course_name']):
                    text_parts.append(str(course['course_name']))
                
                # Add description if available
                if 'description' in course and pd.notna(course['description']):
                    text_parts.append(str(course['description']))
                
                # Add department if available  
                if 'department' in course and pd.notna(course['department']):
                    text_parts.append(str(course['department']))
                
                # Add keywords if available
                if 'keywords' in course and pd.notna(course['keywords']):
                    text_parts.append(str(course['keywords']))
                
                # Add level if available
                if 'level' in course and pd.notna(course['level']):
                    text_parts.append(str(course['level']))
                
                # Combine all text
                search_text = ' '.join(text_parts)
                
                if search_text.strip():  # Only add if we have text
                    self.combined_data.append({
                        'text': search_text,
                        'type': 'course',
                        'data': course.to_dict()
                    })
        
        # Index FAQs from loaded data
        if not self.faqs_df.empty:
            self.logger.info(f"Indexing {len(self.faqs_df)} FAQs for search...")
            
            for _, faq in self.faqs_df.iterrows():
                text_parts = []
                
                # Add question
                if 'question' in faq and pd.notna(faq['question']):
                    text_parts.append(str(faq['question']))
                
                # Add answer  
                if 'answer' in faq and pd.notna(faq['answer']):
                    text_parts.append(str(faq['answer']))
                
                search_text = ' '.join(text_parts)
                
                if search_text.strip():
                    self.combined_data.append({
                        'text': search_text,
                        'type': 'faq',
                        'data': faq.to_dict()
                    })
        
        # Create TF-IDF vectors if we have data
        if self.combined_data:
            try:
                texts = [item['text'] for item in self.combined_data]
                self.all_text_vectors = self.vectorizer.fit_transform(texts)
                self.logger.info(f"✅ Created search index with {len(self.combined_data)} items")
            except Exception as e:
                self.logger.error(f"Error creating TF-IDF vectors: {e}")
                self.all_text_vectors = None
        else:
            self.logger.warning("No data available for search indexing")
    
    def intelligent_search(self, query: str, top_k: int = 5) -> List[Dict]:
        """Intelligent search across all data"""
        if not SKLEARN_AVAILABLE or not self.combined_data or self.all_text_vectors is None:
            return []
        
        try:
            query_vector = self.vectorizer.transform([query])
            similarities = cosine_similarity(query_vector, self.all_text_vectors).flatten()
            top_indices = similarities.argsort()[-top_k:][::-1]
            
            results = []
            for idx in top_indices:
                if similarities[idx] > 0.1:
                    result = self.combined_data[idx].copy()
                    result['similarity'] = similarities[idx]
                    results.append(result)
            
            return results
        except Exception as e:
            self.logger.error(f"Search error: {e}")
            return []
    
    def get_courses_summary(self) -> Dict:
        """Get summary of courses data"""
        if self.courses_df.empty:
            return {"total_courses": 0, "message": "No courses data loaded"}
        
        summary = {
            "total_courses": len(self.courses_df),
            "columns": list(self.courses_df.columns),
            "sample_course": self.courses_df.iloc[0].to_dict() if len(self.courses_df) > 0 else None
        }
        
        # Get course levels if available
        if 'level' in self.courses_df.columns:
            summary["levels"] = self.courses_df['level'].value_counts().to_dict()
        
        # Get departments if available
        if 'department' in self.courses_df.columns:
            summary["departments"] = self.courses_df['department'].value_counts().to_dict()
        
        return summary
    
    def get_applications_summary(self) -> Dict:
        """Get summary of applications data"""
        if self.applications_df.empty:
            return {"total_applications": 0, "message": "No applications data loaded"}
        
        summary = {
            "total_applications": len(self.applications_df),
            "columns": list(self.applications_df.columns),
        }
        
        # Get status breakdown if available
        if 'status' in self.applications_df.columns:
            summary["status_breakdown"] = self.applications_df['status'].value_counts().to_dict()
        
        # Get course breakdown if available
        if 'course_applied' in self.applications_df.columns:
            summary["popular_courses"] = self.applications_df['course_applied'].value_counts().head(5).to_dict()
        
        return summary

    def get_data_stats(self) -> Dict:
        """Get comprehensive data statistics"""
        return {
            'courses': {
                'total': len(self.courses_df) if not self.courses_df.empty else 0,
                'columns': list(self.courses_df.columns) if not self.courses_df.empty else []
            },
            'applications': {
                'total': len(self.applications_df) if not self.applications_df.empty else 0,
                'columns': list(self.applications_df.columns) if not self.applications_df.empty else []
            },
            'faqs': {
                'total': len(self.faqs_df) if not self.faqs_df.empty else 0,
                'columns': list(self.faqs_df.columns) if not self.faqs_df.empty else []
            },
            'search_index': {
                'indexed_items': len(self.combined_data),
                'search_ready': self.all_text_vectors is not None
            }
        }

# =============================================================================
# OLLAMA SERVICE
# =============================================================================

class OllamaService:
    """Enhanced Ollama service with robust error handling and fallbacks"""
    
    def __init__(self, model_name: str = None, base_url: str = None):
        self.model_name = model_name or config.default_model
        self.base_url = base_url or config.ollama_host
        self.api_url = f"{self.base_url}/api/generate"
        self.conversation_history = []
        self.max_history_length = 10
        self.is_available_cached = None
        self.last_check_time = 0
        self.logger = get_logger(f"{__name__}.OllamaService")  # Add logger instance
        
        self.logger.info(f"Initializing Ollama service: {self.base_url} with model {self.model_name}")
        self._check_availability()
    
    def _check_availability(self):
        """Check if Ollama is available with caching"""
        current_time = time.time()
        
        # Cache availability check for 30 seconds
        if self.is_available_cached is not None and (current_time - self.last_check_time) < 30:
            return self.is_available_cached
        
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                available_models = [model['name'] for model in response.json().get('models', [])]
                
                if self.model_name not in available_models:
                    self.logger.warning(f"Model {self.model_name} not found. Available: {available_models}")
                    # Try to use the first available model as fallback
                    if available_models:
                        self.model_name = available_models[0]
                        self.logger.info(f"Using fallback model: {self.model_name}")
                    else:
                        self.logger.error("No models available in Ollama")
                        self.is_available_cached = False
                        self.last_check_time = current_time
                        return False
                else:
                    self.logger.info(f"Successfully connected to Ollama. Using: {self.model_name}")
                
                self.is_available_cached = True
                self.last_check_time = current_time
                return True
            else:
                self.logger.warning(f"Ollama returned status code {response.status_code}")
                self.is_available_cached = False
                self.last_check_time = current_time
                return False
                
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Failed to connect to Ollama: {e}")
            self.is_available_cached = False
            self.last_check_time = current_time
            return False
    
    def is_available(self) -> bool:
        """Check if Ollama service is available"""
        return self._check_availability()
    
    def generate_response(self, prompt: str, system_prompt: Optional[str] = None, 
                          temperature: float = None, max_tokens: int = None) -> str:
        """Generate response using Ollama with robust error handling"""
        try:
            if not self.is_available():
                self.logger.warning("Ollama not available, using fallback response")
                return self._fallback_response(prompt)
            
            # Prepare request data
            data = {
                "model": self.model_name,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": temperature or config.llm_temperature,
                    "num_predict": max_tokens or config.max_tokens
                }
            }
            
            if system_prompt:
                data["system"] = system_prompt
            
            self.logger.info(f"Sending request to Ollama: {prompt[:100]}...")
            response = requests.post(self.api_url, json=data, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                ai_response = result.get('response', 'No response generated')
                
                # Update conversation history
                self.conversation_history.append({"role": "user", "content": prompt})
                self.conversation_history.append({"role": "assistant", "content": ai_response})
                
                if len(self.conversation_history) > self.max_history_length:
                    self.conversation_history = self.conversation_history[-self.max_history_length:]
                
                self.logger.info(f"Successfully received response from Ollama: {len(ai_response)} characters")
                return ai_response
            else:
                self.logger.error(f"Ollama API error: {response.status_code} - {response.text}")
                return self._fallback_response(prompt)
                
        except requests.exceptions.Timeout:
            self.logger.error("Ollama request timed out")
            return self._fallback_response(prompt, error_type="timeout")
        except requests.exceptions.ConnectionError:
            self.logger.error("Connection error to Ollama")
            return self._fallback_response(prompt, error_type="connection")
        except Exception as e:
            self.logger.error(f"LLM generation error: {e}")
            return self._fallback_response(prompt, error_type="general")
    
    def _fallback_response(self, prompt: str, error_type: str = "general") -> str:
        """Provide intelligent fallback response when LLM is unavailable"""
        prompt_lower = prompt.lower()
        
        # Add context about the issue
        if error_type == "timeout":
            prefix = "I'm experiencing high load right now, but I can still help! "
        elif error_type == "connection":
            prefix = "I'm having connectivity issues, but here's what I can tell you: "
        else:
            prefix = "I'm using my fallback knowledge to help you: "
        
        # Course-related queries
        if any(word in prompt_lower for word in ['course', 'program', 'study', 'degree']):
            return f"""{prefix}

🎓 **UEL Course Information**

We offer excellent programs including:
• **Computer Science** - Programming, AI, Software Development
• **Business Management** - Leadership, Strategy, Entrepreneurship  
• **Data Science** - Analytics, Machine Learning, Statistics
• **Engineering** - Civil, Mechanical, Electronic Engineering
• **Psychology** - Clinical, Counseling, Research Psychology

**Key Features:**
✅ Industry-focused curriculum
✅ Experienced faculty
✅ Modern facilities
✅ Strong career support

For detailed course information, visit our website or contact admissions at {config.admissions_email}."""

        # Application-related queries
        elif any(word in prompt_lower for word in ['apply', 'application', 'admission', 'entry', 'requirement']):
            return f"""{prefix}

📝 **UEL Application Process**

**Entry Requirements:**
• Academic qualifications (varies by course)
• English proficiency (IELTS 6.0-6.5)
• Personal statement
• References

**Application Steps:**
1. Choose your course
2. Check entry requirements
3. Submit online application
4. Upload supporting documents
5. Attend interview (if required)

**Deadlines:**
• September intake: August 1st
• January intake: November 1st

**Contact:** {config.admissions_email} | {config.admissions_phone}"""

        # Fee-related queries
        elif any(word in prompt_lower for word in ['fee', 'cost', 'tuition', 'price', 'money', 'scholarship']):
            return f"""{prefix}

💰 **UEL Fees & Financial Support**

**Tuition Fees (Annual):**
• UK Students: £9,250
• International Students: £13,000-£18,000
• Postgraduate: £12,000-£20,000

**Scholarships Available:**
🏆 Merit-based scholarships up to £5,000
🎯 Subject-specific bursaries
🌍 International student discounts
📚 Hardship funds

**Payment Options:**
• Full payment (5% discount)
• Installment plans available
• Student loans accepted

Contact our finance team for personalized advice: {config.admissions_email}"""

        # General greeting
        elif any(word in prompt_lower for word in ['hello', 'hi', 'hey', 'help']):
            return f"""{prefix}

👋 **Welcome to University of East London!**

I'm your AI assistant, here to help with:

🎓 **Course Information** - Programs, requirements, curriculum
📝 **Applications** - Process, deadlines, requirements  
💰 **Fees & Funding** - Tuition, scholarships, payments
🏫 **Campus Life** - Facilities, accommodation, activities
📞 **Contact Info** - Staff, departments, services

**Popular Questions:**
• "What courses do you offer in computer science?"
• "How do I apply for a Master's degree?"
• "What scholarships are available?"
• "What are the entry requirements?"

How can I help you today?"""

        # Default response
        else:
            return f"""{prefix}

Thank you for contacting **University of East London**. I'm here to help with information about:

• 🎓 **Courses & Programs**
• 📝 **Applications & Admissions** 
• 💰 **Fees & Scholarships**
• 🏫 **Campus & Facilities**

**Quick Contact:**
📧 Email: {config.admissions_email}
📞 Phone: {config.admissions_phone}
🌐 Website: uel.ac.uk

Could you please be more specific about what you'd like to know? I'm ready to provide detailed information about any aspect of UEL!"""

# =============================================================================
# ADDITIONAL SERVICES
# =============================================================================

class SentimentAnalysisEngine:
    """Enhanced sentiment analysis"""
    
    def __init__(self):
        self.sentiment_history = []
        self.logger = get_logger(f"{__name__}.SentimentAnalysisEngine")
    
    def analyze_message_sentiment(self, message: str) -> Dict:
        """Analyze sentiment of message"""
        try:
            if TEXTBLOB_AVAILABLE:
                blob = TextBlob(message)
                polarity = blob.sentiment.polarity
                subjectivity = blob.sentiment.subjectivity
            else:
                # Fallback simple sentiment analysis
                polarity, subjectivity = self._simple_sentiment_analysis(message)
            
            sentiment_label = "positive" if polarity > 0.1 else "negative" if polarity < -0.1 else "neutral"
            
            emotions = self._detect_emotions(message)
            urgency = self._detect_urgency(message)
            
            sentiment_data = {
                "polarity": polarity,
                "subjectivity": subjectivity,
                "sentiment": sentiment_label,
                "emotions": emotions,
                "urgency": urgency,
                "timestamp": datetime.now().isoformat()
            }
            
            self.sentiment_history.append(sentiment_data)
            return sentiment_data
            
        except Exception as e:
            self.logger.error(f"Sentiment analysis error: {e}")
            return {"sentiment": "neutral", "polarity": 0.0, "error": str(e)}
    
    def _simple_sentiment_analysis(self, message: str) -> Tuple[float, float]:
        """Simple fallback sentiment analysis"""
        positive_words = ['good', 'great', 'excellent', 'wonderful', 'amazing', 'happy', 'pleased', 'excited']
        negative_words = ['bad', 'terrible', 'awful', 'horrible', 'sad', 'angry', 'frustrated', 'disappointed']
        
        message_lower = message.lower()
        positive_count = sum(1 for word in positive_words if word in message_lower)
        negative_count = sum(1 for word in negative_words if word in message_lower)
        
        total_words = len(message.split())
        if total_words == 0:
            return 0.0, 0.0
        
        polarity = (positive_count - negative_count) / total_words
        subjectivity = (positive_count + negative_count) / total_words
        
        return polarity, subjectivity
    
    def _detect_emotions(self, message: str) -> List[str]:
        """Detect emotions in message"""
        emotion_keywords = {
            'anxiety': ['worried', 'anxious', 'nervous', 'stressed', 'concerned', 'fear'],
            'excitement': ['excited', 'thrilled', 'eager', 'enthusiastic', 'happy'],
            'frustration': ['frustrated', 'annoyed', 'irritated', 'upset', 'angry'],
            'confusion': ['confused', 'unclear', "don't understand", 'puzzled', 'lost'],
            'satisfaction': ['satisfied', 'pleased', 'glad', 'thankful', 'grateful']
        }
        
        detected_emotions = []
        message_lower = message.lower()
        
        for emotion, keywords in emotion_keywords.items():
            if any(keyword in message_lower for keyword in keywords):
                detected_emotions.append(emotion)
        
        return detected_emotions
    
    def _detect_urgency(self, message: str) -> str:
        """Detect urgency level"""
        urgent_indicators = ['urgent', 'asap', 'immediately', 'emergency', 'deadline', 'hurry']
        high_indicators = ['soon', 'quickly', 'fast', 'rush', 'quick']
        
        message_lower = message.lower()
        
        if any(indicator in message_lower for indicator in urgent_indicators):
            return "urgent"
        elif any(indicator in message_lower for indicator in high_indicators):
            return "high"
        else:
            return "normal"

class DocumentVerificationAI:
    """AI-powered document verification with robust error handling"""
    
    def __init__(self):
        self.logger = get_logger(f"{__name__}.DocumentVerificationAI")
        try:
            self.verification_rules = self._load_verification_rules()
            self.verification_history = []
            self.logger.info("DocumentVerificationAI initialized successfully")
        except Exception as e:
            self.logger.error(f"DocumentVerificationAI initialization error: {e}")
            # Fallback initialization
            self.verification_rules = self._get_default_verification_rules()
            self.verification_history = []
    
    def _load_verification_rules(self) -> Dict:
        """Load document verification rules"""
        return self._get_default_verification_rules()
    
    def _get_default_verification_rules(self) -> Dict:
        """Get default document verification rules"""
        return {
            'transcript': {
                'required_fields': ['institution_name', 'student_name', 'grades', 'graduation_date'],
                'format_requirements': ['pdf_format', 'official_seal'],
                'validation_checks': ['grade_consistency', 'date_validity']
            },
            'ielts_certificate': {
                'required_fields': ['test_taker_name', 'test_date', 'scores', 'test_center'],
                'format_requirements': ['official_format', 'security_features'],
                'validation_checks': ['score_validity', 'date_recency']
            },
            'passport': {
                'required_fields': ['full_name', 'nationality', 'passport_number', 'expiry_date'],
                'format_requirements': ['clear_image', 'readable_text'],
                'validation_checks': ['expiry_check', 'format_validation']
            },
            'personal_statement': {
                'required_fields': ['content', 'word_count', 'format'],
                'format_requirements': ['pdf_or_doc_format', 'readable_text'],
                'validation_checks': ['word_count_check', 'content_relevance']
            },
            'reference_letter': {
                'required_fields': ['referee_name', 'referee_position', 'institution', 'content'],
                'format_requirements': ['official_letterhead', 'signature'],
                'validation_checks': ['authenticity_check', 'contact_verification']
            }
        }
    
    def verify_document(self, document_data: Dict, document_type: str) -> Dict:
        """Verify document using AI analysis"""
        try:
            # Simulate document verification process
            confidence_score = 0.85 + (hash(str(document_data)) % 100) / 1000
            confidence_score = min(max(confidence_score, 0.5), 1.0)
            
            # Get verification rules for this document type
            rules = self.verification_rules.get(document_type.lower(), {})
            required_fields = rules.get('required_fields', [])
            
            # Check for issues
            issues_found = []
            verified_fields = self._extract_verified_fields(document_data, document_type)
            
            # Check missing required fields
            for field in required_fields:
                if field not in document_data or not document_data[field]:
                    issues_found.append(f"Missing required field: {field}")
                    confidence_score -= 0.1
            
            # Adjust confidence based on issues
            confidence_score = max(confidence_score, 0.3)
            
            # Determine verification status
            if confidence_score > 0.8 and not issues_found:
                status = "verified"
            elif confidence_score > 0.6:
                status = "needs_review"
            else:
                status = "rejected"
            
            # Generate recommendations
            recommendations = self._generate_recommendations(document_type, issues_found, confidence_score)
            
            verification_result = {
                "document_type": document_type,
                "verification_status": status,
                "confidence_score": confidence_score,
                "issues_found": issues_found,
                "recommendations": recommendations,
                "verified_fields": verified_fields,
                "timestamp": datetime.now().isoformat(),
                "document_id": self._generate_document_id()
            }
            
            # Store in history
            self.verification_history.append(verification_result)
            
            return verification_result
            
        except Exception as e:
            self.logger.error(f"Document verification error: {e}")
            return {
                "verification_status": "error",
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
                "confidence_score": 0.0
            }
    
    def _extract_verified_fields(self, document_data: Dict, document_type: str) -> Dict:
        """Extract and verify document fields"""
        verified_fields = {}
        
        rules = self.verification_rules.get(document_type.lower(), {})
        required_fields = rules.get('required_fields', [])
        
        for field in required_fields:
            if field in document_data:
                # Simulate field verification
                field_confidence = 0.9 if document_data[field] else 0.0
                verified_fields[field] = {
                    "value": document_data[field],
                    "verified": bool(document_data[field]),
                    "confidence": field_confidence
                }
            else:
                verified_fields[field] = {
                    "value": None,
                    "verified": False,
                    "confidence": 0.0
                }
        
        return verified_fields
    
    def _generate_recommendations(self, document_type: str, issues: List[str], confidence: float) -> List[str]:
        """Generate verification recommendations"""
        recommendations = []
        
        if issues:
            recommendations.append("Please address the identified issues and resubmit")
            for issue in issues:
                if "Missing" in issue:
                    recommendations.append(f"Provide the missing information: {issue.split(': ')[1]}")
        
        if confidence < 0.7:
            recommendations.append("Consider providing additional supporting documentation")
            recommendations.append("Ensure all text is clearly legible in scanned documents")
        
        if document_type.lower() == 'transcript':
            recommendations.append("Ensure transcript includes official institution seal and signature")
        elif document_type.lower() == 'ielts_certificate':
            recommendations.append("Verify IELTS certificate is less than 2 years old")
        elif document_type.lower() == 'passport':
            recommendations.append("Ensure passport is valid for at least 6 months")
        
        if confidence > 0.8:
            recommendations = ["Document appears valid and complete"]
        
        return recommendations
    
    def _generate_document_id(self) -> str:
        """Generate unique document ID"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        random_suffix = hashlib.md5(str(time.time()).encode()).hexdigest()[:6]
        return f"DOC_{timestamp}_{random_suffix}"

class VoiceService:
    """Enhanced voice recognition and text-to-speech service"""
    
    def __init__(self):
        self.is_listening = False
        self.is_speaking = False
        self.recognizer = None
        self.engine = None
        self.microphone = None
        self.logger = get_logger(f"{__name__}.VoiceService")
        
        if VOICE_AVAILABLE:
            try:
                # Initialize speech recognition
                self.recognizer = sr.Recognizer()
                self.microphone = sr.Microphone()
                
                # Test microphone access
                with self.microphone as source:
                    self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
                
                # Initialize text-to-speech
                self.engine = pyttsx3.init()
                self.engine.setProperty('rate', 150)
                self.engine.setProperty('volume', 0.8)
                
                # Set voice (try to use a pleasant voice)
                voices = self.engine.getProperty('voices')
                if voices:
                    # Prefer female voice if available
                    for voice in voices:
                        if 'female' in voice.name.lower() or 'zira' in voice.name.lower():
                            self.engine.setProperty('voice', voice.id)
                            break
                
                self.logger.info("Voice service initialized successfully")
                
            except Exception as e:
                self.logger.error(f"Voice service initialization error: {e}")
                self.recognizer = None
                self.engine = None
                self.microphone = None
    
    def is_available(self) -> bool:
        """Check if voice service is available"""
        return (VOICE_AVAILABLE and 
                self.recognizer is not None and 
                self.engine is not None and 
                self.microphone is not None)
    
    def speech_to_text(self) -> str:
        """Convert speech to text with better error handling"""
        if not self.is_available():
            return "❌ Voice service not available. Please install: pip install SpeechRecognition pyttsx3 pyaudio"
        
        try:
            with self.microphone as source:
                self.logger.info("Adjusting for ambient noise...")
                self.recognizer.adjust_for_ambient_noise(source, duration=1)
                
                self.logger.info("Listening for speech...")
                # Increase timeout and phrase time limit
                audio = self.recognizer.listen(source, timeout=3, phrase_time_limit=10)
            
            self.logger.info("Processing speech...")
            
            # Try Google Speech Recognition first
            try:
                text = self.recognizer.recognize_google(audio)
                self.logger.info(f"Speech recognized: {text}")
                return text
            except sr.RequestError:
                # Fallback to offline recognition if available
                try:
                    text = self.recognizer.recognize_sphinx(audio)
                    self.logger.info(f"Speech recognized (offline): {text}")
                    return text
                except:
                    return "❌ Speech recognition service unavailable. Please check internet connection."
        
        except sr.WaitTimeoutError:
            return "⏰ No speech detected within 10 seconds. Please try again."
        except sr.UnknownValueError:
            return "❌ Could not understand speech. Please speak more clearly and try again."
        except sr.RequestError as e:
            return f"❌ Speech recognition request failed: {e}"
        except OSError as e:
            if "No Default Input Device Available" in str(e):
                return "❌ No microphone detected. Please connect a microphone and try again."
            else:
                return f"❌ Microphone error: {e}"
        except Exception as e:
            self.logger.error(f"Speech recognition error: {e}")
            return f"❌ Voice input failed: {str(e)}"
    
    def text_to_speech(self, text: str) -> bool:
        """Convert text to speech with better error handling"""
        if not self.is_available():
            self.logger.warning("TTS not available")
            return False
        
        if self.is_speaking:
            self.logger.warning("Already speaking")
            return False
        
        try:
            # Clean text for better speech
            clean_text = self._clean_text_for_speech(text)
            
            self.is_speaking = True
            
            def speak_thread():
                try:
                    self.engine.say(clean_text)
                    self.engine.runAndWait()
                except Exception as e:
                    self.logger.error(f"TTS thread error: {e}")
                finally:
                    self.is_speaking = False
            
            # Start speaking in background thread
            thread = threading.Thread(target=speak_thread, daemon=True)
            thread.start()
            
            return True
            
        except Exception as e:
            self.is_speaking = False
            self.logger.error(f"Text-to-speech error: {e}")
            return False
    
    def _clean_text_for_speech(self, text: str) -> str:
        """Clean text for better speech synthesis"""
        import re
        
        # Remove emojis and special characters
        clean_text = re.sub(r'[🎓📝💰📞📧🌐📋📄⏳🎉❌⏸️💡✅⚠️🔍📅🤔👋💳🚀🎯🎤🔊]', '', text)
        
        # Remove markdown formatting
        clean_text = re.sub(r'\*\*([^*]+)\*\*', r'\1', clean_text)  # Bold
        clean_text = re.sub(r'\*([^*]+)\*', r'\1', clean_text)      # Italic
        clean_text = re.sub(r'#{1,6}\s*', '', clean_text)           # Headers
        clean_text = re.sub(r'`([^`]+)`', r'\1', clean_text)        # Code
        
        # Replace common abbreviations for better pronunciation
        replacements = {
            'UEL': 'University of East London',
            'AI': 'A I',
            'ML': 'Machine Learning',
            'UK': 'United Kingdom',
            'USA': 'United States of America',
            'IELTS': 'I E L T S',
            'GPA': 'G P A',
            'MSc': 'Master of Science',
            'BSc': 'Bachelor of Science',
            'MBA': 'Master of Business Administration'
        }
        
        for abbr, full_form in replacements.items():
            clean_text = clean_text.replace(abbr, full_form)
        
        # Remove extra whitespace
        clean_text = ' '.join(clean_text.split())
        
        return clean_text

# =============================================================================
# COURSE RECOMMENDATION SYSTEM
# =============================================================================

class CourseRecommendationSystem:
    """AI-powered course recommendation system using actual CSV data"""
    
    def __init__(self, data_manager: DataManager):
        self.data_manager = data_manager
        self.logger = get_logger(f"{__name__}.CourseRecommendationSystem")
        
        if SKLEARN_AVAILABLE:
            self.vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
            self.course_vectors = None
        else:
            self.vectorizer = None
            self.course_vectors = None
        
        self.courses_df = None
        self.course_keywords = {}
        self._load_and_prepare_data()
    
    def _load_and_prepare_data(self):
        """Load courses from CSV and prepare for recommendations"""
        try:
            # Use the data manager's loaded courses
            self.courses_df = self.data_manager.courses_df.copy()
            
            if self.courses_df.empty:
                self.logger.error("No courses data available for recommendations")
                return
            
            self.logger.info(f"✅ Loaded {len(self.courses_df)} courses for recommendations")
            
            # Prepare search vectors if sklearn available
            if SKLEARN_AVAILABLE and len(self.courses_df) > 0:
                self._create_search_vectors()
            
        except Exception as e:
            self.logger.error(f"❌ Error loading courses for recommendations: {e}")
            self.courses_df = pd.DataFrame()
    
    def _create_search_vectors(self):
        """Create TF-IDF vectors for course matching"""
        try:
            if self.courses_df.empty:
                self.logger.warning("No courses data to create vectors from")
                return
            
            # Combine text fields for each course
            course_texts = []
            for idx, course in self.courses_df.iterrows():
                text_parts = [
                    str(course.get('course_name', '')),
                    str(course.get('description', '')),
                    str(course.get('keywords', '')),
                    str(course.get('department', '')),
                    str(course.get('career_prospects', ''))
                ]
                
                combined_text = ' '.join(text_parts).lower()
                course_texts.append(combined_text)
            
            # Create TF-IDF vectors
            if course_texts:
                self.course_vectors = self.vectorizer.fit_transform(course_texts)
                self.logger.info(f"✅ Created search vectors for {len(course_texts)} courses")
            
        except Exception as e:
            self.logger.error(f"Error creating search vectors: {e}")
            self.course_vectors = None
    
    def recommend_courses(self, user_profile: Dict, preferences: Dict = None) -> List[Dict]:
        """Generate course recommendations based on user profile"""
        try:
            if self.courses_df.empty:
                self.logger.error("No courses data available for recommendations")
                return []
            
            self.logger.info(f"\n🤖 GENERATING COURSE RECOMMENDATIONS")
            self.logger.info(f"📊 Total courses in database: {len(self.courses_df)}")
            self.logger.info(f"👤 User profile: {user_profile.get('first_name', 'Unknown')}")
            self.logger.info(f"🎯 Primary interest: {user_profile.get('field_of_interest', 'Not specified')}")
            
            recommendations = []
            
            # Score each course
            for idx, course in self.courses_df.iterrows():
                try:
                    # Calculate different score components
                    interest_result = self._calculate_interest_match_score(course, user_profile)
                    academic_score = self._calculate_academic_progression_score(course, user_profile)
                    requirements_score = self._calculate_academic_requirements_score(course, user_profile)
                    trending_score = self._calculate_trending_score(course)
                    
                    # Weighted overall score
                    overall_score = (
                        interest_result['total_score'] * 0.50 +  # 50% interest match
                        academic_score * 0.25 +                 # 25% academic progression
                        requirements_score * 0.15 +             # 15% requirements met
                        trending_score * 0.10                   # 10% trending/demand
                    )
                    
                    # Generate reasons
                    reasons = self._generate_recommendation_reasons(
                        course, interest_result, academic_score, requirements_score
                    )
                    
                    # Determine match quality
                    if overall_score >= 0.8:
                        match_quality = "🎯 Perfect Match"
                        match_color = "#10b981"
                    elif overall_score >= 0.6:
                        match_quality = "⭐ Excellent Match"
                        match_color = "#3b82f6"
                    elif overall_score >= 0.4:
                        match_quality = "👍 Good Match"
                        match_color = "#f59e0b"
                    else:
                        match_quality = "✅ Suitable"
                        match_color = "#6b7280"
                    
                    # Create recommendation object
                    recommendation = {
                        "course_id": f"UEL_{idx+1}",
                        "course_name": str(course.get('course_name', 'Unknown Course')),
                        "level": str(course.get('level', 'Unknown')).title(),
                        "duration": str(course.get('duration', 'Not specified')),
                        "department": str(course.get('department', 'General Studies')),
                        "description": str(course.get('description', 'No description available')),
                        "keywords": str(course.get('keywords', '')),
                        "career_prospects": str(course.get('career_prospects', 'Various opportunities')),
                        
                        # Scores
                        "score": overall_score,
                        "academic_fit": academic_score,
                        "interest_fit": interest_result['total_score'],
                        "requirements_fit": requirements_score,
                        "trending_score": float(course.get('trending_score', 5.0)),
                        
                        # Match info
                        "match_quality": match_quality,
                        "match_color": match_color,
                        "matched_keywords": interest_result['matched_keywords'],
                        "reasons": reasons,
                        
                        # Requirements
                        "fees": f"£{int(course.get('fees_international', 15000)):,} per year",
                        "min_gpa": float(course.get('min_gpa', 2.5)),
                        "min_ielts": float(course.get('min_ielts', 6.0))
                    }
                    
                    recommendations.append(recommendation)
                    
                    self.logger.info(f"📚 {course.get('course_name', 'Unknown')}: {overall_score:.2f} "
                              f"(Interest: {interest_result['total_score']:.2f}, "
                              f"Academic: {academic_score:.2f}, "
                              f"Requirements: {requirements_score:.2f})")
                    
                except Exception as e:
                    self.logger.error(f"Error scoring course {course.get('course_name', 'Unknown')}: {e}")
                    continue
            
            # Sort by overall score
            recommendations.sort(key=lambda x: x['score'], reverse=True)
            
            # Return top recommendations
            top_recommendations = recommendations[:10]  # Top 10
            
            self.logger.info(f"✅ Generated {len(top_recommendations)} recommendations")
            
            return top_recommendations
            
        except Exception as e:
            self.logger.error(f"❌ Error in course recommendation: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return []
    
    def _calculate_interest_match_score(self, course_row, user_profile: Dict) -> Dict:
        """Calculate how well a course matches user interests"""
        result = {
            'total_score': 0.0,
            'matches': [],
            'matched_keywords': []
        }
        
        # Get user interests
        user_interests = []
        
        # Primary interest (highest weight)
        primary_interest = user_profile.get('field_of_interest', '').lower()
        if primary_interest:
            user_interests.append((primary_interest, 1.0))  # Weight 1.0
        
        # Secondary interests (medium weight)
        secondary = user_profile.get('interests', [])
        if isinstance(secondary, list):
            for interest in secondary:
                if interest and interest.strip():
                    user_interests.append((interest.lower().strip(), 0.7))  # Weight 0.7
        
        # Career goals (medium weight)
        career_goals = user_profile.get('career_goals', '').lower()
        if career_goals:
            # Extract meaningful words from career goals
            career_words = [word for word in career_goals.split() if len(word) > 3]
            for word in career_words[:5]:  # Limit to top 5 words
                user_interests.append((word, 0.5))  # Weight 0.5
        
        # Get course content for matching
        course_name = str(course_row.get('course_name', '')).lower()
        course_description = str(course_row.get('description', '')).lower()
        course_keywords = str(course_row.get('keywords', '')).lower()
        course_department = str(course_row.get('department', '')).lower()
        course_prospects = str(course_row.get('career_prospects', '')).lower()
        
        # Combine all course text
        course_text = f"{course_name} {course_description} {course_keywords} {course_department} {course_prospects}"
        
        total_score = 0.0
        max_possible_score = sum(weight for _, weight in user_interests)
        
        # Check each user interest against course content
        for interest, weight in user_interests:
            if not interest:
                continue
            
            interest_words = interest.split()
            match_found = False
            
            # Direct phrase match in course name (highest priority)
            if interest in course_name:
                score_boost = weight * 1.0
                total_score += score_boost
                result['matches'].append(f"Direct match: '{interest}' in course name")
                result['matched_keywords'].append(interest)
                match_found = True
            
            # Direct phrase match in keywords (high priority)
            elif interest in course_keywords:
                score_boost = weight * 0.9
                total_score += score_boost
                result['matches'].append(f"Keyword match: '{interest}' in course keywords")
                result['matched_keywords'].append(interest)
                match_found = True
            
            # Direct phrase match in description (medium priority)
            elif interest in course_description:
                score_boost = weight * 0.7
                total_score += score_boost
                result['matches'].append(f"Description match: '{interest}' found")
                result['matched_keywords'].append(interest)
                match_found = True
            
            # Word-level matching if no phrase match found
            elif not match_found:
                word_matches = 0
                for word in interest_words:
                    if len(word) > 3 and word in course_text:
                        word_matches += 1
                
                if word_matches > 0:
                    # Score based on percentage of words matched
                    word_match_ratio = word_matches / len(interest_words)
                    score_boost = weight * 0.5 * word_match_ratio
                    total_score += score_boost
                    result['matches'].append(f"Partial match: {word_matches}/{len(interest_words)} words from '{interest}'")
                    result['matched_keywords'].append(interest)
        
        # Normalize score (0-1 range)
        if max_possible_score > 0:
            result['total_score'] = min(total_score / max_possible_score, 1.0)
        else:
            result['total_score'] = 0.0
        
        return result
    
    def _calculate_academic_progression_score(self, course_row, user_profile: Dict) -> float:
        """Calculate academic progression compatibility"""
        user_level = user_profile.get('academic_level', 'undergraduate').lower()
        course_level = str(course_row.get('level', 'undergraduate')).lower()
        
        # Define progression logic
        progression_scores = {
            # From high school
            ('high school', 'undergraduate'): 1.0,
            ('high school', 'foundation'): 1.0,
            
            # From undergraduate
            ('undergraduate', 'undergraduate'): 0.3,  # Second degree
            ('undergraduate', 'postgraduate'): 1.0,   # Normal progression
            ('undergraduate', 'masters'): 1.0,        # Normal progression
            
            # From graduate/postgraduate
            ('graduate', 'postgraduate'): 0.7,
            ('graduate', 'masters'): 1.0,
            ('graduate', 'phd'): 0.8,
            
            # From postgraduate/masters
            ('postgraduate', 'masters'): 0.5,         # Lateral move
            ('postgraduate', 'phd'): 1.0,             # Normal progression
            ('masters', 'phd'): 1.0,                  # Normal progression
            ('masters', 'masters'): 0.4,              # Second masters
        }
        
        # Check for exact match
        key = (user_level, course_level)
        if key in progression_scores:
            return progression_scores[key]
        
        # Default logic for unmatched combinations
        if 'undergraduate' in user_level and 'masters' in course_level:
            return 1.0
        elif 'masters' in user_level and 'phd' in course_level:
            return 1.0
        elif user_level == course_level:
            return 0.4  # Same level
        else:
            return 0.6  # Default moderate score
    
    def _calculate_academic_requirements_score(self, course_row, user_profile: Dict) -> float:
        """Calculate if user meets academic requirements"""
        user_gpa = user_profile.get('gpa', 3.0)
        user_ielts = user_profile.get('ielts_score', 6.5)
        
        course_min_gpa = float(course_row.get('min_gpa', 2.5))
        course_min_ielts = float(course_row.get('min_ielts', 6.0))
        
        # Calculate scores
        gpa_score = 1.0 if user_gpa >= course_min_gpa else max(0.0, user_gpa / course_min_gpa)
        ielts_score = 1.0 if user_ielts >= course_min_ielts else max(0.0, user_ielts / course_min_ielts)
        
        # Combined score (both requirements must be reasonably met)
        combined_score = (gpa_score * 0.6) + (ielts_score * 0.4)
        
        return combined_score
    
    def _calculate_trending_score(self, course_row) -> float:
        """Calculate course trending/demand score"""
        trending_score = float(course_row.get('trending_score', 5.0))
        return min(trending_score / 10.0, 1.0)  # Normalize to 0-1
    
    def _generate_recommendation_reasons(self, course_row, interest_result: Dict, 
                                       academic_score: float, requirements_score: float) -> List[str]:
        """Generate specific reasons why this course is recommended"""
        reasons = []
        
        # Interest-based reasons
        for match in interest_result['matches'][:3]:  # Top 3 matches
            if 'Direct match' in match:
                reasons.append(f"🎯 {match}")
            elif 'Keyword match' in match:
                reasons.append(f"🔑 {match}")
            elif 'Description match' in match:
                reasons.append(f"📝 {match}")
            else:
                reasons.append(f"🔗 {match}")
        
        # Academic progression reasons
        if academic_score >= 0.9:
            reasons.append("🎓 Perfect academic progression for your current level")
        elif academic_score >= 0.7:
            reasons.append("📚 Good academic fit for your background")
        
        # Requirements reasons
        if requirements_score >= 0.9:
            reasons.append("✅ You exceed the academic requirements")
        elif requirements_score >= 0.7:
            reasons.append("👍 You meet the academic requirements")
        elif requirements_score < 0.7:
            reasons.append("📈 Consider improving grades/English score for better chances")
        
        # Trending/demand reasons
        trending_score = float(course_row.get('trending_score', 5.0))
        if trending_score >= 8.0:
            reasons.append("🔥 High industry demand - excellent career prospects")
        elif trending_score >= 7.0:
            reasons.append("📈 Growing field with good job opportunities")
        
        # Department/university strength
        department = course_row.get('department', '')
        if department and department != 'General Studies':
            reasons.append(f"🏛️ Strong reputation in {department}")
        
        # Default reason if no specific reasons found
        if not reasons:
            reasons.append("✅ Suitable course based on your profile")
        
        return reasons[:5]  # Limit to top 5 reasons

# =============================================================================
# PREDICTIVE ANALYTICS ENGINE
# =============================================================================

class PredictiveAnalyticsEngine:
    """Enhanced ML-based predictive analytics with robust error handling"""
    
    def __init__(self, data_manager: DataManager):
        self.data_manager = data_manager
        self.admission_predictor = None
        self.success_probability_model = None
        self.models_trained = False
        self.logger = get_logger(f"{__name__}.PredictiveAnalyticsEngine")
        
        if SKLEARN_AVAILABLE:
            # Enhanced feature set
            self.feature_names = [
                'gpa', 'ielts_score', 'work_experience_years', 
                'course_difficulty', 'application_timing', 'international_status',
                'education_level_score', 'education_compatibility', 'gpa_percentile',
                'ielts_percentile', 'overall_academic_strength'
            ]
            self.logger.info("Predictive Analytics Engine initialized. Starting model training...")
            self.train_models()
        else:
            self.logger.warning("Scikit-learn not available. Predictive analytics disabled.")
            self.models_trained = False

    def train_models(self):
        """Train ML models with enhanced error handling"""
        if not SKLEARN_AVAILABLE:
            self.logger.warning("Scikit-learn not available. ML predictions disabled.")
            self.models_trained = False
            return
        
        try:
            self.logger.info("Starting ML model training...")
            
            # Get applications data
            applications = self._get_training_data()
            
            if not applications:
                self.logger.error("No training data available")
                self._set_fallback_models()
                return
            
            if len(applications) < 5:  # Reduced minimum requirement
                self.logger.warning(f"Limited training data ({len(applications)} samples). Generating additional synthetic data...")
                additional_data = self._generate_synthetic_data(20)  # Generate 20 more samples
                applications.extend(additional_data)
            
            self.logger.info(f"Training with {len(applications)} applications")
            
            # Prepare training data
            features, targets = self._prepare_training_data(applications)
            
            if len(features) < 2:
                self.logger.error("Insufficient valid training samples after preprocessing")
                self._set_fallback_models()
                return
            
            self.logger.info(f"Prepared {len(features)} valid training samples with {len(features[0])} features")
            
            # Train Random Forest Classifier for admission prediction
            self.admission_predictor = RandomForestClassifier(
                n_estimators=50,  # Reduced for faster training with small data
                random_state=42,
                max_depth=10,
                min_samples_split=2,  # Reduced for small datasets
                min_samples_leaf=1,
                class_weight='balanced'
            )
            
            self.admission_predictor.fit(features, targets)
            self.logger.info("✅ Admission predictor trained successfully")
            
            # Train Gradient Boosting Regressor for success probability
            self.success_probability_model = GradientBoostingRegressor(
                n_estimators=50,  # Reduced for faster training
                learning_rate=0.1,
                max_depth=4,  # Reduced for small datasets
                random_state=42,
                subsample=0.8
            )
            
            # Convert targets to float for regression
            regression_targets = targets.astype(float)
            self.success_probability_model.fit(features, regression_targets)
            self.logger.info("✅ Success probability model trained successfully")
            
            # Validate models
            train_accuracy = self.admission_predictor.score(features, targets)
            train_r2 = self.success_probability_model.score(features, regression_targets)
            
            self.logger.info(f"📊 Training accuracy: {train_accuracy:.3f}")
            self.logger.info(f"📊 Training R²: {train_r2:.3f}")
            
            self.models_trained = True
            self.logger.info(f"🎉 ML models successfully trained with {len(features)} samples!")
            
        except Exception as e:
            self.logger.error(f"❌ Model training failed: {e}")
            import traceback
            self.logger.error(f"Full traceback: {traceback.format_exc()}")
            self._set_fallback_models()

    def _get_training_data(self) -> List[Dict]:
        """Get training data from database or generate synthetic data"""
        try:
            # Try to get real data from data manager first
            applications = self._load_real_applications_data()
            
            if len(applications) < 10:
                self.logger.info("Insufficient real data, generating synthetic training data...")
                # Generate synthetic data if not enough real data
                synthetic_data = self._generate_synthetic_data(100)  # Generate 100 samples
                applications.extend(synthetic_data)
            
            self.logger.info(f"Training data loaded: {len(applications)} applications")
            return applications
            
        except Exception as e:
            self.logger.error(f"Error loading training data: {e}")
            # Fallback to synthetic data
            return self._generate_synthetic_data(50)

    def _load_real_applications_data(self) -> List[Dict]:
        """Load real applications data from data manager"""
        try:
            # Try to load from data manager's applications DataFrame
            if hasattr(self.data_manager, 'applications_df') and not self.data_manager.applications_df.empty:
                applications = []
                for _, row in self.data_manager.applications_df.iterrows():
                    app_data = {
                        'gpa': float(row.get('gpa', 3.0)),
                        'ielts_score': float(row.get('ielts_score', 6.5)),
                        'work_experience_years': int(row.get('work_experience_years', 0)),
                        'course_applied': str(row.get('course_applied', 'General Studies')),
                        'nationality': str(row.get('nationality', 'UK')),
                        'status': str(row.get('status', 'under_review')),
                        'current_education': str(row.get('current_education', 'undergraduate')),
                        'application_date': str(row.get('application_date', datetime.now().strftime('%Y-%m-%d')))
                    }
                    applications.append(app_data)
                return applications
            
            return []  # Return empty list if no real data
            
        except Exception as e:
            self.logger.error(f"Error loading real applications data: {e}")
            return []

    def _generate_synthetic_data(self, count: int) -> List[Dict]:
        """Generate enhanced synthetic training data"""
        import random
        
        synthetic_data = []
        
        # Define realistic data distributions
        education_levels = [
            ('high school', 0.15),
            ('undergraduate', 0.35), 
            ('bachelor', 0.25),
            ('masters', 0.20),
            ('phd', 0.05)
        ]
        
        courses = [
            ('Computer Science BSc', 'undergraduate'),
            ('Computer Science MSc', 'masters'),
            ('Business Management BA', 'undergraduate'),
            ('Business Management MBA', 'masters'),
            ('Engineering BEng', 'undergraduate'),
            ('Engineering MEng', 'masters'),
            ('Data Science MSc', 'masters'),
            ('Psychology BSc', 'undergraduate'),
            ('Law LLB', 'undergraduate'),
            ('Medicine MBBS', 'undergraduate')
        ]
        
        nationalities = ['UK', 'India', 'China', 'Nigeria', 'Pakistan', 'US', 'Canada']
        
        for i in range(count):
            # Select education level based on distribution
            education_level = random.choices(
                [level for level, _ in education_levels],
                weights=[weight for _, weight in education_levels]
            )[0]
            
            # Select appropriate course
            suitable_courses = [course for course, level in courses 
                              if (education_level in ['masters', 'phd'] and level == 'masters') or
                                 (education_level in ['high school', 'undergraduate', 'bachelor'] and level == 'undergraduate')]
            
            if not suitable_courses:
                suitable_courses = [course for course, _ in courses]
            
            course_applied = random.choice(suitable_courses)
            
            # Generate realistic GPA based on education level
            if education_level in ['masters', 'phd']:
                gpa = random.uniform(3.2, 4.0)  # Higher GPA for graduate students
            elif education_level == 'bachelor':
                gpa = random.uniform(2.8, 4.0)
            else:
                gpa = random.uniform(2.5, 4.0)
            
            # Generate IELTS score
            ielts_score = random.uniform(5.5, 9.0)
            
            # Generate work experience
            if education_level in ['masters', 'phd']:
                work_exp = random.randint(0, 8)
            else:
                work_exp = random.randint(0, 4)
            
            # Generate realistic success based on multiple factors
            success_factors = []
            
            # GPA factor
            if gpa >= 3.7:
                success_factors.append(0.3)
            elif gpa >= 3.3:
                success_factors.append(0.2)
            elif gpa >= 2.8:
                success_factors.append(0.1)
            else:
                success_factors.append(-0.1)
            
            # IELTS factor
            if ielts_score >= 7.5:
                success_factors.append(0.2)
            elif ielts_score >= 6.5:
                success_factors.append(0.1)
            elif ielts_score >= 6.0:
                success_factors.append(0.0)
            else:
                success_factors.append(-0.2)
            
            # Work experience factor
            if work_exp >= 3:
                success_factors.append(0.15)
            elif work_exp >= 1:
                success_factors.append(0.05)
            
            # Calculate final success probability
            base_probability = 0.5
            final_probability = base_probability + sum(success_factors)
            final_probability = max(0.1, min(0.95, final_probability))  # Clamp between 0.1 and 0.95
            
            # Determine status based on probability
            status = "accepted" if random.random() < final_probability else "rejected"
            
            synthetic_data.append({
                'gpa': round(gpa, 2),
                'ielts_score': round(ielts_score, 1),
                'work_experience_years': work_exp,
                'nationality': random.choice(nationalities),
                'status': status,
                'course_applied': course_applied,
                'current_education': education_level,
                'application_date': (datetime.now() - timedelta(days=random.randint(1, 365))).strftime('%Y-%m-%d')
            })
        
        return synthetic_data

    def _prepare_training_data(self, applications: List[Dict]) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare enhanced features and targets for training"""
        features = []
        targets = []
        
        for app in applications:
            try:
                # Extract basic features
                gpa = float(app.get('gpa', 3.0)) if app.get('gpa') else 3.0
                ielts_score = float(app.get('ielts_score', 6.5)) if app.get('ielts_score') else 6.5
                work_exp = int(app.get('work_experience_years', 0)) if app.get('work_experience_years') else 0
                
                # Education level features
                current_education = app.get('current_education', 'undergraduate')
                education_level_score = self._get_education_level_score(current_education)
                
                # Course features
                course_name = app.get('course_applied', '')
                course_difficulty = self._get_course_difficulty(course_name)
                education_compatibility = self._calculate_education_compatibility(current_education, course_name)
                
                # Timing and status
                app_timing = self._get_application_timing_score(app.get('application_date'))
                is_international = 1 if app.get('nationality', 'UK') != 'UK' else 0
                
                # Calculate percentiles
                gpa_percentile = self._calculate_gpa_percentile(gpa, current_education)
                ielts_percentile = self._calculate_ielts_percentile(ielts_score)
                
                # Overall academic strength (weighted combination)
                overall_strength = (
                    gpa_percentile * 0.4 +
                    ielts_percentile * 0.3 +
                    education_compatibility * 0.2 +
                    min(work_exp / 5.0, 1.0) * 0.1  # Normalize work experience
                )
                
                feature_vector = [
                    gpa, ielts_score, work_exp, course_difficulty, app_timing, is_international,
                    education_level_score, education_compatibility, gpa_percentile,
                    ielts_percentile, overall_strength
                ]
                
                features.append(feature_vector)
                
                # Target
                status = str(app.get('status', 'Under Review'))
                target = 1 if 'accept' in status.lower() else 0
                targets.append(target)
                
            except (ValueError, TypeError) as e:
                logger.warning(f"Skipping application due to data error: {e}")
                continue
        
        return np.array(features), np.array(targets)

    def _get_education_level_score(self, education_level: str) -> float:
        """Convert education level to numerical score"""
        education_scores = {
            'high school': 1.0,
            'diploma': 1.5,
            'foundation': 2.0,
            'undergraduate': 3.0,
            'bachelor': 3.0,
            'graduate': 3.5,
            'postgraduate': 4.0,
            'masters': 4.0,
            'mba': 4.2,
            'phd': 5.0,
            'doctorate': 5.0
        }
        return education_scores.get(education_level.lower(), 2.5)

    def _get_course_difficulty(self, course_name: str) -> float:
        """Get course difficulty score based on course name and level"""
        try:
            course_lower = course_name.lower()
            
            # Difficulty based on course level
            if any(level in course_lower for level in ['phd', 'doctorate']):
                base_difficulty = 5.0
            elif any(level in course_lower for level in ['msc', 'ma', 'mba', 'masters', 'postgraduate']):
                base_difficulty = 4.0
            elif any(level in course_lower for level in ['bsc', 'ba', 'beng', 'undergraduate']):
                base_difficulty = 3.0
            elif any(level in course_lower for level in ['foundation', 'diploma']):
                base_difficulty = 2.0
            else:
                base_difficulty = 3.0  # Default to undergraduate level
            
            # Adjust based on subject area
            difficulty_adjustments = {
                # High difficulty fields
                'medicine': 1.5,
                'law': 1.2,
                'engineering': 1.1,
                'computer science': 1.0,
                'data science': 1.0,
                'artificial intelligence': 1.1,
                'physics': 1.0,
                'mathematics': 1.0,
                'chemistry': 0.9,
                
                # Medium difficulty fields
                'business': 0.8,
                'management': 0.8,
                'marketing': 0.7,
                'psychology': 0.8,
                'economics': 0.9,
                'finance': 0.9,
                
                # Lower difficulty fields
                'arts': 0.6,
                'design': 0.6,
                'media': 0.6,
                'communications': 0.6,
                'social sciences': 0.7,
                'education': 0.7
            }
            
            # Apply subject-based adjustment
            adjustment = 0.0
            for subject, adj in difficulty_adjustments.items():
                if subject in course_lower:
                    adjustment = adj
                    break
            
            final_difficulty = min(base_difficulty + adjustment, 5.0)  # Cap at 5.0
            return final_difficulty
            
        except Exception as e:
            self.logger.error(f"Error calculating course difficulty: {e}")
            return 3.0  # Default difficulty

    def _calculate_education_compatibility(self, current_education: str, target_course: str) -> float:
        """Calculate compatibility between current education and target course"""
        current_score = self._get_education_level_score(current_education)
        course_difficulty = self._get_course_difficulty(target_course)
        
        # Education progression rules
        level_difference = course_difficulty - current_score
        
        if level_difference <= 0:
            # Applying for same or lower level
            return max(0.5, 1.0 - abs(level_difference) * 0.2)
        elif level_difference <= 1.0:
            # Normal progression (one level up)
            return 0.95
        elif level_difference <= 2.0:
            # Challenging but possible progression
            return 0.7
        else:
            # Very challenging progression
            return max(0.3, 1.0 - level_difference * 0.2)

    def _get_application_timing_score(self, application_date: str) -> float:
        """Calculate application timing score (earlier applications get higher scores)"""
        try:
            if not application_date:
                return 0.5  # Default score
            
            # Parse application date
            if isinstance(application_date, str):
                app_date = datetime.fromisoformat(application_date.replace('Z', '+00:00'))
            else:
                app_date = application_date
            
            # Calculate months until typical intake (September)
            current_date = datetime.now()
            
            # Determine next September intake
            if current_date.month <= 9:
                next_intake = datetime(current_date.year, 9, 1)
            else:
                next_intake = datetime(current_date.year + 1, 9, 1)
            
            days_ahead = (next_intake - app_date).days
            
            # Score based on timing
            if days_ahead >= 180:  # 6+ months early
                return 1.0
            elif days_ahead >= 90:  # 3-6 months early
                return 0.8
            elif days_ahead >= 30:  # 1-3 months early
                return 0.6
            elif days_ahead >= 0:   # On time
                return 0.4
            else:                   # Late application
                return 0.2
                
        except Exception as e:
            self.logger.error(f"Error calculating timing score: {e}")
            return 0.5

    def _calculate_gpa_percentile(self, gpa: float, education_level: str) -> float:
        """Calculate GPA percentile based on education level"""
        # Define typical GPA distributions by education level
        gpa_ranges = {
            'high school': (2.0, 4.0),
            'undergraduate': (2.5, 4.0),
            'bachelor': (2.5, 4.0),
            'masters': (3.0, 4.0),
            'phd': (3.2, 4.0)
        }
        
        min_gpa, max_gpa = gpa_ranges.get(education_level.lower(), (2.0, 4.0))
        
        # Normalize to 0-1 scale
        normalized = (gpa - min_gpa) / (max_gpa - min_gpa)
        return max(0.0, min(1.0, normalized))

    def _calculate_ielts_percentile(self, ielts_score: float) -> float:
        """Calculate IELTS score percentile"""
        # IELTS range is typically 4.0 - 9.0
        min_score, max_score = 4.0, 9.0
        normalized = (ielts_score - min_score) / (max_score - min_score)
        return max(0.0, min(1.0, normalized))

    def _set_fallback_models(self):
        """Set fallback models when training fails"""
        self.logger.warning("Setting up fallback prediction models...")
        self.models_trained = False
        self.admission_predictor = None
        self.success_probability_model = None

    def predict_admission_success(self, applicant_data: Dict) -> Dict:
        """Predict admission success probability with comprehensive analysis"""
        try:
            if not self.models_trained or not SKLEARN_AVAILABLE:
                # Use rule-based fallback prediction
                return self._fallback_prediction(applicant_data)
            
            # Prepare features for prediction
            features = self._prepare_prediction_features(applicant_data)
            
            # Get predictions
            admission_prediction = self.admission_predictor.predict([features])[0]
            admission_probability = self.admission_predictor.predict_proba([features])[0][1]
            
            success_probability = self.success_probability_model.predict([features])[0]
            success_probability = max(0.0, min(1.0, success_probability))  # Clamp to valid range
            
            # Feature importance analysis
            feature_importance = self._analyze_feature_importance(features)
            
            # Generate recommendations
            recommendations = self._generate_improvement_recommendations(applicant_data, features)
            
            return {
                "success_probability": round(float(success_probability), 3),
                "admission_prediction": bool(admission_prediction),
                "confidence": round(float(admission_probability), 3),
                "feature_analysis": feature_importance,
                "recommendations": recommendations,
                "model_used": "ML_trained",
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"ML prediction error: {e}")
            return self._fallback_prediction(applicant_data)

    def _prepare_prediction_features(self, applicant_data: Dict) -> List[float]:
        """Prepare features for ML prediction"""
        # Extract basic features
        gpa = float(applicant_data.get('gpa', 3.0))
        ielts_score = float(applicant_data.get('ielts_score', 6.5))
        work_exp = int(applicant_data.get('work_experience_years', 0))
        
        # Education and course features
        current_education = applicant_data.get('current_education', 'undergraduate')
        course_applied = applicant_data.get('course_applied', '')
        
        education_level_score = self._get_education_level_score(current_education)
        course_difficulty = self._get_course_difficulty(course_applied)
        education_compatibility = self._calculate_education_compatibility(current_education, course_applied)
        
        # Timing and status features
        app_timing = self._get_application_timing_score(applicant_data.get('application_date'))
        is_international = 1 if applicant_data.get('nationality', 'UK') != 'UK' else 0
        
        # Calculate percentiles
        gpa_percentile = self._calculate_gpa_percentile(gpa, current_education)
        ielts_percentile = self._calculate_ielts_percentile(ielts_score)
        
        # Overall academic strength
        overall_strength = (
            gpa_percentile * 0.4 +
            ielts_percentile * 0.3 +
            education_compatibility * 0.2 +
            min(work_exp / 5.0, 1.0) * 0.1
        )
        
        return [
            gpa, ielts_score, work_exp, course_difficulty, app_timing, is_international,
            education_level_score, education_compatibility, gpa_percentile,
            ielts_percentile, overall_strength
        ]

    def _analyze_feature_importance(self, features: List[float]) -> Dict:
        """Analyze feature importance for the prediction"""
        try:
            if self.admission_predictor and hasattr(self.admission_predictor, 'feature_importances_'):
                importances = self.admission_predictor.feature_importances_
                
                feature_analysis = {}
                for i, (name, value, importance) in enumerate(zip(self.feature_names, features, importances)):
                    feature_analysis[name] = {
                        "value": round(float(value), 3),
                        "importance": round(float(importance), 3),
                        "contribution": round(float(value * importance), 3)
                    }
                
                return feature_analysis
            
        except Exception as e:
            logger.error(f"Feature importance analysis error: {e}")
        
        return {}

    def _generate_improvement_recommendations(self, applicant_data: Dict, features: List[float]) -> List[str]:
        """Generate recommendations for improving admission chances"""
        recommendations = []
        
        gpa = features[0]
        ielts_score = features[1]
        work_exp = features[2]
        education_compatibility = features[7]
        
        # GPA recommendations
        if gpa < 3.0:
            recommendations.append("🎓 Consider improving your GPA to at least 3.0 for better admission chances")
        elif gpa < 3.5:
            recommendations.append("📚 A higher GPA (3.5+) would significantly strengthen your application")
        
        # IELTS recommendations
        if ielts_score < 6.0:
            recommendations.append("📝 IELTS score below 6.0 may not meet minimum requirements - consider retaking")
        elif ielts_score < 6.5:
            recommendations.append("🗣️ Improving IELTS to 6.5+ would meet most program requirements")
        elif ielts_score < 7.0:
            recommendations.append("⭐ An IELTS score of 7.0+ would make you highly competitive")
        
        # Work experience recommendations
        if work_exp == 0:
            recommendations.append("💼 Consider gaining some work experience or internships to strengthen your profile")
        elif work_exp < 2:
            recommendations.append("🚀 Additional work experience would enhance your application")
        
        # Education compatibility
        if education_compatibility < 0.7:
            recommendations.append("🎯 Consider foundation courses to better prepare for your target program")
        
        # Course-specific recommendations
        course_applied = applicant_data.get('course_applied', '').lower()
        if 'computer science' in course_applied or 'data science' in course_applied:
            recommendations.append("💻 Consider showcasing programming projects or technical certifications")
        elif 'business' in course_applied or 'management' in course_applied:
            recommendations.append("📈 Leadership experience and business projects would strengthen your application")
        
        if not recommendations:
            recommendations.append("✅ Your profile looks strong! Consider applying early for the best chances")
        
        return recommendations[:5]  # Limit to top 5 recommendations

    def _fallback_prediction(self, applicant_data: Dict) -> Dict:
        """Rule-based fallback prediction when ML models are unavailable"""
        try:
            # Extract key factors
            gpa = float(applicant_data.get('gpa', 3.0))
            ielts_score = float(applicant_data.get('ielts_score', 6.5))
            work_exp = int(applicant_data.get('work_experience_years', 0))
            
            # Calculate score based on rules
            score = 0.0
            factors = []
            
            # GPA scoring
            if gpa >= 3.7:
                score += 0.4
                factors.append("🎓 Excellent GPA")
            elif gpa >= 3.3:
                score += 0.3
                factors.append("📚 Good GPA")
            elif gpa >= 2.8:
                score += 0.2
                factors.append("✅ Acceptable GPA")
            else:
                score += 0.1
                factors.append("⚠️ GPA below recommended level")
            
            # IELTS scoring
            if ielts_score >= 7.5:
                score += 0.3
                factors.append("🗣️ Excellent English proficiency")
            elif ielts_score >= 6.5:
                score += 0.25
                factors.append("👍 Good English proficiency")
            elif ielts_score >= 6.0:
                score += 0.2
                factors.append("✅ Meets minimum English requirements")
            else:
                score += 0.1
                factors.append("⚠️ English score may be insufficient")
            
            # Work experience scoring
            if work_exp >= 3:
                score += 0.2
                factors.append("💼 Substantial work experience")
            elif work_exp >= 1:
                score += 0.15
                factors.append("🚀 Some work experience")
            else:
                score += 0.05
                factors.append("📝 Limited work experience")
            
            # Additional factors
            nationality = applicant_data.get('nationality', 'UK')
            if nationality == 'UK':
                score += 0.1
                factors.append("🇬🇧 Domestic student advantage")
            
            # Cap score and determine prediction
            final_score = min(score, 1.0)
            prediction = final_score > 0.6
            
            recommendations = []
            if gpa < 3.0:
                recommendations.append("Improve academic performance")
            if ielts_score < 6.5:
                recommendations.append("Retake IELTS exam for higher score")
            if work_exp == 0:
                recommendations.append("Gain relevant work experience")
            
            return {
                "success_probability": round(final_score, 3),
                "admission_prediction": prediction,
                "confidence": 0.7,  # Lower confidence for rule-based
                "factors": factors,
                "recommendations": recommendations,
                "model_used": "rule_based_fallback",
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Fallback prediction error: {e}")
            return {
                "success_probability": 0.5,
                "admission_prediction": True,
                "confidence": 0.3,
                "error": str(e),
                "model_used": "error_fallback",
                "timestamp": datetime.now().isoformat()
            }

# =============================================================================
# UNIFIED UEL AI AGENT
# =============================================================================

class UnifiedUELAIAgent:
    """Enhanced AI Agent with comprehensive profile integration"""
    
    def __init__(self):
        """Initialize all AI services with profile support"""
        self.logger = logging.getLogger(__name__)
        self.logger.info("Initializing UEL AI Agent with Profile Integration...")
        
        try:
            # Initialize core services
            self.db_manager = DatabaseManager()
            self.profile_manager = ProfileManager(self.db_manager)
            self.data_manager = DataManager()
            
            # Initialize AI services
            self.ollama_service = OllamaService()
            self.sentiment_engine = SentimentAnalysisEngine()
            self.predictive_engine = PredictiveAnalyticsEngine(self.data_manager)  
            self.recommendation_system = CourseRecommendationSystem(self.data_manager)
            self.document_verifier = DocumentVerificationAI()
            self.voice_service = VoiceService()
            
            # System state
            self.conversation_history = []
            self.session_start_time = datetime.now()
            self.interaction_count = 0
            self.error_count = 0
            
            self.logger.info("✅ UEL AI Agent initialized successfully")
            
        except Exception as e:
            self.logger.error(f"❌ Error initializing UEL AI Agent: {e}")
            self._initialize_fallback_services()
    
    def get_ai_response(self, user_input: str, force_profile_check: bool = True) -> Dict:
        """Get AI response with automatic profile integration"""
        try:
            self.interaction_count += 1
            start_time = time.time()
            
            # Get current profile
            current_profile = self.profile_manager.get_current_profile()
            
            if force_profile_check and not current_profile:
                return {
                    'response': self._generate_profile_required_message(),
                    'requires_profile': True,
                    'sentiment': {'sentiment': 'neutral', 'polarity': 0.0},
                    'llm_used': 'system',
                    'response_time': time.time() - start_time
                }
            
            # Update profile activity
            if current_profile:
                current_profile.add_interaction('ai_chat')
                self.profile_manager.save_profile(current_profile)
            
            # Analyze sentiment
            try:
                sentiment_data = self.sentiment_engine.analyze_message_sentiment(user_input)
            except Exception as e:
                self.logger.warning(f"Sentiment analysis failed: {e}")
                sentiment_data = {"sentiment": "neutral", "polarity": 0.0}
            
            # Build enhanced context with profile
            context = self._build_comprehensive_context(current_profile, sentiment_data)
            enhanced_prompt = self._enhance_prompt_with_profile(user_input, current_profile)
            
            # Get LLM response
            try:
                if self.ollama_service.is_available():
                    ai_response = self.ollama_service.generate_response(
                        enhanced_prompt, 
                        system_prompt=context
                    )
                    llm_used = self.ollama_service.model_name
                else:
                    ai_response = self._fallback_response_with_profile(user_input, current_profile)
                    llm_used = "fallback"
            except Exception as e:
                self.logger.error(f"LLM response error: {e}")
                ai_response = self._fallback_response_with_profile(user_input, current_profile)
                llm_used = "fallback_error"
            
            response_time = time.time() - start_time
            
            # Store interaction
            try:
                self.conversation_history.append({
                    'user_input': user_input,
                    'ai_response': ai_response,
                    'profile_id': current_profile.id if current_profile else None,
                    'sentiment': sentiment_data,
                    'timestamp': datetime.now().isoformat(),
                    'response_time': response_time,
                    'llm_used': llm_used
                })
            except Exception as e:
                self.logger.warning(f"Error storing conversation: {e}")
            
            return {
                'response': ai_response,
                'profile_integrated': bool(current_profile),
                'profile_completion': current_profile.profile_completion if current_profile else 0,
                'sentiment': sentiment_data,
                'llm_used': llm_used,
                'response_time': response_time,
                'confidence': 0.9 if llm_used != "fallback" else 0.6
            }
            
        except Exception as e:
            self.error_count += 1
            self.logger.error(f"Error getting AI response: {e}")
            return {
                'response': "I apologize, but I'm experiencing technical difficulties. Please try again or contact support.",
                'error': str(e),
                'requires_profile': False,
                'sentiment': {'sentiment': 'neutral', 'polarity': 0.0},
                'llm_used': 'error',
                'response_time': 0.0
            }
    
    def predict_admission_success(self, course_name: str = None, additional_data: Dict = None) -> Dict:
        """Predict admission success using profile data"""
        try:
            current_profile = self.profile_manager.get_current_profile()
            
            if not current_profile:
                return {
                    "error": "Profile required for predictions",
                    "requires_profile": True,
                    "success_probability": 0.0
                }
            
            # Update profile activity
            current_profile.add_interaction('predictions')
            self.profile_manager.save_profile(current_profile)
            
            # Build applicant data from profile
            applicant_data = self._build_applicant_data_from_profile(current_profile, course_name, additional_data)
            
            # Get prediction
            prediction_result = self.predictive_engine.predict_admission_success(applicant_data)
            
            # Add profile context to result
            prediction_result.update({
                'profile_id': current_profile.id,
                'profile_completion': current_profile.profile_completion,
                'based_on_profile': True
            })
            
            return prediction_result
            
        except Exception as e:
            self.logger.error(f"Prediction error: {e}")
            return {
                "error": str(e),
                "success_probability": 0.5,
                "confidence": 0.0
            }
    
    def get_course_recommendations(self, preferences: Dict = None) -> List[Dict]:
        """Get course recommendations using profile data"""
        try:
            current_profile = self.profile_manager.get_current_profile()
            
            if not current_profile:
                return []
            
            # Update profile activity
            current_profile.add_interaction('recommendations')
            self.profile_manager.save_profile(current_profile)
            
            # Build user profile for recommendations
            user_profile = self._build_recommendation_profile(current_profile)
            
            # Get recommendations
            recommendations = self.recommendation_system.recommend_courses(user_profile, preferences)
            
            # Add profile context to recommendations
            for rec in recommendations:
                rec['based_on_profile'] = True
                rec['profile_id'] = current_profile.id
                rec['profile_completion'] = current_profile.profile_completion
            
            return recommendations
            
        except Exception as e:
            self.logger.error(f"Recommendation error: {e}")
            return []
    
    def _generate_profile_required_message(self) -> str:
        """Generate message prompting profile creation"""
        return """👋 **Welcome to UEL AI Assistant!**

To provide you with personalized assistance, I need to know a bit about you first. 

🎯 **Why create a profile?**
• Get personalized course recommendations
• Receive tailored AI responses
• Get accurate admission predictions
• Save time across all features

**Quick Setup:**
Just click the "Create Profile" button to get started. It takes less than 2 minutes!

Once your profile is ready, I'll be able to help you with:
• Intelligent course matching
• Admission success predictions  
• Personalized academic guidance
• Tailored university information

Ready to begin your UEL journey? 🚀"""
    
    def _build_comprehensive_context(self, profile: Optional[UserProfile], sentiment_data: Dict) -> str:
        """Build comprehensive system context with profile"""
        context_parts = [
            f"You are an intelligent AI assistant for {config.university_name} (UEL).",
            "You help prospective and current students with admissions, courses, applications, and university life.",
            "You are knowledgeable, helpful, professional, and personalized.",
            ""
        ]
        
        # Add university information
        context_parts.extend([
            "UNIVERSITY INFORMATION:",
            f"- Name: {config.university_name}",
            f"- Contact: {config.admissions_email}, {config.admissions_phone}",
            "- Location: London, UK",
            "- Known for: Diverse programs, practical education, industry connections",
            ""
        ])
        
        # Add profile context if available
        if profile:
            context_parts.extend([
                "STUDENT PROFILE:",
                f"- Name: {profile.first_name} {profile.last_name}",
                f"- Primary Interest: {profile.field_of_interest}",
                f"- Academic Level: {profile.academic_level}",
                f"- Country: {profile.country}",
                f"- Profile Completion: {profile.profile_completion:.1f}%",
                ""
            ])
            
            # Add academic background
            if profile.gpa > 0 or profile.ielts_score > 0:
                context_parts.extend([
                    "ACADEMIC BACKGROUND:",
                    f"- GPA: {profile.gpa if profile.gpa > 0 else 'Not specified'}",
                    f"- IELTS: {profile.ielts_score if profile.ielts_score > 0 else 'Not specified'}",
                    f"- Work Experience: {profile.work_experience_years} years",
                    ""
                ])
            
            # Add interests and goals
            if profile.interests or profile.career_goals:
                context_parts.extend([
                    "INTERESTS & GOALS:",
                    f"- Interests: {', '.join(profile.interests) if profile.interests else 'Not specified'}",
                    f"- Career Goals: {profile.career_goals[:100]}{'...' if len(profile.career_goals) > 100 else ''}",
                    ""
                ])
        
        # Add sentiment context
        if sentiment_data:
            sentiment = sentiment_data.get('sentiment', 'neutral')
            emotions = sentiment_data.get('emotions', [])
            urgency = sentiment_data.get('urgency', 'normal')
            
            context_parts.extend([
                "CONVERSATION CONTEXT:",
                f"- User sentiment: {sentiment}",
                f"- Detected emotions: {', '.join(emotions) if emotions else 'None'}",
                f"- Urgency level: {urgency}",
                ""
            ])
        
        # Add personalized guidelines
        context_parts.extend([
            "PERSONALIZED RESPONSE GUIDELINES:",
            "- Use the student's name naturally in conversation",
            "- Reference their interests and goals when relevant",
            "- Provide advice tailored to their academic level",
            "- Consider their profile completion and suggest improvements",
            "- Be encouraging about their academic journey",
            "- Offer specific guidance based on their background",
            ""
        ])
        
        return "\n".join(context_parts)
    
    def _enhance_prompt_with_profile(self, user_input: str, profile: Optional[UserProfile]) -> str:
        """Enhance user prompt with profile context"""
        if not profile:
            return user_input
        
        enhanced_parts = [user_input]
        
        # Add relevant profile context
        context_info = []
        
        # Add interest context if relevant
        if profile.field_of_interest and any(word in user_input.lower() for word in ['course', 'program', 'study', 'major']):
            context_info.append(f"(My primary interest is {profile.field_of_interest})")
        
        # Add academic level context
        if profile.academic_level and any(word in user_input.lower() for word in ['admission', 'apply', 'requirement']):
            context_info.append(f"(My current academic level is {profile.academic_level})")
        
        # Add location context if relevant
        if profile.country and any(word in user_input.lower() for word in ['visa', 'international', 'fee']):
            context_info.append(f"(I am from {profile.country})")
        
        if context_info:
            enhanced_parts.extend(context_info)
        
        # Add conversation history context
        if self.conversation_history:
            recent_context = self.conversation_history[-2:]  # Last 2 interactions
            if recent_context:
                enhanced_parts.append("\nRecent conversation context:")
                for interaction in recent_context:
                    enhanced_parts.append(f"Previous Q: {interaction['user_input'][:100]}...")
        
        return "\n".join(enhanced_parts)
    
    def _fallback_response_with_profile(self, user_input: str, profile: Optional[UserProfile]) -> str:
        """Provide fallback response with profile personalization"""
        user_lower = user_input.lower()
        
        # Personalization prefix
        greeting = ""
        if profile:
            greeting = f"Hi {profile.first_name}! "
        
        # Course-related queries
        if any(word in user_lower for word in ['course', 'program', 'study', 'degree']):
            response = f"""{greeting}Here are some excellent UEL programs that might interest you:

🎓 **Popular Programs:**
• Computer Science - Programming, AI, Software Development
• Business Management - Leadership, Strategy, Finance  
• Data Science - Analytics, Machine Learning, Statistics
• Engineering - Civil, Mechanical, Electronic
• Psychology - Clinical, Research, Counseling"""

            # Add personalized course suggestions
            if profile and profile.field_of_interest:
                response += f"\n\n🎯 **Based on your interest in {profile.field_of_interest}:**"
                if 'computer' in profile.field_of_interest.lower():
                    response += "\n• Computer Science BSc/MSc programs would be perfect for you"
                    response += "\n• Data Science MSc for advanced analytics skills"
                elif 'business' in profile.field_of_interest.lower():
                    response += "\n• Business Management BA/MBA programs"
                    response += "\n• Finance or Marketing specializations"
            
            response += f"\n\nFor detailed information, contact {config.admissions_email}"
            return response

        # Application-related queries
        elif any(word in user_lower for word in ['apply', 'application', 'admission']):
            response = f"""{greeting}I'd be happy to help with your UEL application!

📝 **Application Process:**
1. Choose your course
2. Check entry requirements
3. Submit online application  
4. Upload supporting documents
5. Attend interview (if required)"""

            # Add personalized requirements
            if profile:
                if profile.academic_level:
                    response += f"\n\n🎓 **For {profile.academic_level} level students:**"
                    if 'undergraduate' in profile.academic_level.lower():
                        response += "\n• A-levels or equivalent required"
                        response += "\n• IELTS 6.0 minimum"
                    elif 'postgraduate' in profile.academic_level.lower():
                        response += "\n• Bachelor's degree required"
                        response += "\n• IELTS 6.5 minimum"
                
                if profile.country and profile.country.lower() != 'uk':
                    response += f"\n\n🌍 **For international students from {profile.country}:**"
                    response += "\n• Visa guidance available"
                    response += "\n• International fee rates apply"
            
            response += f"\n\n**Contact:** {config.admissions_email} | {config.admissions_phone}"
            return response

        # Default personalized response
        else:
            response = f"""{greeting}Welcome to **University of East London**!

I'm your AI assistant, here to help with:
🎓 **Course Information** - Programs, requirements, curriculum
📝 **Applications** - Process, deadlines, requirements  
💰 **Fees & Funding** - Tuition, scholarships, payments
🏫 **Campus Life** - Facilities, accommodation, activities"""

            if profile:
                response += f"\n\n👤 **Your Profile Status:**"
                response += f"\n• Profile: {profile.profile_completion:.1f}% complete"
                response += f"\n• Primary Interest: {profile.field_of_interest}"
                if profile.profile_completion < 80:
                    response += f"\n💡 Complete your profile for better recommendations!"
            
            response += f"\n\nHow can I help you today?"
            return response
    
    def _build_applicant_data_from_profile(self, profile: UserProfile, course_name: str = None, additional_data: Dict = None) -> Dict:
        """Build applicant data from profile for predictions"""
        applicant_data = {
            'name': f"{profile.first_name} {profile.last_name}",
            'email': profile.email,
            'gpa': profile.gpa if profile.gpa > 0 else 3.0,
            'ielts_score': profile.ielts_score if profile.ielts_score > 0 else 6.5,
            'work_experience_years': profile.work_experience_years,
            'course_applied': course_name or profile.field_of_interest,
            'nationality': profile.nationality or profile.country,
            'application_date': datetime.now().strftime('%Y-%m-%d'),
            'current_education': profile.academic_level,
            'major_field': profile.current_major or profile.field_of_interest,
            'age': self._calculate_age_from_profile(profile),
            'career_goals': profile.career_goals,
            'target_industry': profile.target_industry,
            'preferred_start_date': profile.preferred_start_date
        }
        
        # Add additional data if provided
        if additional_data:
            applicant_data.update(additional_data)
        
        return applicant_data
    
    def _build_recommendation_profile(self, profile: UserProfile) -> Dict:
        """Build user profile for course recommendations"""
        return {
            'first_name': profile.first_name,
            'last_name': profile.last_name,
            'field_of_interest': profile.field_of_interest,
            'academic_level': profile.academic_level,
            'interests': profile.interests,
            'career_goals': profile.career_goals,
            'gpa': profile.gpa,
            'ielts_score': profile.ielts_score,
            'work_experience_years': profile.work_experience_years,
            'target_industry': profile.target_industry,
            'country': profile.country,
            'nationality': profile.nationality,
            'budget_range': profile.budget_range,
            'preferred_study_mode': profile.preferred_study_mode,
            'preferred_courses': profile.preferred_courses,
            'rejected_courses': profile.rejected_courses
        }
    
    def _calculate_age_from_profile(self, profile: UserProfile) -> int:
        """Calculate age from profile data"""
        try:
            if profile.date_of_birth:
                birth_date = datetime.fromisoformat(profile.date_of_birth)
                today = datetime.now()
                age = today.year - birth_date.year
                if today.month < birth_date.month or (today.month == birth_date.month and today.day < birth_date.day):
                    age -= 1
                return age
        except:
            pass
        
        # Default age based on academic level
        age_mapping = {
            'high school': 18,
            'undergraduate': 20,
            'graduate': 23,
            'postgraduate': 25,
            'masters': 26,
            'phd': 28
        }
        
        return age_mapping.get(profile.academic_level.lower(), 22)
    
    def _initialize_fallback_services(self):
        """Initialize minimal fallback services when main initialization fails"""
        try:
            self.logger.info("Initializing fallback services...")
            
            # Initialize minimal required services
            self.db_manager = None
            self.profile_manager = None
            self.data_manager = None
            
            # Set up basic AI services with error handling
            self.ollama_service = None
            self.sentiment_engine = None
            self.predictive_engine = None
            self.recommendation_system = None
            self.document_verifier = None
            self.voice_service = None
            
            # Initialize basic state
            self.conversation_history = []
            self.session_start_time = datetime.now()
            self.interaction_count = 0
            self.error_count = 0
            
            self.logger.warning("⚠️ Running in fallback mode with limited functionality")
            
        except Exception as e:
            self.logger.error(f"❌ Even fallback initialization failed: {e}")
            # Initialize absolute minimum required for class to function
            self.conversation_history = []
            self.session_start_time = datetime.now()
            self.interaction_count = 0
            self.error_count = 0

# =============================================================================
# EXPORTS AND INITIALIZATION
# =============================================================================

def initialize_uel_ai_system():
    """Initialize the UEL AI system with proper error handling"""
    try:
        logger.info("🚀 Initializing UEL AI System...")
        
        # Test basic imports and configurations
        logger.info("✅ Basic imports successful")
        logger.info(f"✅ Configuration loaded: {config.university_name}")
        
        # Initialize the main agent
        agent = UnifiedUELAIAgent()
        logger.info("✅ UEL AI System initialized successfully!")
        
        return agent
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize AI system: {e}")
        import traceback
        logger.error(f"Full traceback: {traceback.format_exc()}")
        raise

# Re-export everything to maintain compatibility
__all__ = [
    'UnifiedUELAIAgent',
    'UserProfile', 
    'ProfileManager',
    'SystemConfig', 
    'config',
    'DatabaseManager',
    'DataManager',
    'OllamaService',
    'SentimentAnalysisEngine',
    'PredictiveAnalyticsEngine',
    'CourseRecommendationSystem',
    'DocumentVerificationAI',
    'VoiceService',
    'initialize_uel_ai_system'
]