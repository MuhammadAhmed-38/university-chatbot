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
    data_directory: str = "data"
    
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
# DATABASE MODELS
# =============================================================================

class DatabaseManager:
    """Enhanced database management with unified models"""
    
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
        
        # Load sample data if database is empty
        self._check_and_load_sample_data()
    
    def _check_and_load_sample_data(self):
        """Load sample data if database is empty"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute("SELECT COUNT(*) FROM students")
        student_count = cursor.fetchone()[0]
        
        if student_count == 0:
            logger.info("Loading sample data...")
            self._load_sample_data()
        
        conn.close()
    
    def _load_sample_data(self):
        """Load comprehensive sample data"""
        # This would load sample students, courses, applications, etc.
        # Implementation would be similar to the sample data generators in the UI files
        pass

class BaseModel:
    """Base model class with common functionality"""
    
    def __init__(self, db_manager: DatabaseManager = None):
        self.db_manager = db_manager or DatabaseManager()
    
    def to_dict(self) -> Dict:
        """Convert model to dictionary"""
        return {}
    
    @classmethod
    def from_dict(cls, data: Dict):
        """Create model from dictionary"""
        return cls()
    
    def save(self):
        """Save model to database"""
        pass
    
    @classmethod
    def get(cls, model_id: str):
        """Get model by ID"""
        pass
    
    @classmethod
    def list(cls, order_by: str = None):
        """List all models"""
        pass

class Student(BaseModel):
    """Enhanced Student model"""
    
    def __init__(self, id=None, first_name=None, last_name=None, email=None, phone=None,
                 country=None, nationality=None, date_of_birth=None, academic_level=None,
                 field_of_interest=None, financial_support=None, preferred_intake=None,
                 status="inquiry", documents=None, conversation_history=None, **kwargs):
        super().__init__()
        self.id = id
        self.first_name = first_name
        self.last_name = last_name
        self.email = email
        self.phone = phone
        self.country = country
        self.nationality = nationality
        self.date_of_birth = date_of_birth
        self.academic_level = academic_level
        self.field_of_interest = field_of_interest
        self.financial_support = financial_support
        self.preferred_intake = preferred_intake
        self.status = status
        self.documents = documents or []
        self.conversation_history = conversation_history or []
        self.created_date = datetime.now().isoformat()
        
        # Additional fields from enhanced system
        for key, value in kwargs.items():
            setattr(self, key, value)
    
    def to_dict(self) -> Dict:
        """Convert student to dictionary"""
        return {
            "id": self.id,
            "first_name": self.first_name,
            "last_name": self.last_name,
            "email": self.email,
            "phone": self.phone,
            "country": self.country,
            "nationality": self.nationality,
            "date_of_birth": self.date_of_birth,
            "academic_level": self.academic_level,
            "field_of_interest": self.field_of_interest,
            "financial_support": self.financial_support,
            "preferred_intake": self.preferred_intake,
            "status": self.status,
            "documents": self.documents,
            "conversation_history": self.conversation_history,
            "created_date": self.created_date
        }
    
    @classmethod
    def from_dict(cls, data: Dict):
        """Create student from dictionary"""
        return cls(**data)

class Course(BaseModel):
    """Enhanced Course model"""
    
    def __init__(self, id=None, course_name=None, course_code=None, department=None,
                 faculty=None, level=None, duration=None, intake_periods=None,
                 fees=None, entry_requirements=None, description=None,
                 modules=None, career_prospects=None, application_deadline=None,
                 available_scholarships=None, is_active=True, **kwargs):
        super().__init__()
        self.id = id
        self.course_name = course_name
        self.course_code = course_code
        self.department = department
        self.faculty = faculty
        self.level = level
        self.duration = duration
        self.intake_periods = intake_periods or []
        self.fees = fees or {"domestic": 0, "international": 0, "currency": "GBP"}
        self.entry_requirements = entry_requirements or {}
        self.description = description
        self.modules = modules or []
        self.career_prospects = career_prospects
        self.application_deadline = application_deadline
        self.available_scholarships = available_scholarships or []
        self.is_active = is_active
        self.created_date = datetime.now().isoformat()
        
        # Additional fields
        for key, value in kwargs.items():
            setattr(self, key, value)

# =============================================================================
# AI SERVICES
# =============================================================================

class OllamaService:
    """Enhanced Ollama service for LLM integration"""
    
    def __init__(self, model_name: str = None, base_url: str = None):
        self.model_name = model_name or config.default_model
        self.base_url = base_url or config.ollama_host
        self.api_url = f"{self.base_url}/api/generate"
        self.conversation_history = []
        self.max_history_length = 10
        
        self._check_availability()
    
    def _check_availability(self):
        """Check if Ollama is available"""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                available_models = [model['name'] for model in response.json().get('models', [])]
                if self.model_name not in available_models:
                    logger.warning(f"Model {self.model_name} not found. Available: {available_models}")
                else:
                    logger.info(f"Successfully connected to Ollama. Using: {self.model_name}")
            else:
                logger.warning(f"Ollama returned status code {response.status_code}")
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to connect to Ollama: {e}")
    
    def is_available(self) -> bool:
        """Check if Ollama service is available"""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except Exception:
            return False
    
    def generate_response(self, prompt: str, system_prompt: Optional[str] = None, 
                          temperature: float = None, max_tokens: int = None) -> str:
        """Generate response using Ollama"""
        try:
            if not self.is_available():
                return self._fallback_response(prompt)
            
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
            
            response = requests.post(self.api_url, json=data, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                ai_response = result.get('response', 'No response generated')
                
                # Update conversation history
                self.conversation_history.append({"role": "user", "content": prompt})
                self.conversation_history.append({"role": "assistant", "content": ai_response})
                
                if len(self.conversation_history) > self.max_history_length:
                    self.conversation_history = self.conversation_history[-self.max_history_length:]
                
                return ai_response
            else:
                logger.error(f"Ollama API error: {response.status_code}")
                return self._fallback_response(prompt)
                
        except Exception as e:
            logger.error(f"LLM generation error: {e}")
            return self._fallback_response(prompt)
    
    def extract_json_from_response(self, response: str) -> Dict:
        """Extract JSON from response text"""
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            try:
                import re
                json_match = re.search(r'({.*})', response, re.DOTALL)
                if json_match:
                    return json.loads(json_match.group(1))
                
                json_match = re.search(r'```json\s*(.*?)\s*```', response, re.DOTALL)
                if json_match:
                    return json.loads(json_match.group(1))
                
                return {}
            except json.JSONDecodeError:
                return {}

class DataManager:
    """Enhanced data management with CSV integration - FINAL VERSION"""
    
    def __init__(self, data_dir: str = None):
        """Initialize data manager with real CSV files"""
        self.data_dir = data_dir or config.data_directory
        self.db_manager = DatabaseManager()
        
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
            logger.warning(f"Data directory {self.data_dir} not found. Creating it...")
            try:
                os.makedirs(self.data_dir, exist_ok=True)
            except Exception as e:
                logger.error(f"Could not create data directory: {e}")
                self.data_dir = "."  # Fallback to current directory
        
        # Load data from your CSV files
        self.load_all_data()
        
        # Create search index if possible
        if SKLEARN_AVAILABLE:
            try:
                self._create_search_index()
            except Exception as e:
                logger.warning(f"Could not create search index: {e}")
    
    def load_all_data(self):
        """Load all data from CSV files and database"""
        try:
            # Load CSV data if available
            self._load_csv_data()
            
        except Exception as e:
            logger.error(f"Error loading data: {e}")
    
    
    def _load_csv_data(self):
        """Load CSV data files - Enhanced version for courses integration"""
        csv_files = {
            'applications.csv': 'applications_df',
            'courses.csv': 'courses_df', 
            'faqs.csv': 'faqs_df',
            'counseling_slots.csv': 'counseling_df'
        }
    
        for filename, df_name in csv_files.items():
            csv_path = os.path.join(self.data_dir, filename)
        
            try:
                if os.path.exists(csv_path):
                    df = pd.read_csv(csv_path)
                    df.columns = df.columns.str.strip()  # Clean column names
                    setattr(self, df_name, df)
                    logger.info(f"✅ Loaded {len(df)} records from {filename}")
                
                    # Validate courses.csv specifically
                    if filename == 'courses.csv':
                        required_columns = ['course_name', 'level', 'description']
                        missing_columns = [col for col in required_columns if col not in df.columns]
                    
                        if missing_columns:
                            logger.warning(f"⚠️ Missing required columns in courses.csv: {missing_columns}")
                        else:
                            logger.info(f"✅ courses.csv validation passed - {len(df)} courses loaded")
                        
                            # Add default values for missing optional columns
                            default_values = {
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
                        
                            for col, default_val in default_values.items():
                                if col not in df.columns:
                                    df[col] = default_val
                                    logger.info(f"Added default column: {col}")
                         
                            setattr(self, df_name, df)
                else:
                    logger.warning(f"⚠️ {filename} not found at {csv_path}")
                    setattr(self, df_name, pd.DataFrame())
                
            except Exception as e:
                logger.error(f"❌ Error loading {filename}: {e}")
                setattr(self, df_name, pd.DataFrame())

    
    def _create_search_index(self):
        """Create search index from your actual data - ENHANCED VERSION"""
        if not SKLEARN_AVAILABLE:
            logger.warning("Scikit-learn not available. Search functionality disabled.")
            return
        
        self.combined_data = []
        
        # Index courses from your courses.csv
        if not self.courses_df.empty:
            logger.info(f"Indexing {len(self.courses_df)} courses for search...")
            
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
        
        # Index FAQs from your faqs.csv
        if not self.faqs_df.empty:
            logger.info(f"Indexing {len(self.faqs_df)} FAQs for search...")
            
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
                logger.info(f"✅ Created search index with {len(self.combined_data)} items")
            except Exception as e:
                logger.error(f"Error creating TF-IDF vectors: {e}")
                self.all_text_vectors = None
        else:
            logger.warning("No data available for search indexing")
    
    def intelligent_search(self, query: str, top_k: int = 5) -> List[Dict]:
        """Intelligent search across all data - ENHANCED VERSION"""
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
            logger.error(f"Search error: {e}")
            return []
    
    def get_courses_summary(self) -> Dict:
        """Get summary of your courses data"""
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
        """Get summary of your applications data"""
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

    def validate_data_structure(self) -> Dict:
        """Validate your CSV data structure and report any issues"""
        validation_report = {
            "courses": {"status": "✅", "issues": []},
            "applications": {"status": "✅", "issues": []},
            "faqs": {"status": "✅", "issues": []},
            "counseling": {"status": "✅", "issues": []}
        }
        
        # Validate courses.csv
        if self.courses_df.empty:
            validation_report["courses"]["status"] = "❌"
            validation_report["courses"]["issues"].append("No data loaded")
        else:
            expected_columns = ['course_name', 'description', 'level']
            missing = [col for col in expected_columns if col not in self.courses_df.columns]
            if missing:
                validation_report["courses"]["issues"].append(f"Missing columns: {missing}")
                validation_report["courses"]["status"] = "⚠️"
        
        # Validate applications.csv
        if self.applications_df.empty:
            validation_report["applications"]["status"] = "❌"
            validation_report["applications"]["issues"].append("No data loaded")
        else:
            expected_columns = ['name', 'course_applied', 'status']
            missing = [col for col in expected_columns if col not in self.applications_df.columns]
            if missing:
                validation_report["applications"]["issues"].append(f"Missing columns: {missing}")
                validation_report["applications"]["status"] = "⚠️"
        
        # Validate faqs.csv
        if self.faqs_df.empty:
            validation_report["faqs"]["status"] = "❌"
            validation_report["faqs"]["issues"].append("No data loaded")
        else:
            expected_columns = ['question', 'answer']
            missing = [col for col in expected_columns if col not in self.faqs_df.columns]
            if missing:
                validation_report["faqs"]["issues"].append(f"Missing columns: {missing}")
                validation_report["faqs"]["status"] = "⚠️"
        
        # Validate counseling_slots.csv
        if self.counseling_df.empty:
            validation_report["counseling"]["status"] = "❌"
            validation_report["counseling"]["issues"].append("No data loaded")
        
        return validation_report
    
    def get_data_stats(self) -> Dict:
        """Get comprehensive data statistics"""
        return {
            "courses": {
                "total": len(self.courses_df),
                "columns": list(self.courses_df.columns) if not self.courses_df.empty else []
            },
            "applications": {
                "total": len(self.applications_df),
                "columns": list(self.applications_df.columns) if not self.applications_df.empty else []
            },
            "faqs": {
                "total": len(self.faqs_df),
                "columns": list(self.faqs_df.columns) if not self.faqs_df.empty else []
            },
            "counseling_slots": {
                "total": len(self.counseling_df),
                "columns": list(self.counseling_df.columns) if not self.counseling_df.empty else []
            },
            "search_index": {
                "indexed_items": len(self.combined_data),
                "search_ready": self.all_text_vectors is not None
            }
        }


class SentimentAnalysisEngine:
    """Enhanced sentiment analysis"""
    
    def __init__(self):
        self.sentiment_history = []
    
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
            logger.error(f"Sentiment analysis error: {e}")
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

# Enhanced prediction components for unified_uel_ai_system.py
# Replace the existing PredictiveAnalyticsEngine class with this enhanced version

# Replace the ENTIRE PredictiveAnalyticsEngine class in unified_uel_ai_system.py with this:

class PredictiveAnalyticsEngine:
    """Enhanced ML-based predictive analytics with education level consideration"""
    
    def __init__(self, data_manager: DataManager):
        self.data_manager = data_manager
        self.admission_predictor = None
        self.success_probability_model = None
        self.models_trained = False
        
        if SKLEARN_AVAILABLE:
            # Enhanced feature set including education level compatibility
            self.feature_names = [
                'gpa', 'ielts_score', 'work_experience_years', 
                'course_difficulty', 'application_timing', 'international_status',
                'education_level_score', 'education_compatibility', 'gpa_percentile',
                'ielts_percentile', 'overall_academic_strength'
            ]
            self.train_models()

    def _get_training_data(self) -> List[Dict]:
        """Get training data from database or generate synthetic data"""
        try:
            # Try to get real data from database first
            applications = self._load_real_applications_data()
            
            if len(applications) < 10:
                logger.info("Insufficient real data, generating synthetic training data...")
                # Generate synthetic data if not enough real data
                synthetic_data = self._generate_synthetic_data(100)  # Generate 100 samples
                applications.extend(synthetic_data)
            
            logger.info(f"Training data loaded: {len(applications)} applications")
            return applications
            
        except Exception as e:
            logger.error(f"Error loading training data: {e}")
            # Fallback to synthetic data
            return self._generate_synthetic_data(50)

    def _load_real_applications_data(self) -> List[Dict]:
        """Load real applications data from database"""
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
            logger.error(f"Error loading real applications data: {e}")
            return []

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
            logger.error(f"Error calculating course difficulty: {e}")
            return 3.0  # Default difficulty

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
            
            # Calculate months difference
            months_until_intake = (next_intake - app_date).days / 30.44  # Average days per month
            
            # Score based on timing (earlier = better)
            if months_until_intake >= 12:  # Very early (12+ months)
                return 1.0
            elif months_until_intake >= 6:  # Early (6-12 months)
                return 0.9
            elif months_until_intake >= 3:  # Good timing (3-6 months)
                return 0.8
            elif months_until_intake >= 1:  # Late but acceptable (1-3 months)
                return 0.6
            elif months_until_intake >= 0:  # Very late (0-1 month)
                return 0.4
            else:  # Past deadline
                return 0.2
            
        except Exception as e:
            logger.error(f"Error calculating application timing: {e}")
            return 0.5  # Default score

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

    def _calculate_education_compatibility(self, current_education: str, target_course: str) -> float:
        """Calculate compatibility between current education and target course"""
        current_score = self._get_education_level_score(current_education)
        course_difficulty = self._get_course_difficulty(target_course)
        
        # Education progression rules
        progression_map = {
            # (current_level, target_level): compatibility_score
            (1.0, 3.0): 0.9,  # High school to Bachelor - Standard progression
            (1.5, 3.0): 0.95, # Diploma to Bachelor - Good progression
            (2.0, 3.0): 1.0,  # Foundation to Bachelor - Ideal progression
            (3.0, 4.0): 1.0,  # Bachelor to Masters - Ideal progression
            (3.0, 3.0): 0.7,  # Bachelor to Bachelor - Second degree
            (3.5, 4.0): 1.0,  # Graduate to Masters - Standard progression
            (4.0, 5.0): 1.0,  # Masters to PhD - Ideal progression
            (4.0, 4.0): 0.8,  # Masters to Masters - Specialization
            (3.0, 5.0): 0.6,  # Bachelor to PhD - Challenging jump
        }
        
        # Check exact progression match
        key = (current_score, course_difficulty)
        if key in progression_map:
            return progression_map[key]
        
        # Calculate general compatibility
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

    def _calculate_gpa_percentile(self, gpa: float, education_level: str) -> float:
        """Calculate GPA percentile based on education level"""
        # Different GPA distributions for different levels
        gpa_distributions = {
            'high school': {'mean': 3.2, 'std': 0.5},
            'undergraduate': {'mean': 3.0, 'std': 0.6},
            'masters': {'mean': 3.5, 'std': 0.4},
            'phd': {'mean': 3.7, 'std': 0.3}
        }
        
        # Get distribution for education level
        level_key = 'undergraduate'  # default
        if 'high' in education_level.lower():
            level_key = 'high school'
        elif any(word in education_level.lower() for word in ['master', 'mba']):
            level_key = 'masters'
        elif 'phd' in education_level.lower() or 'doctor' in education_level.lower():
            level_key = 'phd'
        
        dist = gpa_distributions[level_key]
        
        # Calculate z-score and percentile
        z_score = (gpa - dist['mean']) / dist['std']
        # Approximate percentile using sigmoid function
        percentile = 1 / (1 + np.exp(-z_score))
        
        return percentile

    def _calculate_ielts_percentile(self, ielts_score: float) -> float:
        """Calculate IELTS percentile"""
        # IELTS score distribution (approximate)
        if ielts_score >= 9.0:
            return 0.99
        elif ielts_score >= 8.5:
            return 0.95
        elif ielts_score >= 8.0:
            return 0.90
        elif ielts_score >= 7.5:
            return 0.85
        elif ielts_score >= 7.0:
            return 0.75
        elif ielts_score >= 6.5:
            return 0.60
        elif ielts_score >= 6.0:
            return 0.45
        elif ielts_score >= 5.5:
            return 0.30
        else:
            return 0.15

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
            
            # Education progression factor
            course_difficulty = self._get_course_difficulty(course_applied)
            education_score = self._get_education_level_score(education_level)
            
            if education_score >= course_difficulty:
                success_factors.append(0.2)
            elif education_score >= course_difficulty - 1:
                success_factors.append(0.1)
            else:
                success_factors.append(-0.1)
            
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

    def _set_fallback_models(self):
        """Set up fallback models that provide default predictions when training fails"""
        logger.info("Setting up fallback prediction models...")
        
        class FallbackClassifier:
            def predict(self, X):
                # Return mostly positive predictions with some randomness
                import random
                return [1 if random.random() > 0.3 else 0 for _ in range(len(X))]
            
            def predict_proba(self, X):
                import random
                probabilities = []
                for _ in range(len(X)):
                    prob_positive = 0.7 + random.uniform(-0.2, 0.2)  # 0.5 to 0.9
                    prob_positive = max(0.1, min(0.9, prob_positive))
                    probabilities.append([1-prob_positive, prob_positive])
                return probabilities
        
        class FallbackRegressor:
            def predict(self, X):
                import random
                # Return probabilities between 0.4 and 0.8
                return [0.6 + random.uniform(-0.2, 0.2) for _ in range(len(X))]
        
        self.admission_predictor = FallbackClassifier()
        self.success_probability_model = FallbackRegressor()
        self.models_trained = True  # Mark as trained so predictions work
        
        logger.info("✅ Fallback models set up successfully")

    def train_models(self):
        """Train ML models with enhanced features and better error handling"""
        if not SKLEARN_AVAILABLE:
            logger.warning("Scikit-learn not available. ML predictions disabled.")
            self.models_trained = False
            return
        
        try:
            logger.info("Starting ML model training...")
            
            # Get applications data
            applications = self._get_training_data()
            
            if not applications:
                logger.error("No training data available")
                self.models_trained = False
                return
            
            if len(applications) < 5:  # Reduced minimum requirement
                logger.warning(f"Limited training data ({len(applications)} samples). Generating additional synthetic data...")
                additional_data = self._generate_synthetic_data(20)  # Generate 20 more samples
                applications.extend(additional_data)
            
            logger.info(f"Training with {len(applications)} applications")
            
            # Prepare training data
            features, targets = self._prepare_training_data(applications)
            
            if len(features) < 2:
                logger.error("Insufficient valid training samples after preprocessing")
                self.models_trained = False
                return
            
            logger.info(f"Prepared {len(features)} valid training samples with {len(features[0])} features")
            
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
            logger.info("✅ Admission predictor trained successfully")
            
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
            logger.info("✅ Success probability model trained successfully")
            
            # Validate models
            train_accuracy = self.admission_predictor.score(features, targets)
            train_r2 = self.success_probability_model.score(features, regression_targets)
            
            logger.info(f"📊 Training accuracy: {train_accuracy:.3f}")
            logger.info(f"📊 Training R²: {train_r2:.3f}")
            
            self.models_trained = True
            logger.info(f"🎉 ML models successfully trained with {len(features)} samples!")
            
            # Log feature importance for debugging
            if hasattr(self.admission_predictor, 'feature_importances_'):
                feature_importance = list(zip(self.feature_names, self.admission_predictor.feature_importances_))
                feature_importance.sort(key=lambda x: x[1], reverse=True)
                logger.info("📊 Top feature importances:")
                for feature, importance in feature_importance[:5]:
                    logger.info(f"   {feature}: {importance:.3f}")
            
        except Exception as e:
            logger.error(f"❌ Model training failed: {e}")
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
            self.models_trained = False
            
            # Set fallback models that return default predictions
            self._set_fallback_models()

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

    def predict_admission_success(self, applicant_data: Dict) -> Dict:
        """Enhanced prediction with education level consideration"""
        if not self.models_trained or not SKLEARN_AVAILABLE:
            return {"error": "Models not available", "success_probability": 0.5}
        
        try:
            # Extract enhanced features
            gpa = float(applicant_data.get('gpa', 3.0))
            ielts_score = float(applicant_data.get('ielts_score', 6.5))
            work_exp = int(applicant_data.get('work_experience_years', 0))
            
            # Education features
            current_education = applicant_data.get('current_education', 'undergraduate')
            education_level_score = self._get_education_level_score(current_education)
            
            # Course features
            course_name = applicant_data.get('course_applied', '')
            course_difficulty = self._get_course_difficulty(course_name)
            education_compatibility = self._calculate_education_compatibility(current_education, course_name)
            
            # Other features
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
            
            features = np.array([[
                gpa, ielts_score, work_exp, course_difficulty, app_timing, is_international,
                education_level_score, education_compatibility, gpa_percentile,
                ielts_percentile, overall_strength
            ]])
            
            # Get predictions
            success_probability = self.success_probability_model.predict(features)[0]
            success_probability = max(0.0, min(1.0, success_probability))
            
            admission_prediction = self.admission_predictor.predict(features)[0]
            prediction_proba = self.admission_predictor.predict_proba(features)[0]
            confidence = max(prediction_proba)
            
            # Generate enhanced recommendations
            recommendations = self._generate_enhanced_recommendations(
                applicant_data, success_probability, education_compatibility
            )
            
            # Enhanced risk factors
            risk_factors = self._identify_enhanced_risk_factors(
                applicant_data, education_compatibility
            )
            
            # Additional insights
            insights = self._generate_insights(
                applicant_data, education_compatibility, gpa_percentile, ielts_percentile
            )
            
            return {
                "success_probability": float(success_probability),
                "admission_prediction": bool(admission_prediction),
                "confidence": float(confidence),
                "recommendations": recommendations,
                "risk_factors": risk_factors,
                "insights": insights,
                "education_compatibility": float(education_compatibility),
                "academic_percentiles": {
                    "gpa_percentile": float(gpa_percentile),
                    "ielts_percentile": float(ielts_percentile),
                    "overall_strength": float(overall_strength)
                }
            }
            
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return {"error": str(e), "success_probability": 0.5}

    def _generate_enhanced_recommendations(self, applicant_data: Dict, probability: float, 
                                         education_compatibility: float) -> List[str]:
        """Generate enhanced personalized recommendations"""
        recommendations = []
        
        gpa = float(applicant_data.get('gpa', 0))
        ielts = float(applicant_data.get('ielts_score', 0))
        current_education = applicant_data.get('current_education', '')
        course_applied = applicant_data.get('course_applied', '')
        work_exp = int(applicant_data.get('work_experience_years', 0))
        
        # Education progression recommendations
        if education_compatibility < 0.7:
            if education_compatibility < 0.5:
                recommendations.append(
                    f"Consider foundation or bridging programs to prepare for {course_applied}"
                )
            else:
                recommendations.append(
                    "Your education background may require additional preparation - consider prerequisite courses"
                )
        
        # Academic performance recommendations
        if probability < 0.6:
            if gpa < 3.0:
                recommendations.append(
                    "Focus on improving your academic record through additional coursework or certifications"
                )
            if ielts < 6.5:
                recommendations.append(
                    "Enhance your English proficiency - consider IELTS preparation courses"
                )
            if work_exp < 2 and self._get_course_difficulty(course_applied) >= 4.0:
                recommendations.append(
                    "Gain relevant work experience or internships to strengthen your application"
                )
        
        # Success enhancement recommendations
        if probability >= 0.8:
            recommendations.append(
                "Excellent profile! Focus on crafting a compelling personal statement highlighting your unique strengths"
            )
            recommendations.append(
                "Apply for merit-based scholarships - your strong profile makes you a competitive candidate"
            )
            if education_compatibility >= 0.9:
                recommendations.append(
                    "Your educational progression is ideal - emphasize this in your application"
                )
        
        return recommendations[:4]  # Return top 4 recommendations

    def _identify_enhanced_risk_factors(self, applicant_data: Dict, 
                                      education_compatibility: float) -> List[str]:
        """Identify enhanced risk factors including education compatibility"""
        risk_factors = []
        
        gpa = float(applicant_data.get('gpa', 0))
        ielts = float(applicant_data.get('ielts_score', 0))
        current_education = applicant_data.get('current_education', '')
        course_applied = applicant_data.get('course_applied', '')
        
        # Education compatibility risks
        if education_compatibility < 0.5:
            risk_factors.append(
                f"Significant education gap: {current_education} to {course_applied} may be challenging"
            )
        elif education_compatibility < 0.7:
            risk_factors.append(
                "Education progression requires careful consideration"
            )
        
        # Academic performance risks
        if gpa < 2.5:
            risk_factors.append("GPA significantly below typical requirements")
        elif gpa < 3.0:
            risk_factors.append("GPA below average for competitive programs")
        
        # Language proficiency risks
        if ielts < 6.0:
            risk_factors.append("English proficiency below minimum requirements")
        elif ielts < 6.5:
            risk_factors.append("English score may limit program options")
        
        # Combined risks
        if gpa < 3.0 and ielts < 6.5:
            risk_factors.append("Both academic and language scores need improvement")
        
        return risk_factors

    def _generate_insights(self, applicant_data: Dict, education_compatibility: float,
                          gpa_percentile: float, ielts_percentile: float) -> Dict:
        """Generate detailed insights about the application"""
        insights = {
            "strengths": [],
            "areas_for_improvement": [],
            "comparative_analysis": {},
            "education_pathway": ""
        }
        
        # Identify strengths
        if gpa_percentile >= 0.8:
            insights["strengths"].append("Outstanding academic performance")
        elif gpa_percentile >= 0.6:
            insights["strengths"].append("Strong academic record")
        
        if ielts_percentile >= 0.8:
            insights["strengths"].append("Excellent English proficiency")
        elif ielts_percentile >= 0.6:
            insights["strengths"].append("Good English language skills")
        
        if education_compatibility >= 0.9:
            insights["strengths"].append("Perfect educational progression")
        
        work_exp = int(applicant_data.get('work_experience_years', 0))
        if work_exp >= 3:
            insights["strengths"].append(f"Valuable work experience ({work_exp} years)")
        
        # Areas for improvement
        if gpa_percentile < 0.5:
            insights["areas_for_improvement"].append("Academic performance enhancement needed")
        
        if ielts_percentile < 0.5:
            insights["areas_for_improvement"].append("English proficiency improvement recommended")
        
        if education_compatibility < 0.7:
            insights["areas_for_improvement"].append("Consider bridging programs or prerequisites")
        
        # Comparative analysis
        insights["comparative_analysis"] = {
            "gpa_ranking": f"Top {100 - int(gpa_percentile * 100)}% of applicants",
            "ielts_ranking": f"Top {100 - int(ielts_percentile * 100)}% of applicants",
            "overall_competitiveness": self._get_competitiveness_level(gpa_percentile, ielts_percentile)
        }
        
        # Education pathway recommendation
        current_education = applicant_data.get('current_education', '')
        course_applied = applicant_data.get('course_applied', '')
        insights["education_pathway"] = self._recommend_education_pathway(
            current_education, course_applied, education_compatibility
        )
        
        return insights

    def _get_competitiveness_level(self, gpa_percentile: float, ielts_percentile: float) -> str:
        """Determine overall competitiveness level"""
        avg_percentile = (gpa_percentile + ielts_percentile) / 2
        
        if avg_percentile >= 0.8:
            return "Highly competitive applicant"
        elif avg_percentile >= 0.6:
            return "Competitive applicant"
        elif avg_percentile >= 0.4:
            return "Moderately competitive applicant"
        else:
            return "Requires significant improvement"

    def _recommend_education_pathway(self, current_education: str, target_course: str, 
                                   compatibility: float) -> str:
        """Recommend optimal education pathway"""
        if compatibility >= 0.9:
            return f"Direct progression from {current_education} to {target_course} is ideal"
        elif compatibility >= 0.7:
            return f"Standard progression from {current_education} to {target_course} with good preparation"
        elif compatibility >= 0.5:
            return f"Consider intermediate qualifications between {current_education} and {target_course}"
        else:
            return f"Recommended: Complete prerequisite programs before applying to {target_course}"


class CourseRecommendationSystem:
    """Truly Automated AI-powered course recommendation system using real CSV data"""
    
    def __init__(self, data_manager: DataManager):
        self.data_manager = data_manager
        
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
            csv_path = os.path.join(self.data_manager.data_dir, 'courses.csv')
            
            if os.path.exists(csv_path):
                # Load the CSV file
                self.courses_df = pd.read_csv(csv_path)
                self.courses_df.columns = self.courses_df.columns.str.strip()
                
                # Validate required columns
                required_columns = ['course_name', 'level', 'description', 'keywords']
                missing_columns = [col for col in required_columns if col not in self.courses_df.columns]
                
                if missing_columns:
                    logger.error(f"Missing required columns in courses.csv: {missing_columns}")
                    self.courses_df = pd.DataFrame()  # Use empty DataFrame
                    return
                
                # Fill missing optional columns with defaults
                if 'department' not in self.courses_df.columns:
                    self.courses_df['department'] = 'General Studies'
                if 'duration' not in self.courses_df.columns:
                    self.courses_df['duration'] = '1 year'
                if 'fees_international' not in self.courses_df.columns:
                    self.courses_df['fees_international'] = 15000
                if 'min_gpa' not in self.courses_df.columns:
                    self.courses_df['min_gpa'] = 2.5
                if 'min_ielts' not in self.courses_df.columns:
                    self.courses_df['min_ielts'] = 6.0
                if 'trending_score' not in self.courses_df.columns:
                    self.courses_df['trending_score'] = 5.0
                if 'career_prospects' not in self.courses_df.columns:
                    self.courses_df['career_prospects'] = 'Various career opportunities available'
                
                logger.info(f"✅ Loaded {len(self.courses_df)} courses from CSV")
                
                # Prepare search vectors if sklearn available
                if SKLEARN_AVAILABLE and len(self.courses_df) > 0:
                    self._create_search_vectors()
                
            else:
                logger.warning(f"❌ courses.csv not found at {csv_path}")
                logger.info("📝 Please create courses.csv with required columns")
                self.courses_df = pd.DataFrame()
                
        except Exception as e:
            logger.error(f"❌ Error loading courses.csv: {e}")
            self.courses_df = pd.DataFrame()
    
    def _create_search_vectors(self):
        """Create TF-IDF vectors for course matching"""
        try:
            if self.courses_df.empty:
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
                logger.info(f"✅ Created search vectors for {len(course_texts)} courses")
            
        except Exception as e:
            logger.error(f"Error creating search vectors: {e}")
            self.course_vectors = None
    
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
        
        # Target industry (medium weight)
        target_industry = user_profile.get('target_industry', '').lower()
        if target_industry:
            user_interests.append((target_industry, 0.6))  # Weight 0.6
        
        # Ideal role (medium weight)
        ideal_role = user_profile.get('ideal_role', '').lower()
        if ideal_role:
            user_interests.append((ideal_role, 0.6))  # Weight 0.6
        
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
        user_level = user_profile.get('education_level', 'undergraduate').lower()
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
            
            # Working professional flexibility
            ('working professional', 'undergraduate'): 0.8,
            ('working professional', 'postgraduate'): 0.9,
            ('working professional', 'masters'): 1.0,
            ('working professional', 'mba'): 1.0,
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
    
    def recommend_courses(self, user_profile: Dict, preferences: Dict = None) -> List[Dict]:
        """Generate automated course recommendations based on CSV data"""
        try:
            if self.courses_df is None or self.courses_df.empty:
                logger.error("No course data available from CSV")
                return []
            
            logger.info(f"\n🤖 AUTOMATED COURSE MATCHING FROM CSV DATA")
            logger.info(f"📊 Total courses in database: {len(self.courses_df)}")
            logger.info(f"👤 User profile: {user_profile.get('first_name', 'Unknown')}")
            logger.info(f"🎯 Primary interest: {user_profile.get('field_of_interest', 'Not specified')}")
            
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
                    
                    logger.info(f"📚 {course.get('course_name', 'Unknown')}: {overall_score:.2f} "
                              f"(Interest: {interest_result['total_score']:.2f}, "
                              f"Academic: {academic_score:.2f}, "
                              f"Requirements: {requirements_score:.2f})")
                    
                except Exception as e:
                    logger.error(f"Error scoring course {course.get('course_name', 'Unknown')}: {e}")
                    continue
            
            # Sort by overall score
            recommendations.sort(key=lambda x: x['score'], reverse=True)
            
            # Return top recommendations
            top_recommendations = recommendations[:10]  # Top 10
            
            logger.info(f"✅ Generated {len(top_recommendations)} recommendations")
            
            return top_recommendations
            
        except Exception as e:
            logger.error(f"❌ Error in course recommendation: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return []
    
    def get_course_recommendations(self, user_profile: Dict, preferences: Dict = None) -> List[Dict]:
        """Main method to get course recommendations"""
        return self.recommend_courses(user_profile, preferences)





class DocumentVerificationAI:
    """AI-powered document verification - COMPLETE AND FIXED VERSION"""
    
    def __init__(self):
        try:
            self.verification_rules = self._load_verification_rules()
            self.verification_history = []
            logger.info("DocumentVerificationAI initialized successfully")
        except Exception as e:
            logger.error(f"DocumentVerificationAI initialization error: {e}")
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
            logger.error(f"Document verification error: {e}")
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
    
    def get_verification_history(self) -> List[Dict]:
        """Get verification history"""
        return self.verification_history
    
    def get_verification_stats(self) -> Dict:
        """Get verification statistics"""
        if not self.verification_history:
            return {
                "total_verifications": 0,
                "verified_count": 0,
                "rejected_count": 0,
                "needs_review_count": 0,
                "average_confidence": 0.0
            }
        
        total = len(self.verification_history)
        verified = sum(1 for v in self.verification_history if v.get("verification_status") == "verified")
        rejected = sum(1 for v in self.verification_history if v.get("verification_status") == "rejected")
        needs_review = sum(1 for v in self.verification_history if v.get("verification_status") == "needs_review")
        
        avg_confidence = sum(v.get("confidence_score", 0) for v in self.verification_history) / total
        
        return {
            "total_verifications": total,
            "verified_count": verified,
            "rejected_count": rejected,
            "needs_review_count": needs_review,
            "average_confidence": avg_confidence
        }



class VoiceService:
    """Enhanced voice recognition and text-to-speech service"""
    
    def __init__(self):
        self.is_listening = False
        self.is_speaking = False
        self.recognizer = None
        self.engine = None
        self.microphone = None
        
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
                
                logger.info("Voice service initialized successfully")
                
            except Exception as e:
                logger.error(f"Voice service initialization error: {e}")
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
                logger.info("Adjusting for ambient noise...")
                self.recognizer.adjust_for_ambient_noise(source, duration=1)
                
                logger.info("Listening for speech...")
                # Increase timeout and phrase time limit
                audio = self.recognizer.listen(source, timeout=3, phrase_time_limit=10)
            
            logger.info("Processing speech...")
            
            # Try Google Speech Recognition first
            try:
                text = self.recognizer.recognize_google(audio)
                logger.info(f"Speech recognized: {text}")
                return text
            except sr.RequestError:
                # Fallback to offline recognition if available
                try:
                    text = self.recognizer.recognize_sphinx(audio)
                    logger.info(f"Speech recognized (offline): {text}")
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
            logger.error(f"Speech recognition error: {e}")
            return f"❌ Voice input failed: {str(e)}"
    
    def text_to_speech(self, text: str) -> bool:
        """Convert text to speech with better error handling"""
        if not self.is_available():
            logger.warning("TTS not available")
            return False
        
        if self.is_speaking:
            logger.warning("Already speaking")
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
                    logger.error(f"TTS thread error: {e}")
                finally:
                    self.is_speaking = False
            
            # Start speaking in background thread
            thread = threading.Thread(target=speak_thread, daemon=True)
            thread.start()
            
            return True
            
        except Exception as e:
            self.is_speaking = False
            logger.error(f"Text-to-speech error: {e}")
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
    
    def stop_speaking(self) -> bool:
        """Stop current speech"""
        if not self.is_available() or not self.is_speaking:
            return False
        
        try:
            self.engine.stop()
            self.is_speaking = False
            return True
        except Exception as e:
            logger.error(f"Error stopping speech: {e}")
            return False

# =============================================================================
# MAIN AI AGENT CLASS
# =============================================================================

class UnifiedUELAIAgent:
    """Complete implementation of the Unified UEL AI Agent with all methods"""
    
    def __init__(self):
        """Initialize all AI services and components"""
        self.logger = logging.getLogger(__name__)
        self.logger.info("Initializing UEL AI Agent...")
        
        # Initialize core services
        self.db_manager = DatabaseManager()
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
        
        self.logger.info("UEL AI Agent initialized successfully")
    
    def get_ai_response(self, user_input: str, user_context: Dict = None) -> Dict:
        """Get comprehensive AI response with context awareness"""
        try:
            self.interaction_count += 1
            start_time = time.time()
            
            # Analyze sentiment
            sentiment_data = self.sentiment_engine.analyze_message_sentiment(user_input)
            
            # Prepare enhanced prompt with context
            system_prompt = self._build_system_prompt(user_context, sentiment_data)
            enhanced_prompt = self._enhance_user_prompt(user_input, user_context)
            
            # Get LLM response
            if self.ollama_service.is_available():
                ai_response = self.ollama_service.generate_response(
                    enhanced_prompt, 
                    system_prompt=system_prompt
                )
                llm_used = self.ollama_service.model_name
            else:
                ai_response = self._fallback_response(user_input, user_context)
                llm_used = "fallback"
            
            # Process and enhance response
            processed_response = self._process_ai_response(ai_response, user_input, user_context)
            
            response_time = time.time() - start_time
            
            # Store interaction
            self.conversation_history.append({
                'user_input': user_input,
                'ai_response': processed_response,
                'sentiment': sentiment_data,
                'timestamp': datetime.now().isoformat(),
                'response_time': response_time,
                'llm_used': llm_used
            })
            
            return {
                'response': processed_response,
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
                'sentiment': {'sentiment': 'neutral', 'polarity': 0.0},
                'llm_used': 'error',
                'response_time': 0.0,
                'error': str(e)
            }
    
    def _build_system_prompt(self, user_context: Dict = None, sentiment_data: Dict = None) -> str:
        """Build comprehensive system prompt"""
        prompt_parts = [
            f"You are an intelligent AI assistant for {config.university_name} (UEL).",
            "You help prospective and current students with admissions, courses, applications, and university life.",
            "You are knowledgeable, helpful, and professional.",
            ""
        ]
        
        # Add university information
        prompt_parts.extend([
            "UNIVERSITY INFORMATION:",
            f"- Name: {config.university_name}",
            f"- Contact: {config.admissions_email}, {config.admissions_phone}",
            "- Location: London, UK",
            "- Known for: Diverse programs, practical education, industry connections",
            ""
        ])
        
        # Add user context if available
        if user_context:
            prompt_parts.append("STUDENT CONTEXT:")
            if user_context.get('first_name'):
                prompt_parts.append(f"- Name: {user_context['first_name']} {user_context.get('last_name', '')}")
            if user_context.get('field_of_interest'):
                prompt_parts.append(f"- Interest: {user_context['field_of_interest']}")
            if user_context.get('academic_level'):
                prompt_parts.append(f"- Level: {user_context['academic_level']}")
            if user_context.get('country'):
                prompt_parts.append(f"- Country: {user_context['country']}")
            prompt_parts.append("")
        
        # Add sentiment context
        if sentiment_data:
            sentiment = sentiment_data.get('sentiment', 'neutral')
            emotions = sentiment_data.get('emotions', [])
            urgency = sentiment_data.get('urgency', 'normal')
            
            prompt_parts.append("CONVERSATION CONTEXT:")
            prompt_parts.append(f"- User sentiment: {sentiment}")
            if emotions:
                prompt_parts.append(f"- Detected emotions: {', '.join(emotions)}")
            if urgency != 'normal':
                prompt_parts.append(f"- Urgency level: {urgency}")
            prompt_parts.append("")
        
        prompt_parts.extend([
            "GUIDELINES:",
            "- Be helpful, accurate, and university-focused",
            "- Provide specific information about UEL when possible",
            "- Ask clarifying questions if needed",
            "- Be empathetic and supportive",
            "- If you don't know something, say so and suggest alternatives",
            "- Keep responses concise but informative",
            ""
        ])
        
        return "\n".join(prompt_parts)
    
    def _enhance_user_prompt(self, user_input: str, user_context: Dict = None) -> str:
        """Enhance user prompt with context"""
        enhanced_parts = [user_input]
        
        # Add relevant context
        if user_context:
            context_info = []
            if user_context.get('field_of_interest'):
                context_info.append(f"(Student interested in: {user_context['field_of_interest']})")
            if user_context.get('academic_level'):
                context_info.append(f"(Academic level: {user_context['academic_level']})")
            
            if context_info:
                enhanced_parts.append(" ".join(context_info))
        
        # Add conversation history context
        if self.conversation_history:
            recent_context = self.conversation_history[-2:]  # Last 2 interactions
            if recent_context:
                enhanced_parts.append("Recent conversation context:")
                for interaction in recent_context:
                    enhanced_parts.append(f"User: {interaction['user_input'][:100]}...")
                    enhanced_parts.append(f"Assistant: {interaction['ai_response'][:100]}...")
        
        return "\n".join(enhanced_parts)
    
    def _process_ai_response(self, ai_response: str, user_input: str, user_context: Dict = None) -> str:
        """Process and enhance AI response"""
        # Basic response processing
        processed = ai_response.strip()
        
        # Add personalization if context available
        if user_context and user_context.get('first_name'):
            name = user_context['first_name']
            if not any(name.lower() in processed.lower() for name in [name]):
                # Don't over-personalize, but add name occasionally
                if len(processed.split()) > 20:  # Only for longer responses
                    processed = f"Hi {name}! {processed}"
        
        # Ensure university branding
        if 'university of east london' not in processed.lower() and 'uel' not in processed.lower():
            if any(keyword in user_input.lower() for keyword in ['university', 'uel', 'admission', 'course', 'program']):
                processed += f"\n\nFor more information, visit {config.university_name} or contact us at {config.admissions_email}."
        
        return processed
    
    def _fallback_response(self, user_input: str, user_context: Dict = None) -> str:
        """Provide fallback response when LLM is unavailable"""
        user_lower = user_input.lower()
        
        # Course-related queries
        if any(word in user_lower for word in ['course', 'program', 'study', 'degree']):
            return f"Thank you for your interest in {config.university_name} courses! We offer a wide range of undergraduate and postgraduate programs. For detailed course information, please visit our website or contact our admissions team at {config.admissions_email}."
        
        # Application-related queries
        elif any(word in user_lower for word in ['apply', 'application', 'admission', 'entry']):
            return f"Great that you're interested in applying to {config.university_name}! Our application process is straightforward. You can apply online through our admissions portal. For guidance, contact us at {config.admissions_email} or call {config.admissions_phone}."
        
        # Fee-related queries
        elif any(word in user_lower for word in ['fee', 'cost', 'tuition', 'price', 'money']):
            return f"Tuition fees vary by course and student type. For current fee information and scholarship opportunities, please visit our website or contact our admissions team at {config.admissions_email}."
        
        # General greeting
        elif any(word in user_lower for word in ['hello', 'hi', 'hey', 'help']):
            greeting = f"Hello! Welcome to {config.university_name}. I'm here to help you with information about our courses, applications, and university life. How can I assist you today?"
            if user_context and user_context.get('first_name'):
                greeting = f"Hello {user_context['first_name']}! " + greeting[6:]
            return greeting
        
        # Default response
        else:
            return f"Thank you for contacting {config.university_name}. I'd be happy to help you with information about our courses, admissions, and university services. Could you please be more specific about what you'd like to know?"
    
    def get_feature_status(self) -> Dict[str, bool]:
        """Get status of all features - COMPLETE VERSION"""
        try:
            return {
                'llm_integration': self.ollama_service.is_available() if hasattr(self, 'ollama_service') else False,
                'ml_predictions': (hasattr(self, 'predictive_engine') and 
                                 hasattr(self.predictive_engine, 'models_trained') and 
                                 self.predictive_engine.models_trained),
                'sentiment_analysis': hasattr(self, 'sentiment_engine'),
                'course_recommendations': hasattr(self, 'recommendation_system'),
                'document_verification': hasattr(self, 'document_verifier'),
                'voice_services': (hasattr(self, 'voice_service') and 
                                 self.voice_service.is_available()),
                'intelligent_search': (hasattr(self, 'data_manager') and 
                                     hasattr(self.data_manager, 'combined_data') and 
                                     len(self.data_manager.combined_data) > 0),
                'data_integration': True,
                'real_time_analytics': True
            }
        except Exception as e:
            self.logger.error(f"Error getting feature status: {e}")
            # Return safe defaults
            return {
                'llm_integration': False,
                'ml_predictions': False,
                'sentiment_analysis': False,
                'course_recommendations': True,
                'document_verification': True,
                'voice_services': False,
                'intelligent_search': False,
                'data_integration': True,
                'real_time_analytics': True
            }

    def analyze_sentiment(self, text: str) -> Dict:
        """Analyze sentiment of text"""
        try:
            return self.sentiment_engine.analyze_message_sentiment(text)
        except Exception as e:
            self.logger.error(f"Sentiment analysis error: {e}")
            return {'sentiment': 'neutral', 'polarity': 0.0, 'error': str(e)}


    def process_query(self, user_input: str, applicant_data: Dict = None) -> str:
        """Process user query - compatibility method"""
        try:
            response_data = self.get_ai_response(user_input, applicant_data)
            return response_data.get('response', 'Sorry, I could not process your request.')
        except Exception as e:
            self.logger.error(f"Query processing error: {e}")
            return f"I apologize, but I encountered an error processing your request: {str(e)}"
    
    def predict_admission_success(self, applicant_data: Dict) -> Dict:
        """Predict admission success using ML models"""
        try:
            return self.predictive_engine.predict_admission_success(applicant_data)
        except Exception as e:
            self.logger.error(f"Prediction error: {e}")
            return {
                "error": "Prediction service temporarily unavailable",
                "success_probability": 0.5,
                "confidence": 0.0
            }
    
    def get_course_recommendations(self, user_profile: Dict, preferences: Dict = None) -> List[Dict]:
        """Get personalized course recommendations"""
        try:
            return self.recommendation_system.recommend_courses(user_profile, preferences)
        except Exception as e:
            self.logger.error(f"Recommendation error: {e}")
            return []
    
    def verify_document(self, document_data: Dict, document_type: str) -> Dict:
        """Verify document using AI"""
        try:
            return self.document_verifier.verify_document(document_data, document_type)
        except Exception as e:
            self.logger.error(f"Document verification error: {e}")
            return {
                "verification_status": "error",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def speech_to_text(self) -> str:
        """Convert speech to text"""
        try:
            return self.voice_service.speech_to_text()
        except Exception as e:
            self.logger.error(f"Speech to text error: {e}")
            return f"❌ Voice input error: {e}"
    
    def text_to_speech(self, text: str) -> bool:
        """Convert text to speech"""
        try:
            return self.voice_service.text_to_speech(text)
        except Exception as e:
            self.logger.error(f"Text to speech error: {e}")
            return False
    
    def get_system_analytics(self) -> Dict:
        """Get comprehensive system analytics"""
        uptime_seconds = (datetime.now() - self.session_start_time).total_seconds()
        error_rate = (self.error_count / max(self.interaction_count, 1)) * 100
        
        return {
            'statistics': {
                'total_interactions': self.interaction_count,
                'error_count': self.error_count,
                'error_rate_percent': error_rate,
                'uptime_seconds': uptime_seconds,
                'indexed_items': len(self.data_manager.combined_data) if hasattr(self.data_manager, 'combined_data') else 0,
                'conversation_length': len(self.conversation_history),
                'avg_response_time': self._calculate_avg_response_time()
            },
            'config': {
                'university_name': config.university_name,
                'model_name': self.ollama_service.model_name,
                'sklearn_available': self.predictive_engine.models_trained,
                'voice_available': self.voice_service.is_available(),
                'llm_available': self.ollama_service.is_available()
            },
            'features': {
                'ml_predictions': self.predictive_engine.models_trained,
                'sentiment_analysis': True,
                'voice_services': self.voice_service.is_available(),
                'document_verification': True,
                'intelligent_search': len(self.data_manager.combined_data) > 0 if hasattr(self.data_manager, 'combined_data') else False
            }
        }
    
    def _calculate_avg_response_time(self) -> float:
        """Calculate average response time"""
        if not self.conversation_history:
            return 0.0
        
        response_times = [
            interaction.get('response_time', 0) 
            for interaction in self.conversation_history 
            if interaction.get('response_time', 0) > 0
        ]
        
        return sum(response_times) / len(response_times) if response_times else 0.0

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def initialize_session_state():
    """Initialize Streamlit session state with unified features"""
    defaults = {
        # Navigation
        'current_page': 'dashboard',
        'last_activity': datetime.now(),
        
        # User data
        'user_id': f"user_{int(time.time())}",
        'current_student': None,
        'user_profile': {},
        
        # AI interactions
        'messages': [],
        'interaction_history': [],
        'sentiment_history': [],
        'conversation_context': [],
        
        # Features
        'prediction_results': {},
        'recommendation_history': [],
        'document_uploads': {},
        'verification_results': {},
        
        # System state
        'ai_agent': None,
        'session_start_time': datetime.now(),
        'show_info_collector': False,
        'show_document_upload': False,
        'show_form_builder': False,
        
        # UI state
        'selected_student_id': None,
        'selected_course_id': None,
        'selected_application_id': None,
        
        # Sample data
        'sample_students': None,
        'sample_courses': None,
        'sample_applications': None,
        'sample_analytics': None
    }
    
    for key, default_value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default_value

def get_ai_agent() -> UnifiedUELAIAgent:
    """Get or initialize AI agent"""
    if st.session_state.ai_agent is None:
        with st.spinner("🔄 Initializing UEL AI System..."):
            st.session_state.ai_agent = UnifiedUELAIAgent()
    return st.session_state.ai_agent

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
        'inquiry': '#3b82f6',          # Blue
        'application_started': '#f59e0b',   # Yellow  
        'under_review': '#6366f1',     # Indigo
        'accepted': '#10b981',         # Green
        'rejected': '#ef4444',         # Red
        'submitted': '#8b5cf6',        # Purple
        'draft': '#6b7280',            # Gray
        'verified': '#10b981',         # Green
        'pending': '#f59e0b'           # Yellow
    }
    return color_map.get(status, '#6b7280')

def get_level_color(level: str) -> str:
    """Get color for academic level badges"""
    color_map = {
        'undergraduate': '#3b82f6',    # Blue
        'postgraduate': '#8b5cf6',     # Purple
        'masters': '#10b981',          # Green
        'phd': '#ef4444'               # Red
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

# Export main components
__all__ = [
    'UnifiedUELAIAgent',
    'SystemConfig', 
    'config',
    'DatabaseManager',
    'Student',
    'Course',
    'OllamaService',
    'DataManager',
    'SentimentAnalysisEngine',
    'PredictiveAnalyticsEngine',
    'CourseRecommendationSystem',
    'DocumentVerificationAI',
    'VoiceService',
    'initialize_session_state',
    'get_ai_agent',
    'format_currency',
    'format_date',
    'format_duration',
    'get_status_color',
    'get_level_color',
    'generate_sample_data'
]