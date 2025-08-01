# 🚀 UEL Enhanced AI Assistant - Profile-Driven Experience

Version 3.0 - A revolutionary AI-powered university admission assistant designed to provide a personalized, seamless, and intelligent experience for prospective students. By integrating a mandatory user profile, this system ensures that every interaction, recommendation, and prediction is uniquely tailored to the user's background and aspirations.

🌟 What's New in Version 3.0: The Profile-First Revolution
This version marks a significant leap forward by placing the user's profile at the core of the entire system.

🎯 Profile-First Experience
Mandatory Profile Creation: Users are guided through a comprehensive profile setup process upon their first visit. Access to advanced features is unlocked and enhanced by a complete profile.
Seamless Integration: All features dynamically pull information from the user's profile, eliminating the need for repetitive data entry.
No Re-entry: Once your profile is created and updated, the system automatically leverages this data across all functionalities.
Personalized Everything: AI responses, course recommendations, admission predictions, and even interview preparation are deeply personalized based on your unique profile data.

🔗 Interconnected Features
The system is designed as a cohesive ecosystem where each feature enhances the others through shared profile data:
Create Profile Once: Your single source of truth for all interactions.
AI Chat: The AI assistant understands your context, academic level, and goals without you having to repeat them.
Course Recommendations: Generates highly relevant course suggestions by analyzing your complete profile.
Admission Predictions: Automatically uses your academic and professional data to provide accurate probability assessments.
Cross-Feature Intelligence: Interactions within one feature (e.g., saving a preferred course) can inform and refine other features (e.g., future recommendations).

🚀 Key Features & Capabilities

👤 Comprehensive Profile Management
A robust system to capture and manage detailed user information:
Basic Information: Name, email, phone, date of birth, country, nationality, city, postal code.
Academic Background: Current education level, field of interest, current institution, major, GPA, graduation year, IELTS/TOEFL scores, other English certifications, previous applications, rejected courses.
Professional Background: Years of work experience, current job title, target industry, professional skills.
Interests & Preferences: Personal interests, career goals, preferred study mode (full-time, part-time, online), preferred start date, budget range, preferred courses.
System Data: Tracks profile creation/update dates, last active timestamp, and calculates profile completion percentage.
AI Interaction History: Logs interaction count, favorite features, and AI preferences to further tailor the experience.

🤖 Profile-Aware AI Chat Assistant
Contextual Conversations: The AI remembers your name, academic background, and interests, providing highly relevant and personalized responses.
Intelligent Guidance: Offers specific advice on applications, courses, and university services, tailored to your profile's details.
Sentiment Analysis: Understands the emotional tone of your messages (positive, negative, neutral, urgency, emotions) to respond empathetically.
Voice Integration: Supports both speech-to-text for input and text-to-speech for AI responses, offering a natural conversational experience (if voice libraries are available).

🎯 Advanced Course Recommendation System
Leverages multiple machine learning approaches for highly accurate and diverse recommendations:
Content-Based Filtering: Matches courses based on your stated interests, academic background, and career goals.
BERT-based Semantic Similarity: Utilizes advanced natural language processing to understand the deeper meaning of your profile and course descriptions, finding semantically similar matches.
Neural Collaborative Filtering (Placeholder): Designed to learn from user-item interactions (e.g., what similar students liked) to suggest courses.
Ensemble Modeling: Combines the strengths of various models (content-based, BERT, neural) with weighted scores for robust recommendations.
Explainable AI (XAI): Provides reasons why a particular course is recommended, highlighting matching factors and feature importance (e.g., "Matches your Computer Science interest," "High GPA fit").
Diversity & Novelty Metrics: Ensures recommendations are not only relevant but also diverse (offering variety) and novel (suggesting courses you might not have considered).

🔮 Intelligent Admission Probability Prediction
Provides data-driven insights into your chances of admission:
Automated Data Input: Automatically uses your GPA, IELTS/TOEFL scores, academic level, and work experience from your profile.
Machine Learning Models: Employs RandomForestClassifier for admission prediction and GradientBoostingRegressor for success probability, trained on historical application data.
Key Factor Analysis: Identifies and explains the primary factors influencing your prediction (e.g., "Excellent academic performance," "Strong English proficiency").
Personalized Recommendations: Offers actionable advice to improve your admission chances based on your current profile and the prediction outcome (e.g., "Consider retaking IELTS," "Strengthen your personal statement").
Fallback Models: Ensures predictions are still provided even if advanced ML models cannot be trained due to insufficient data, using rule-based logic.

📄 AI-Powered Document Verification
Streamlines the document submission process with intelligent checks:
Automated Checks: Verifies required fields and format requirements for various document types (transcripts, IELTS certificates, passports, personal statements, reference letters).
Confidence Scoring: Provides a confidence score for each verification, indicating the AI's certainty.
Issue Identification: Highlights missing information or potential issues in uploaded documents.
Actionable Recommendations: Offers clear suggestions for addressing identified issues and improving document quality for successful submission.

🎤 Interview Preparation Center (NEW!)
An AI-powered mock interview system to help you ace your admissions interviews:
Personalized Mock Interviews: Creates tailored interview sessions based on your academic level, field of interest, and chosen interview type (undergraduate, postgraduate, subject-specific, scholarship).
Dynamic Question Banks: Draws from comprehensive question banks covering general, behavioral, and subject-specific areas (Computer Science, Business, Engineering, Psychology).
AI Response Analysis: Evaluates your interview responses for relevance, coherence, content quality, depth, clarity, and confidence.
Detailed Feedback: Provides specific strengths, areas for improvement, and actionable suggestions for each answer.
Performance Reports: Generates an overall performance score, grade, and summary of your mock interview, including communication, relevance, and confidence scores.
Interview History: Keeps a record of all your mock interview sessions, allowing you to track progress over time.
Interview Tips & Guidance: Offers general and question-specific tips to help you prepare effectively.

📈 Analytics Dashboard
Provides insights into system usage and data:
System Status: Real-time overview of AI component availability (LLM, Voice, ML models, Data).
Data Overview: Summaries of loaded courses, applications, and FAQs, including distributions by academic level, department, and application status.
User Activity: Tracks your personal interaction count, profile completion, and favorite features.
Research Evaluation Framework (A+ Grade Feature):
Comprehensive Evaluation: Conducts academic-grade evaluation of recommendation and prediction systems using metrics like Precision@K, Recall@K, NDCG, MSE, MAE, and AUC-ROC.
Baseline Comparison: Statistically compares the performance of advanced models against various baseline recommendation methods (random, popularity, content-based, collaborative).
Bias Analysis: Analyzes potential biases in recommendations across different demographic groups (e.g., country, academic level) to ensure fairness.
Statistical Significance: Calculates p-values and effect sizes to validate the significance of performance improvements.
User Experience & System Performance Metrics: Tracks simulated UX metrics (e.g., session duration, adoption rate) and system performance (e.g., response times, memory usage).
Research Report Generation: Compiles a detailed academic report summarizing all evaluation findings.

📚 Dataset Information
The system utilizes a combination of real-world inspired and synthetic datasets to power its features:
applications.csv: Contains dummy data representing student applications, including applicant details, courses applied for, status, GPA, IELTS scores, nationality, and work experience. This dataset is primarily used for training the Admission Probability Prediction model and populating the Applications Analytics.
courses.csv: Features a mix of dummy data and information inspired by top 50 universities' course catalogs. This includes course names, codes, departments, academic levels, durations, domestic and international fees, descriptions, minimum GPA/IELTS requirements, trending scores, keywords, and career prospects. This dataset is crucial for Course Recommendations and general course information.
faqs.csv: Comprises dummy data for frequently asked questions and their answers, used by the AI Chat Assistant for quick information retrieval.
counseling_slots.csv: Contains dummy data for simulated counseling slot availability.
Note on Data Origin: While some data points (especially for courses) are inspired by real university offerings to ensure realism in structure and content, the specific values and student records are largely dummy or synthetically generated. This approach allows for robust testing and demonstration of the AI's capabilities without using actual sensitive personal or institutional data.

🛠️ Installation & Setup
Prerequisites

Python 3.8+
Ollama (for advanced AI features, specifically the deepseek:latest model)
Quick Start

Clone the Repository

bash

Run
Copy code
1git clone <repository-url>
2cd uel-enhanced-ai-assistant
Setup Enhanced System (Creates Data, DB, Config, Docs) This command will create the necessary data/ directory structure, initialize the SQLite database, generate sample CSV files, create configuration files, and update documentation.

bash

Run
Copy code
1python main.py --setup-enhanced
Install Python Dependencies The --setup-enhanced command generates a requirements.txt (or requirements_aplus.txt for advanced ML).

bash

Run
Copy code
1pip install -r requirements.txt
2# OR if you want all advanced ML features (recommended for A+ grade evaluation)
3# pip install -r requirements_aplus.txt
Start Ollama (if not already running) Ensure the Ollama server is running in the background.

bash

Run
Copy code
1ollama serve
Pull the Required LLM Model The system is configured to use deepseek:latest by default.

bash

Run
Copy code
1ollama pull deepseek:latest
Run the Streamlit Application

bash

Run
Copy code
1python main.py --run
Access the Application Open your web browser and navigate to: http://localhost:8501

You will be guided through the mandatory profile setup on your first visit.

📋 Profile Setup Process
The profile setup is a guided, multi-step process designed to be quick yet comprehensive, ensuring the AI has sufficient data to personalize your experience.

Step 1: Basic Information (1 min)
Your name, email, phone, date of birth.
Country of residence, nationality, city, and postal code.

Step 2: Academic Background (1 min)
Current academic level (e.g., High School, Undergraduate, Master's).
Current/previous institution, major, expected/actual graduation year.
Your GPA/Grade Average, IELTS, TOEFL, or other English certification scores.

Step 3: Interests & Goals (2 min)
Your primary field of interest (e.g., Computer Science, Business).
Additional areas of interest (multi-select).
Detailed career goals and aspirations.
Target industry and key professional skills.

Step 4: Preferences (1 min)
Preferred study mode (full-time, part-time, online).
Budget range for tuition fees.
Preferred start date for studies.
Notification preferences (email, AI personalization, data analytics).

Step 5: Review & Complete (30 sec)
Review all the information you've provided.
Accept terms and complete your profile.
Your profile completion percentage will be calculated and displayed.

🔗 Feature Integration Flow

Run
Copy code
1Profile Creation
2       ↓
3┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
4│   AI Chat       │←→  │ Recommendations │←→  │  Predictions    │←→  │ Interview Prep  │
5│                 │    │                 │    │                 │    │                 │
6│ • Knows your    │    │ • Uses complete │    │ • Auto-fills    │    │ • Tailored      │
7│   name & goals  │    │   profile data  │    │   your data     │    │   questions     │
8│ • References    │    │ • Matches your  │    │ • Only asks for │    │ • Personalized  │
9│   your interests│    │   interests     │    │   target course │    │   feedback      │
10│ • Personalized  │    │ • Considers     │    │ • Tracks your   │    │ • Tracks        │
11│   responses     │    │   your level    │    │   improvements  │    │   performance   │
12└─────────────────┘    └─────────────────┘    └─────────────────┘    └─────────────────┘
13       ↑                                                                       ↑
14       └────────────────────────── Profile Data Flows Automatically ────────────┘
💡 User Experience Highlights

Before (Version 2.0 - Generic AI)

❌ Fill out forms repeatedly across different features. ❌ Re-enter the same personal and academic information multiple times. ❌ Receive generic AI responses that lack personal context. ❌ Get basic course recommendations without deep personalization. ❌ Manual data entry required for every admission prediction.
After (Version 3.0 - Profile-Driven AI)

✅ One-time, guided profile setup that serves as your central data hub. ✅ Automatic data flow across all features, saving time and effort. ✅ Personalized AI that understands your unique background, interests, and goals. ✅ Smart, highly relevant recommendations based on your complete profile. ✅ Instant, accurate predictions leveraging your pre-filled academic data. ✅ AI-powered mock interviews with feedback tailored to your profile.

📊 Technical Architecture

Profile Management System

UserProfile (Dataclass): A comprehensive data structure defining all aspects of a user's profile, including basic info, academic background, professional experience, interests, preferences, and system-tracked data.
ProfileManager: Handles the lifecycle of user profiles, including creation, updates, retrieval, validation, completion tracking, and persistence to the database. It also manages an in-memory cache for quick access.

# Core AI Services

OllamaService: Manages interaction with the local Ollama LLM server, providing AI response generation with robust error handling and fallback mechanisms.
SentimentAnalysisEngine: Analyzes the emotional tone and urgency of user messages.
DocumentVerificationAI: Simulates AI-powered document verification based on predefined rules and provides feedback.
VoiceService: Integrates speech-to-text and text-to-speech capabilities for voice interactions.
Machine Learning Components

AdvancedCourseRecommendationSystem: Implements sophisticated recommendation logic using TF-IDF, BERT embeddings, and an ensemble approach for superior relevance and diversity. Includes XAI features.
PredictiveAnalyticsEngine: Trains and utilizes RandomForestClassifier and GradientBoostingRegressor models for admission probability and success prediction, with robust data preparation and synthetic data generation for training.

Database Schema (SQLite)
The system uses an SQLite database (data/uel_ai_system.db) for persistent storage:

student_profiles: Stores complete user profiles (JSON data).
profile_interactions: Logs user interactions with different features.
ai_conversations: Stores profile-linked chat history.
recommendation_cache: Caches profile-based course recommendations.
prediction_cache: Caches profile-linked admission predictions.
system_analytics: Records overall system events and metrics.

🔒 Privacy & Security Considerations
Local Data Storage: All user profiles and interaction data are stored locally within your system's SQLite database.
No External Profile Sharing: User data is designed to remain on your local machine and is not shared with external services or third parties.
User Control: You have complete control over your profile data, with options to edit, export, or delete it.
Password Hashing: Passwords for profile login are hashed using SHA256 before storage, ensuring they are not stored in plain text.

🎯 Benefits for Different Users
🎓 Prospective Students

Experience a truly personalized application journey.
Receive highly accurate and relevant course recommendations.
Get data-driven insights into your admission chances.
Practice interviews with an AI coach tailored to your profile.
Streamline document submission with AI verification.

👨‍🏫 Academic Advisors & Counselors
Gain a deeper understanding of student profiles and needs.
Provide more targeted and effective guidance.
Track student engagement and progress across the platform.

🏫 University Administrators
Analyze aggregated student interests and application trends (if analytics are enabled).
Understand feature usage patterns to improve service delivery.
Generate insights for strategic planning and course development.

📈 Performance & Scalability
Profile Caching: Frequently accessed profiles are cached in memory for faster retrieval.
Optimized Database Queries: Indexes are applied to database tables for efficient data access.
Modular Design: Components are designed to be independent, allowing for easier scaling and maintenance.
Asynchronous Operations: Where applicable, operations are designed to be non-blocking for a smoother user experience.
