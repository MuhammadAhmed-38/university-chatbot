# Troubleshooting Guide - Verification Script
# Save this as verify_prediction_changes.py and run it

import sys
import importlib
import inspect
from datetime import datetime

print("🔍 Verifying Prediction Enhancement Implementation")
print("=" * 50)

# Step 1: Check if the files can be imported
try:
    # Force reload to get latest changes
    if 'unified_uel_ai_system' in sys.modules:
        del sys.modules['unified_uel_ai_system']
    if 'unified_uel_ui' in sys.modules:
        del sys.modules['unified_uel_ui']
    
    import unified_uel_ai_system
    import unified_uel_ui
    
    print("✅ Successfully imported both modules")
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)

# Step 2: Check PredictiveAnalyticsEngine features
print("\n📊 Checking PredictiveAnalyticsEngine...")
try:
    # Create instance
    from unified_uel_ai_system import DataManager, PredictiveAnalyticsEngine
    
    # Check if the class has new methods
    engine_methods = [method for method in dir(PredictiveAnalyticsEngine) if not method.startswith('_')]
    
    # Check for new methods that should exist
    required_methods = [
        '_get_education_level_score',
        '_calculate_education_compatibility',
        '_calculate_gpa_percentile',
        '_calculate_ielts_percentile',
        '_generate_insights',
        '_get_competitiveness_level',
        '_recommend_education_pathway'
    ]
    
    missing_methods = []
    for method in required_methods:
        if not hasattr(PredictiveAnalyticsEngine, method):
            missing_methods.append(method)
    
    if missing_methods:
        print(f"❌ Missing methods in PredictiveAnalyticsEngine: {missing_methods}")
        print("   → The enhanced version was NOT properly implemented")
    else:
        print("✅ All new methods found in PredictiveAnalyticsEngine")
        
    # Check feature names
    data_manager = DataManager()
    engine = PredictiveAnalyticsEngine(data_manager)
    
    if hasattr(engine, 'feature_names'):
        feature_count = len(engine.feature_names)
        print(f"📊 Feature count: {feature_count}")
        if feature_count == 6:
            print("❌ Still using OLD feature set (6 features)")
            print("   Features:", engine.feature_names)
        elif feature_count == 11:
            print("✅ Using NEW enhanced feature set (11 features)")
            print("   Features:", engine.feature_names)
        else:
            print(f"⚠️  Unexpected feature count: {feature_count}")
    
except Exception as e:
    print(f"❌ Error checking PredictiveAnalyticsEngine: {e}")

# Step 3: Check UI changes
print("\n🖥️  Checking UI render_predictions_page...")
try:
    # Get the function source
    render_func = unified_uel_ui.render_predictions_page
    source_code = inspect.getsource(render_func)
    
    # Check for key indicators of the new version
    indicators = {
        "Current/Highest Education Level": "Education level selector",
        "enhanced_prediction_form": "Enhanced form name",
        "Education Pathway Analysis": "New analysis section",
        "education_compatibility": "Compatibility calculation",
        "Academic Performance Ranking": "Percentile section",
        "course_category": "Dynamic course selection"
    }
    
    missing_indicators = []
    found_indicators = []
    
    for indicator, description in indicators.items():
        if indicator in source_code:
            found_indicators.append(description)
        else:
            missing_indicators.append(description)
    
    if missing_indicators:
        print(f"❌ Missing UI elements: {missing_indicators}")
        print("   → The enhanced UI was NOT properly implemented")
    else:
        print("✅ All new UI elements found")
        
    if found_indicators:
        print(f"✅ Found {len(found_indicators)} new UI elements")
    
    # Check if it's the old version
    if "prediction_form" in source_code and "enhanced_prediction_form" not in source_code:
        print("❌ Still using OLD prediction form")
    
except Exception as e:
    print(f"❌ Error checking UI: {e}")

# Step 4: Test the prediction function with new parameters
print("\n🧪 Testing prediction with education level...")
try:
    from unified_uel_ai_system import UnifiedUELAIAgent
    
    agent = UnifiedUELAIAgent()
    
    # Test data with new fields
    test_data = {
        'name': 'Test User',
        'gpa': 3.5,
        'ielts_score': 7.0,
        'work_experience_years': 2,
        'course_applied': 'Computer Science MSc',
        'nationality': 'UK',
        'application_date': datetime.now().strftime('%Y-%m-%d'),
        'current_education': 'Bachelor\'s Degree'  # NEW FIELD
    }
    
    result = agent.predict_admission_success(test_data)
    
    # Check if result has new fields
    if 'education_compatibility' in result:
        print(f"✅ Education compatibility found: {result['education_compatibility']:.2f}")
    else:
        print("❌ Education compatibility NOT found in results")
        
    if 'insights' in result:
        print("✅ Insights section found")
    else:
        print("❌ Insights section NOT found")
        
    if 'academic_percentiles' in result:
        print("✅ Academic percentiles found")
    else:
        print("❌ Academic percentiles NOT found")
    
    print(f"\n📊 Prediction result keys: {list(result.keys())}")
    
except Exception as e:
    print(f"❌ Error testing prediction: {e}")
    import traceback
    traceback.print_exc()

# Step 5: Quick fix instructions
print("\n🔧 QUICK FIX INSTRUCTIONS:")
print("=" * 50)
print("""
If the enhancements are NOT showing:

1. STOP the Streamlit app (Ctrl+C)

2. Clear all caches:
   streamlit cache clear

3. Make sure you've saved the files after pasting the code

4. Check that you replaced the ENTIRE class/function, not just parts

5. For unified_uel_ai_system.py:
   - Find: class PredictiveAnalyticsEngine
   - Select the ENTIRE class (from 'class' to the last method)
   - Replace with the enhanced version

6. For unified_uel_ui.py:
   - Find: def render_predictions_page
   - Select the ENTIRE function
   - Replace with the enhanced version

7. Restart the application:
   python main.py --run

8. Force refresh browser (Ctrl+F5)
""")

print("\n💡 Additional debugging:")
print("Run this to see exactly what version you have:")
print("grep -n 'education_compatibility' unified_uel_ai_system.py")
print("(Should show multiple results if enhanced version is installed)")