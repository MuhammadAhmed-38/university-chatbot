# test_ollama.py
from unified_uel_ai_system import OllamaService

# Initialize the service
ollama_service = OllamaService()

# Check if Ollama is running
if ollama_service.is_available():
    print("✅ Ollama is available!")
    
    # Test with a simple prompt
    prompt = "What are the admission requirements for international students at UEL?"
    print(f"\n🤖 Asking: {prompt}\n")
    
    response = ollama_service.generate_response(prompt)
    print(f"📝 Response: {response}")
else:
    print("❌ Ollama is not available. Make sure Ollama is running!")
    print("To start Ollama, run: ollama serve")
    
    # Still get a response (fallback)
    response = ollama_service.generate_response("Hello")
    print(f"\n📝 Fallback Response: {response}")