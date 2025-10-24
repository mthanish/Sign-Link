import sys
import os

# Add the project root directory to Python path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

try:
    from speech_to_text.speech_to_text import listen
    print("Successfully imported 'listen' from speech_to_text module!")
    
    from text_to_speech.text_to_speech import speak_text
    print("Successfully imported 'speak_text' from text_to_speech module!")
    
    print("All imports are working correctly!")
except ImportError as e:
    print(f"Import error: {e}")