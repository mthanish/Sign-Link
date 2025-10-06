import speech_recognition as sr
import datetime

# Initialize recognizer
recognizer = sr.Recognizer()

def stt():
    """Function to capture voice input and convert it to text"""
    with sr.Microphone() as source:
        print("\n[INFO] Clearing background noise... Please wait.")
        recognizer.adjust_for_ambient_noise(source, duration=1)
        print("[INFO] Listening... Speak now!")

        try:
            audio = recognizer.listen(source, timeout=5, phrase_time_limit=10)
        except sr.WaitTimeoutError:
            print("[WARNING] Listening timed out, no speech detected.")
            return None

    try:
        # Convert speech to text using Google API
        text = recognizer.recognize_google(audio, language="en-US")
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] You said: {text}")

        # Save recognized text to log file
        with open("speech_log.txt", "a", encoding="utf-8") as f:
            f.write(f"[{timestamp}] {text}\n")

        return text
    except sr.UnknownValueError:
        print("[ERROR] Could not understand the audio.")
        return None
    except sr.RequestError as e:
        print(f"[ERROR] Could not request results from Google API; {e}")
        return None
    except Exception as ex:
        print(f"[ERROR] {ex}")
        return None

if __name__ == "__main__":
    print("🎤 Speech-to-Text is running...")
    print("Say 'exit' or 'quit' to stop.\n")

    while True:
        user_text = stt()
        if user_text:
            if user_text.lower() in ["exit", "quit", "stop"]:
                print("👋 Exiting... Goodbye!")
                break
