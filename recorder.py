import sys
import threading
import speech_recognition as sr
import os
import keyboard
from datetime import datetime

if len(sys.argv) > 1:
    timeout = int(sys.argv[1])
else:
    timeout = -1


def voice_to_text(stop_event_ref):
    # Function to handle speech recognition
    def recognize_audio(audio_source):
        try:
            text = r.recognize_google(audio_source, language=google_lang[selected_lang])
            print(f"Recognized Text: {text}")
            with open(transcript_file, 'a', encoding='utf-8') as file:
                file.write(text + " ")
            with open(latest_transcript_file, 'a', encoding='utf-8') as file:
                file.write(text + " ")
        except sr.UnknownValueError:
            print("Speech recognition could not understand audio.")
        except sr.RequestError as e:
            print("Could not request results from Google Speech Recognition service; {0}".format(e))
        except Exception as e:
            print(e)

    # Record audio continuously until stop event is set
    with sr.Microphone(device_index=0) as source:
        while not stop_event_ref.is_set():
            print("Recording audio...")
            try:
                audio = r.listen(source, phrase_time_limit=5)
                print("Audio recording complete.")
            except sr.WaitTimeoutError:
                print("No speech detected. Trying again...")
                continue

            # Create a new thread for speech recognition
            recognition_thread = threading.Thread(target=recognize_audio, args=(audio,))
            recognition_thread.start()


languages = ['en', 'de']
google_lang = {
    'en': 'en-US',
    'de': 'de-DE'
}
google_lang_arr = ['en-US', 'de-DE']
reset_words = {
    'en': 'reset',
    'de': 'zurücksetzen'
}

selected_lang = 'de'

r = sr.Recognizer()

# use current timestamp as unique identifier for the transcript
conversation_id = datetime.now().strftime("%Y-%m-%d-%H_%M_%S")
transcript_folder = "transcripts/"
transcript_file = f"{transcript_folder}transcript_{conversation_id}.txt"
latest_transcript_file = f"{transcript_folder}transcript_latest.txt"

# Create an event to control the audio recording
stop_event = threading.Event()

# Create a lock to synchronize access to the accumulated text
lock = threading.Lock()


# Function to handle the stop key press
def stop_key_press(event):
    with lock:
        print("Stopping...")
        os.system(f"python plotter.py {transcript_file}")
        stop_event.set()


def handle_lang_change(event):
    global selected_lang
    selected_lang = languages[1] if selected_lang == languages[0] else languages[0]
    print(f"Changed language to {selected_lang}")


def handle_reset(event):
    global conversation_id, transcript_file
    print("Resetting...")
    conversation_id = datetime.now().strftime("%Y-%m-%d-%H_%M_%S")
    transcript_file = f"transcripts/transcript_{conversation_id}.txt"
    with open(latest_transcript_file, 'w', encoding='utf-8') as file:
        file.write(transcript_file)


# Register the stop key press callback
keyboard.on_press_key("F2", stop_key_press)
keyboard.on_press_key("F4", handle_lang_change)
keyboard.on_press_key("F3", handle_reset)

# Create a thread for continuous audio recording
audio_thread = threading.Thread(target=voice_to_text, args=(stop_event,))
print("Starting audio thread...")
audio_thread.start()

# check if the directory exists. If not, create it
if not os.path.exists('maps'):
    os.makedirs('maps')
if not os.path.exists('transcripts'):
    os.makedirs('transcripts')

if timeout > 0:
    print(f"Recording for {timeout} seconds before stopping...")
    stop_event.wait(timeout)
