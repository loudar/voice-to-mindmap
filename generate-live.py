import threading
import speech_recognition as sr
import os
import queue
import keyboard
from datetime import datetime

from lib.mapper import create_mind_map_force
from lib.plotly_wrapper import create_plot
from lib.text_processing import extract_logical_links
from dash import Dash


def voice_to_text(q, stop_event_ref):
    # Function to handle speech recognition
    def recognize_audio(audio_source):
        try:
            text = r.recognize_google(audio_source, language=google_lang[selected_lang])
            q.put(text)  # Put the recognized text in the queue
            print(f"Recognized Text: {text}")
            update_thread = threading.Thread(target=update_plot, args=(q,))
            update_thread.start()
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

            # Join the speech recognition thread with the main thread
            recognition_thread.join()


def update_plot(q):
    global accumulated_text, update

    # Check if new text is available in the queue
    while not q.empty():
        text = q.get()  # Get the latest recognized text

        accumulated_text += text + " "
        with open(transcript_file, 'a') as file:
            file.write(text)

    # Extract logical links from the new text
    logical_links = extract_logical_links(accumulated_text, selected_lang)

    # Create the mind map using the accumulated logical links
    G, subgraph_positions = create_mind_map_force(logical_links)
    create_plot(G, subgraph_positions, update)
    update = True


# Function to handle the stop key press
def stop_key_press(event):
    with lock:
        print("Stopping...")
        stop_event.set()


def handle_lang_change(event):
    global selected_lang
    selected_lang = languages[1] if selected_lang == languages[0] else languages[0]
    print(f"Changed language to {selected_lang}")


def handle_reset(event):
    global accumulated_text, conversation_id, transcript_file
    accumulated_text = ""
    print("Resetting...")
    conversation_id = datetime.now().strftime("%Y-%m-%d-%H_%M_%S")
    transcript_file = f"transcripts/transcript_{conversation_id}.txt"


# Register the stop key press callback
keyboard.on_press_key("F2", stop_key_press)
keyboard.on_press_key("F4", handle_lang_change)
keyboard.on_press_key("F3", handle_reset)

languages = ['en', 'de']
google_lang = {
    'en': 'en-US',
    'de': 'de-DE'
}
google_lang_arr = ['en-US', 'de-DE']

selected_lang = 'de'
print(f"Selected language: {selected_lang}")

r = sr.Recognizer()
accumulated_text = ""
global update
update = False

# use current timestamp as unique identifier for the transcript
conversation_id = datetime.now().strftime("%Y-%m-%d-%H_%M_%S")
transcript_file = f"transcripts/transcript_{conversation_id}.txt"

# Create a queue to store the recognized text
text_queue = queue.Queue()

# Create an event to control the audio recording
stop_event = threading.Event()

# Create a lock to synchronize access to the accumulated text
lock = threading.Lock()

# check if the directory exists. If not, create it
if not os.path.exists('transcripts'):
    os.makedirs('transcripts')

# Create a thread for continuous audio recording
audio_thread = threading.Thread(target=voice_to_text, args=(text_queue, stop_event))
print("Starting audio thread...")
audio_thread.start()

app = Dash(__name__)
if __name__ == '__main__':
    app.run(debug=True)

while not stop_event.is_set():
    pass

audio_thread.join()
