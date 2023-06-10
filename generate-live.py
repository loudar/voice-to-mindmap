import threading
import speech_recognition as sr
import networkx as nx
import matplotlib.pyplot as plt
import os
import spacy
import queue
import keyboard
from datetime import datetime
from matplotlib.patches import Ellipse


def voice_to_text(q, stop_event_ref):
    # Function to handle speech recognition
    def recognize_audio(audio_source):
        try:
            text = r.recognize_google(audio_source, language=google_lang[selected_lang])
            q.put(text)  # Put the recognized text in the queue
            print(f"Recognized Text: {text}")
            update_plot(q)
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


def extract_logical_links(text):
    doc = nlp(text)
    logical_links = []

    for sentence in doc.sents:
        subjects = []
        for token in sentence:
            if token.pos_ == 'NOUN' or token.dep_ == 'nsubj' or token.dep_ == 'nsubjpass':
                subjects.append(token.text)
            elif token.pos_ == 'ADJ':
                subjects.append(token.text)

        for i in range(len(subjects) - 1):
            logical_links.append((subjects[i], subjects[i + 1]))

    return logical_links


def create_mind_map(proximity_links):
    G = nx.Graph()

    for word1, word2 in proximity_links:
        if word1 not in G.nodes():
            G.add_node(word1, size=1)
        else:
            G.nodes[word1]['size'] += 1

        if word2 not in G.nodes():
            G.add_node(word2, size=1)
        else:
            G.nodes[word2]['size'] += 1

        if not G.has_edge(word1, word2):
            if word1 == word2:
                continue
            G.add_edge(word1, word2, weight=1)
        else:
            G[word1][word2]['weight'] += 1

    return G


def update_plot(q):
    global accumulated_text
    # Check if new text is available in the queue
    while not q.empty():
        text = q.get()  # Get the latest recognized text
        if text == reset_words[selected_lang]:
            accumulated_text = ""
            print("Resetting...")
            return

        accumulated_text += text + " "
        with open(transcript_file, 'a') as file:
            file.write(text)

    # Extract logical links from the new text



languages = ['en', 'de']
models = {
    'en': 'en_core_web_sm',
    'de': 'de_core_news_sm'
}
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

nlp = spacy.load(models[selected_lang])

r = sr.Recognizer()
accumulated_text = ""

# use current timestamp as unique identifier for the transcript
conversation_id = datetime.now().strftime("%Y-%m-%d-%H_%M_%S")
transcript_file = f"transcripts/transcript_{conversation_id}.txt"
plot_file = f"maps/mindmap_{conversation_id}.svg"

# Create a figure and axes for the plot
fig, ax = plt.subplots()

# Create a queue to store the recognized text
text_queue = queue.Queue()

# Create an event to control the audio recording
stop_event = threading.Event()

# Create a lock to synchronize access to the accumulated text
lock = threading.Lock()


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
    global accumulated_text, conversation_id, transcript_file, plot_file
    accumulated_text = ""
    print("Resetting...")
    conversation_id = datetime.now().strftime("%Y-%m-%d-%H_%M_%S")
    transcript_file = f"transcripts/transcript_{conversation_id}.txt"
    plot_file = f"maps/mindmap_{conversation_id}.svg"


# Register the stop key press callback
keyboard.on_press_key("F2", stop_key_press)
keyboard.on_press_key("F4", handle_lang_change)
keyboard.on_press_key("F3", handle_reset)

# Create a thread for continuous audio recording
audio_thread = threading.Thread(target=voice_to_text, args=(text_queue, stop_event))
print("Starting audio thread...")
audio_thread.start()

# check if the directory exists. If not, create it
if not os.path.exists('maps'):
    os.makedirs('maps')
if not os.path.exists('transcripts'):
    os.makedirs('transcripts')

# Show the live plot
plt.show()