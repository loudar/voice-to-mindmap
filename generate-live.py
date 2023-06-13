import logging
import sys
import threading

import speech_recognition as sr
import os
import queue
import keyboard
from datetime import datetime

from lib.classes.background_processor import BackgroundProcessor
from dash.dependencies import Input, Output
from dash import Dash, dash, html, dcc

app = Dash(__name__)

app.layout = html.Div([
    dcc.Checklist(
        id='show-labels',
        options=[
            {'label': 'Show Labels', 'value': 'show'}
        ],
        value=['show']
    ),
    dcc.Graph(id='live-graph', animate=True, style={'height': '100vh'}),
    dcc.Interval(
        id='graph-update',
        interval=1 * 1000,  # in milliseconds
        n_intervals=0
    )
])

processing_flag = False
recorded_segments = 0
processor = BackgroundProcessor()


def voice_to_text(q, stop_event_ref):
    def recognize_audio(audio_source):
        global recorded_segments
        try:
            text = r.recognize_google(audio_source, language=google_lang[selected_lang])
            q.put(text + " ")
            print(f"Task {recorded_segments} - Recognized Text: {text}")
            recorded_segments += 1
            update_plot(q, recorded_segments)
        except sr.UnknownValueError:
            print("couldn't understand audio.", end="")
        except sr.RequestError as e:
            print("Could not request results from Google Speech Recognition service; {0}".format(e))
        except Exception as e:
            print(e)

    with sr.Microphone(device_index=0) as source:
        while not stop_event_ref.is_set():
            print("Recording...", end="")
            try:
                audio = r.listen(source, phrase_time_limit=20)
            except sr.WaitTimeoutError:
                print("no speech detected. Trying again...")
                continue

            recognition_thread = threading.Thread(target=recognize_audio, args=(audio,))
            recognition_thread.start()


def update_plot(q, n):
    global accumulated_text
    while not q.empty():
        text = q.get()  # Get the latest recognized text
        accumulated_text += text
        with open(transcript_file, 'a', encoding='utf-8') as file:
            file.write(text)
        with open(latest_transcript_file, 'a', encoding='utf-8') as file:
            file.write(text)
        processor.add_task((accumulated_text, selected_lang, conversation_id, n))


@app.callback(
    Output('live-graph', 'figure'),
    [Input('graph-update', 'n_intervals')],
)
def update_graph_scatter(n):
    global accumulated_text, previous_accumulated_text, processing_flag
    if accumulated_text != previous_accumulated_text and accumulated_text != "" and not processing_flag:
        processing_flag = True
        previous_accumulated_text = accumulated_text
        print(f"Task {recorded_segments} - Updating plot...")

        figure = processor.get_data()  # Get the latest processed data

        if figure is not None:
            print(f"Task {recorded_segments} - Plot updated!")
            processing_flag = False
            return figure
    else:
        return dash.no_update


started = False
if not started:
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
    previous_accumulated_text = ""

    # use current timestamp as unique identifier for the transcript
    conversation_id = datetime.now().strftime("%Y-%m-%d-%H_%M_%S")
    transcript_file = f"transcripts/transcript_{conversation_id}.txt"
    latest_transcript_file = f"transcripts/transcript_latest.txt"

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
            sys.exit()


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
        with open(latest_transcript_file, 'w', encoding='utf-8') as file:
            file.write("")


    # Register the stop key press callback
    keyboard.on_press_key("F2", stop_key_press)
    keyboard.on_press_key("F4", handle_lang_change)
    keyboard.on_press_key("F3", handle_reset)

    # check if the directory exists. If not, create it
    if not os.path.exists('transcripts'):
        os.makedirs('transcripts')

    # Create a thread for continuous audio recording
    audio_thread = threading.Thread(target=voice_to_text, args=(text_queue, stop_event))
    print("Starting audio thread...")
    audio_thread.start()
    log = logging.getLogger('werkzeug')
    log.setLevel(logging.ERROR)

started = True

if __name__ == '__main__':
    os.system("start http://127.0.0.1:7080")
    app.run_server(host='127.0.0.1', port='7080', proxy=None, debug=False, dev_tools_ui=None)
