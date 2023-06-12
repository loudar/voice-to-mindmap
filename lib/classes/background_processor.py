from queue import Queue
from threading import Thread
import time

from deepmultilingualpunctuation import PunctuationModel

from lib.advanced_text_processing import extract_logical_links_advanced, get_preprocessed_sentences
from lib.mapper import create_mind_map_force
from lib.plotly_wrapper import create_plot


class BackgroundProcessor:
    def __init__(self):
        self.queue = Queue()
        self.data = None
        self.thread = Thread(target=self._process, daemon=True)
        self.thread.start()
        self.punctuation_model = PunctuationModel()
        self.sentence_map = {}

    def add_task(self, task):
        self.queue.put(task)

    def _process(self):
        while True:
            task = self.queue.get()
            if task is not None:
                print(f"Task {task[3]} - Processing...")
                text, selected_lang, conversation_id, n = task
                text = self.punctuation_model.restore_punctuation(text)
                sentences = get_preprocessed_sentences(text)
                if len(sentences) > 1:
                    for i in range(len(sentences) - 1):
                        if sentences[i] not in self.sentence_map:
                            sentence = sentences[i] + sentences[i + 1]
                            self.sentence_map[i] = extract_logical_links_advanced(sentence, selected_lang, True)

                logical_links = []
                for i in range(len(sentences) - 1):
                    logical_links.extend(self.sentence_map[i])

                print(f"Task {n} - Logical links: {logical_links}")
                G, positions = create_mind_map_force(logical_links)
                self.data = create_plot(G, positions, True, title=f"Transcript: {conversation_id} ({len(text)} characters - task {n})")
            time.sleep(.1)

    def get_data(self):
        return self.data
