from queue import Queue
from threading import Thread
import time
from lib.advanced_text_processing import extract_logical_links_advanced
from lib.mapper import create_mind_map_force
from lib.plotly_wrapper import create_plot


class BackgroundProcessor:
    def __init__(self):
        self.queue = Queue()
        self.data = None
        self.thread = Thread(target=self._process, daemon=True)
        self.thread.start()

    def add_task(self, task):
        self.queue.put(task)

    def _process(self):
        while True:
            task = self.queue.get()
            if task is not None:
                print(f"Task {task[3]} - Processing...")
                text, selected_lang, conversation_id, n = task
                logical_links = extract_logical_links_advanced(text, selected_lang, True)
                G, positions = create_mind_map_force(logical_links)
                self.data = create_plot(G, positions, True, title=f"Transcript: {conversation_id} ({len(text)} characters - task {n})")
            time.sleep(.1)  # to prevent busy looping

    def get_data(self):
        return self.data
