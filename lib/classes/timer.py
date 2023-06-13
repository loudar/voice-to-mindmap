import json
import os
import time


def ensure_folder_exists(folder="timings"):
    if not os.path.exists(folder):
        os.mkdir(folder)


class Timer:
    folder = "timings"

    def __init__(self):
        self.current_key = None
        self.timings = self._load_timings_from_file()

    def start(self, key):
        self.current_key = key
        self.timings[self.current_key] = time.perf_counter()

    def stop(self):
        if self.current_key:
            elapsed_time = time.perf_counter() - self.timings[self.current_key]
            self.timings[self.current_key] = elapsed_time
            self.current_key = None
            self._save_timings_to_file()

    def _load_timings_from_file(self):
        ensure_folder_exists(self.folder)
        filename = f"{self.folder}/timings.json"
        try:
            with open(filename, "r") as file:
                return json.load(file)
        except FileNotFoundError:
            return {}

    def _save_timings_to_file(self):
        ensure_folder_exists(self.folder)
        filename = f"{self.folder}/timings.json"
        with open(filename, "w") as file:
            json.dump(self.timings, file, indent=4)
