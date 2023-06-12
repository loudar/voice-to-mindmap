import json
import time


class Timer:
    version = "v1"  # Static property for versioning

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
        filename = f"timings/timings_{self.version}.json"
        try:
            with open(filename, "r") as file:
                return json.load(file)
        except FileNotFoundError:
            return {}

    def _save_timings_to_file(self):
        filename = f"timings/timings_{self.version}.json"
        with open(filename, "w") as file:
            json.dump(self.timings, file, indent=4)
