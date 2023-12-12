import time

class Timer:
    def __init__(self):
        self.start_time = None
        self.end_time = None

    def start(self):
        self.start_time = time.time()

    def stop(self):
        self.end_time = time.time()

    def elapsed_time(self):
        if self.start_time is None:
            raise ValueError("Timer has not been started.")
        if self.end_time is None:
            self.end_time = time.time()

        return self.end_time - self.start_time

    def __str__(self):
        return f"total run time:\t {round(self.elapsed_time(), 3)} seconds"
