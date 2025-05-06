import multiprocessing
from gunicorn.app.base import BaseApplication
from app import app  # Replace with your actual application import

class StandaloneApplication(BaseApplication):
    def __init__(self, app, options=None):
        self.options = options or {}
        self.application = app
        super().__init__()

    def load_config(self):
        config = {
            key: value for key, value in self.options.items()
            if key in self.cfg.settings and value is not None
        }
        for key, value in config.items():
            self.cfg.set(key.lower(), value)

    def load(self):
        return self.application

options = {
    "bind": "0.0.0.0:8000",
    "workers": 2,
    "worker_class": "uvicorn.workers.UvicornWorker",
}

if __name__ == "__main__":
    # This is crucial for multiprocessing on Windows and when using spawn method
    multiprocessing.freeze_support()

    # Set multiprocessing start method to 'spawn' for compatibility
    # This fixes issues with forking on macOS
    try:
        multiprocessing.set_start_method("fork", True)
    except RuntimeError:
        # If already set, this will raise a RuntimeError
        pass

    StandaloneApplication(app, options).run()
