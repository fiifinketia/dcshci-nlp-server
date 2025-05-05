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
    'bind': '0.0.0.0:8000',
    'workers': 2,
    'worker_class': 'sync',
}

StandaloneApplication(app, options).run()
