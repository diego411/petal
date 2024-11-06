import os
from dotenv import load_dotenv

load_dotenv()


class AppConfig:
    VERSION = '0.0.18'
    AUDIO_DIR = os.environ.get('AUDIO_DIR') or 'audio'
    LOG_THRESHOLD = os.environ.get('LOG_THRESHOLD') or 20
    DROPBOX_APP_KEY = os.environ.get('DROPBOX_APP_KEY')
    DROPBOX_APP_SECRET = os.environ.get('DROPBOX_APP_SECRET')
    DROPBOX_REFRESH_TOKEN = os.environ.get('DROPBOX_REFRESH_TOKEN')
    AUGMENT_WINDOW = int(os.environ.get('AUGMENT_WINDOW')) or 1
    AUGMENT_PADDING = int(os.environ.get('AUGMENT_PADDING')) or 1
