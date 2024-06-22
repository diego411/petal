import dropbox
import os
from dotenv import load_dotenv

load_dotenv()
DROPBOX_REFRESH_TOKEN = os.environ.get("DROPBOX_REFRESH_TOKEN")
DROPBOX_APP_KEY = os.environ.get("DROPBOX_APP_KEY")
DROPBOX_APP_SECRET = os.environ.get("DROPBOX_APP_SECRET")
dbx = dropbox.Dropbox(
    app_key=DROPBOX_APP_KEY,
    app_secret=DROPBOX_APP_SECRET,
    oauth2_refresh_token=DROPBOX_REFRESH_TOKEN
)


def upload_file_to_dropbox(file_path, dropbox_path):
    global dbx

    dbx.check_and_refresh_access_token()

    try:
        # Open the file and upload it
        with open(file_path, 'rb') as f:
            dbx.files_upload(f.read(), dropbox_path)
        print(f'File {file_path} uploaded to {dropbox_path}')
    except Exception as e:
        print(f'Error uploading file: {e}')
