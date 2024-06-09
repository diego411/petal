import dropbox
import os
from dotenv import load_dotenv

load_dotenv()
DROPBOX_ACCESS_TOKEN = os.environ.get("DROPBOX_ACCESS_TOKEN")
dbx = dropbox.Dropbox(DROPBOX_ACCESS_TOKEN)


def upload_file_to_dropbox(file_path, dropbox_path):
    global dbx

    try:
        # Open the file and upload it
        with open(file_path, 'rb') as f:
            dbx.files_upload(f.read(), dropbox_path)
        print(f'File {file_path} uploaded to {dropbox_path}')
    except Exception as e:
        print(f'Error uploading file: {e}')
