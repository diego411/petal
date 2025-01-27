import dropbox
from flask import current_app


def create_dropbox_client(app_key: str, app_secret: str, refresh_token: str) -> dropbox.Dropbox:
    return dropbox.Dropbox(
        app_key=app_key,
        app_secret=app_secret,
        oauth2_refresh_token=refresh_token
    )


def upload_file_to_dropbox(dropbox_client: dropbox.Dropbox, file_path: str, dropbox_path: str) -> str:
    dropbox_client.check_and_refresh_access_token()

    # Open the file and upload it
    with open(file_path, 'rb') as f:
        dropbox_client.files_upload(f.read(), dropbox_path)
    current_app.logger.info(f'File {file_path} uploaded to {dropbox_path}')
    shared_link: str = dropbox_client.sharing_create_shared_link_with_settings(dropbox_path).url

    return shared_link
