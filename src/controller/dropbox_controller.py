import dropbox
from flask import current_app
import os


def create_dropbox_client(app_key: str, app_secret: str, refresh_token: str) -> dropbox.Dropbox:
    return dropbox.Dropbox(
        app_key=app_key,
        app_secret=app_secret,
        oauth2_refresh_token=refresh_token
    )


def upload_file_to_dropbox(file_path: str, dropbox_path: str) -> str:
    dropbox_client: dropbox.Dropbox = current_app.dropbox_client
    dropbox_client.check_and_refresh_access_token()

    # Open the file and upload it
    with open(file_path, 'rb') as f:
        dropbox_client.files_upload(f.read(), dropbox_path)
    current_app.logger.info(f'File {file_path} uploaded to {dropbox_path}')
    shared_link: str = dropbox_client.sharing_create_shared_link_with_settings(dropbox_path).url

    return shared_link


def download_folder(dbx, dropbox_folder, local_folder='data'):
    # Ensure the local folder exists
    if not os.path.exists(local_folder):
        os.makedirs(local_folder)

    # List files in the folder
    for entry in dbx.files_list_folder(dropbox_folder).entries:
        dropbox_path = entry.path_lower
        local_path = os.path.join(local_folder, os.path.basename(dropbox_path))

        if isinstance(entry, dropbox.files.FileMetadata):
            # It's a file, download it
            with open(local_path, "wb") as f:
                metadata, res = dbx.files_download(dropbox_path)
                f.write(res.content)
            print(f"Downloaded: {dropbox_path} → {local_path}")

        elif isinstance(entry, dropbox.files.FolderMetadata):
            # It's a subfolder, recursively download it
            sub_local_folder = os.path.join(local_folder, os.path.basename(dropbox_path))
            download_folder(dbx, dropbox_path, sub_local_folder)


if __name__ == '__main__':
    from src.AppConfig import AppConfig

    dropbox = create_dropbox_client(
        app_key=AppConfig.DROPBOX_APP_KEY,
        app_secret=AppConfig.DROPBOX_APP_SECRET,
        refresh_token=AppConfig.DROPBOX_REFRESH_TOKEN
    )
    print(dropbox.files_list_folder('/EmotionExperiment/labeled').entries)
