import dropbox
from dropbox.files import FolderMetadata, FileMetadata
from flask import current_app
import os
from pathlib import Path
from src.utils.hash import hash_file_name

def create_dropbox_client(app_key: str, app_secret: str, refresh_token: str) -> dropbox.Dropbox:
    return dropbox.Dropbox(
        app_key=app_key,
        app_secret=app_secret,
        oauth2_refresh_token=refresh_token
    )


def upload_file_to_dropbox(file_path: str, dropbox_path: str) -> str:
    dropbox_client: dropbox.Dropbox = current_app.dropbox_client # type: ignore
    dropbox_client.check_and_refresh_access_token()

    # Open the file and upload it
    with open(file_path, 'rb') as f:
        dropbox_client.files_upload(f.read(), dropbox_path)
    current_app.logger.info(f'File {file_path} uploaded to {dropbox_path}')
    shared_link: str = dropbox_client.sharing_create_shared_link_with_settings(dropbox_path).url

    return shared_link


def download_folder(dbx, dropbox_folder: Path, local_folder: Path = Path('data'), verbose: bool = True):
    if not local_folder.exists():
        os.makedirs(local_folder)

    for entry in dbx.files_list_folder(str(dropbox_folder)).entries:
        dropbox_path = Path(entry.path_lower)
        local_path = local_folder / f'{hash_file_name(dropbox_path.stem)}{dropbox_path.suffix}'

        if isinstance(entry, FileMetadata):
            if local_path.exists():
                if verbose:
                    print(f"\033[33m[Dropbox] Skipping: {dropbox_path.name} downloaded version {local_path.name} already exists!\033[0m")
                continue
            # It's a file, download it
            with open(local_path, "wb") as f:
                metadata, res = dbx.files_download(str(dropbox_path))
                f.write(res.content)
            
            if verbose:
                print(f"\033[32m[Dropbox] Downloaded: {dropbox_path} → {local_path}\033[0m")

        elif isinstance(entry, FolderMetadata):
            # It's a subfolder, recursively download it
            sub_local_folder = local_folder / dropbox_path.stem
            download_folder(dbx, dropbox_path, sub_local_folder, verbose)


