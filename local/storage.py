import glob
from pathlib import Path
from os import environ

try:
    from google.cloud import storage as gcloud_storage
except ImportError:
    gcloud_storage = None


GCLOUD_BUCKET = environ.get("GCLOUD_BUCKET", None)
FILE_PATH = Path(environ.get("MODEL_PATH", "/mnt/models/data"))


def set_file_path(file_path):
    global FILE_PATH
    FILE_PATH = Path(file_path)


def file__path_exists(path):
    return (FILE_PATH / path).exists()


def gcloud__path_exists(path):
    storage_client = gcloud_storage.Client()
    bucket = storage_client.bucket(GCLOUD_BUCKET)
    return bucket.blob(path).exists()


def _path_exists():
    if gcloud_storage and GCLOUD_BUCKET:
        return gcloud__path_exists
    else:
        return file__path_exists


path_exists = _path_exists()


def file__path_iter(path):
    paths = glob.glob(f"{FILE_PATH / path}/*")
    for path in paths:
        yield path


def gcloud__path_iter(path):
    storage_client = gcloud_storage.Client()
    return (
        item.name
        for item in storage_client.list_blobs(GCLOUD_BUCKET, prefix=f"{path}/")
    )


def _path_iter():
    if gcloud_storage and GCLOUD_BUCKET:
        return gcloud__path_iter
    else:
        return file__path_iter


path_iter = _path_iter()


def file__path_unlink(path):
    (FILE_PATH / path).unlink()


def gcloud__path_unlink(path):
    storage_client = gcloud_storage.Client()
    bucket = storage_client.bucket(GCLOUD_BUCKET)
    bucket.blob(path).delete()


def _path_unlink():
    if gcloud_storage and GCLOUD_BUCKET:
        return gcloud__path_unlink
    else:
        return file__path_unlink


path_unlink = _path_unlink()


def file__path_open(path, mode="r"):
    (FILE_PATH / path).parent.mkdir(parents=True, exist_ok=True)
    return (FILE_PATH / path).open(mode)


def gcloud__path_open(path, mode="r"):
    storage_client = gcloud_storage.Client()
    bucket = storage_client.bucket(GCLOUD_BUCKET)
    if mode == "a":
        blob = bucket.blob(path)
        data = None
        if blob.exists():
            data = blob.download_as_string().decode()
        file = blob.open("w")
        if data:
            file.write(data)
        return file
    else:
        return bucket.blob(path).open(
            mode, ignore_flush=True if "w" in mode and "b" in mode else False
        )


def _path_open():
    if gcloud_storage and GCLOUD_BUCKET:
        return gcloud__path_open
    else:
        return file__path_open


path_open = _path_open()
