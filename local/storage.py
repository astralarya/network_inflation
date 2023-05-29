import glob
from pathlib import Path
from os import environ

try:
    from google.cloud import storage as gcloud_storage
except ImportError:
    gcloud_storage = None

try:
    from google.api_core import page_iterator
except ImportError:
    gcloud_page_iterator = None


GCLOUD_BUCKET = environ.get("GCLOUD_BUCKET", None)


def file__path_exists(path):
    return Path(path).exists()


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


def file__path_iter(path, shallow=False):
    paths = glob.glob(f"{path}/*")
    for path in paths:
        yield path


def gcloud__path_iter(path, shallow=False):
    if shallow:
        return glcoud_list_directories(path)
    storage_client = gcloud_storage.Client()
    min_path = len(path) + 1
    return (
        item.name
        for item in storage_client.list_blobs(GCLOUD_BUCKET, prefix=f"{path}/")
        if len(item.name) > min_path
    )


def glcoud_list_directories(prefix):
    def _item_to_value(iterator, item):
        return item

    if prefix and not prefix.endswith("/"):
        prefix += "/"

    extra_params = {"projection": "noAcl", "prefix": prefix, "delimiter": "/"}

    gcs = gcloud_storage.Client()

    path = "/b/" + GCLOUD_BUCKET + "/o"

    return page_iterator.HTTPIterator(
        client=gcs,
        api_request=gcs._connection.api_request,
        path=path,
        items_key="prefixes",
        item_to_value=_item_to_value,
        extra_params=extra_params,
    )


def _path_iter():
    if gcloud_storage and GCLOUD_BUCKET:
        return gcloud__path_iter
    else:
        return file__path_iter


path_iter = _path_iter()


def file__path_unlink(path):
    Path(path).unlink()


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
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    return Path(path).open(mode)


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
