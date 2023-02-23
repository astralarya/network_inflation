import glob
from pathlib import Path
import shutil
import sys
import tarfile

from tqdm import tqdm


git_dir = Path(sys.argv[1])
out_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(__file__, "data/imagenet-1k")


def get_files(split: str):
    files = [
        *glob.glob(f"{git_dir}/data/{split}_images.tar.gz"),
        *glob.glob(f"{git_dir}/data/{split}_images_[0-9].tar.gz"),
    ]
    files.sort()
    return files

def transform_name(name: str):
    dots = name.split(".")
    base, ext = ".".join(dots[:-1]), dots[-1]
    bars = base.split("_")
    name, label = f"{'_'.join(bars[:-1])}.{ext}", bars[-1]
    return name, label


# Setup output dir

out_dir.mkdir(parents=True, exist_ok=True)
for split in ["test", "train", "val"]
    out_dir.joinpath(split).mkdir(exist_ok=True)

# Extract files

shutil.copy(git_dir.joinpath("classes.py"), out_dir.joinpath("classes.py"))

for split in ["train", "val"]:
    for filename in get_files(split):
        print(f"Reading `{filename}`")
        with tarfile.open(filename, "r:gz") as tar:
            for member in tqdm(tar.getmembers()):
                name, label = transform_name(member.name)
                label_dir = out_dir.joinpath(split, label)
                label_dir.mkdir(exist_ok=True)
                with tar.extractfile(member) as extract:
                    with label_dir.joinpath(name).open(mode="wb") as output:
                        output.write(extract.read())

for filename in get_files("test"):
    print(f"Reading `{filename}`")
    with tarfile.open(filename, "r:gz") as tar:
        for member in tqdm(tar.getmembers()):
            with tar.extractfile(member) as extract:
                with out_dir.joinpath("test", name).open(mode="wb") as output:
                    output.write(extract.read())
