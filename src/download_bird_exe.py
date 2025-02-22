from pathlib import Path

import csdir

import download_bird_config


directory_compressed: Path = download_bird_config.destination.joinpath("compressed")
csdir.create_directory(directory_compressed)

i: int
for i in range(download_bird_config.start_index, 10 + 1):
    source: str = download_bird_config.url_pattern.format(i)
    destination: Path = directory_compressed.joinpath(f"fold{i:02d}.zip")
    
    print(f"Downloading {source}...")
    import urllib.request
    urllib.request.urlretrieve(source, destination)

    print(f"Extracting {source}...")
    import zipfile
    zip: zipfile.ZipFile
    with zipfile.ZipFile(destination) as zip:
        zip.extractall(download_bird_config.destination)

print("Completed.")
