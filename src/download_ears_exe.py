from pathlib import Path

import csdir

import download_ears_config


directory_compressed: Path = download_ears_config.destination.joinpath("compressed")
csdir.create_directory(directory_compressed)

i: int
for i in range(download_ears_config.start_index, 107 + 1):
    file_name: str = f"p{i:03d}.zip"
    
    import urllib.parse
    source: str = urllib.parse.urljoin(download_ears_config.base_url, file_name)
    destination: Path = directory_compressed.joinpath(file_name)
    
    print(f"Downloading {source}...")
    import urllib.request
    urllib.request.urlretrieve(source, destination)

    print(f"Extracting {source}...")
    import zipfile
    zip: zipfile.ZipFile
    with zipfile.ZipFile(destination) as zip:
        zip.extractall(download_ears_config.destination)

print("Completed.")
