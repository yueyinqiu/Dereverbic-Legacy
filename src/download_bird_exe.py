import download_bird_config as config
import csdir

directory_compressed = config.destination.joinpath("compressed")
csdir.create_directory(directory_compressed)

for i in range(config.start_index, 10 + 1):
    source = config.url_pattern.format(i)
    destination = directory_compressed.joinpath(f"fold{i:02d}.zip")
    
    print(f"Downloading {source}...")
    import urllib.request
    urllib.request.urlretrieve(source, destination)

    print(f"Extracting {source}...")
    import zipfile
    with zipfile.ZipFile(destination) as zip:
        zip.extractall(config.destination)

print("Completed.")
