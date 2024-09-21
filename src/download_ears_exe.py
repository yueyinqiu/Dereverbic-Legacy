import download_ears_config as config
import csdir

directory_compressed = config.destination.joinpath("compressed")
csdir.create_directory(directory_compressed)

for i in range(config.start_index, 107 + 1):
    file_name = f"p{i:03d}.zip"
    
    import urllib.parse
    source = urllib.parse.urljoin(config.base_url, file_name)
    destination = directory_compressed.joinpath(file_name)
    
    print(f"Downloading {source}...")
    import urllib.request
    urllib.request.urlretrieve(source, destination)

    print(f"Extracting {source}...")
    import zipfile
    with zipfile.ZipFile(destination) as zip:
        zip.extractall(config.destination)

print("Completed.")
