from pathlib import Path
import shutil
from urllib.request import urlretrieve

sidd_urllist_path = Path(__file__).parent.parent / 'SIDD_URLs.txt'

def main():
    # Parse URLs to download
    with open(sidd_urllist_path, 'r') as sidd_urllist:
        # Only download sRGB files (GT and noisy)
        urls = [url[:-1] for url in sidd_urllist.readlines() if 'SRGB' in url]

    # Asynchronously download each URL through FTP
    url = urls[0]
    res = urlretrieve(url)
    print(res)
    # for url in urls:
    #     with urlopen(url) as request:
    #         with open('file', 'wb')
    #         shutil.copyfileobj()


if __name__ == '__main__':
    main()
