import pickle
import os
from pytube import Playlist, YouTube

#playlist = "https://www.youtube.com/watch?v=JfiBXL5sFuQ&list=UUgQaLT6vhaucnLT2sarx0Kg&index=1"
playlist_url = "https://www.youtube.com/playlist?list=UUgQaLT6vhaucnLT2sarx0Kg"

def main(bin_folder="../bin/"):
    try:
        with open(bin_folder + "to_download.bin", "rb") as fp:
            to_download = pickle.load()
    except:
        to_download = []
        playlist = Playlist(playlist_url)
        for video_url in playlist.video_urls:
            print(video_url)
        import ipdb;ipdb.set_trace()
        with open(bin_folder + "to_download.bin", "wb") as fp:
            pickle.dump(to_download, fp)

    if len(to_download) > 0:
        pass
    else:
        os.remove(bin_folder + "to_download.bin")

if __name__ == "__main__":
    main()
