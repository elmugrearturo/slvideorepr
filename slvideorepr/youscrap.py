import pickle

import os
import sys

from pytube import YouTube


def main(bin_folder="../bin/", results_folder="~/youscrap_results/"):
    try:
        with open(bin_folder + "video_list.bin", "rb") as fp:
            video_list = pickle.load()
    except:
        print("Not available, run playlist_formatter first with a youtube-dl json dump.")
        sys.exit(0)

    try:
        os.makedirs(results_folder)
    except:
        pass

    if len(video_list) > 0:
        entry = video_list.pop(0)
    
        yt = YouTube(entry[1])
        stream = yt.streams.get_highest_resolution()
        stream.download(results_folder + entry[-1])
        print("Downloaded ", video_list[-2])
        with open(bin_folder + "video_list.bin", "wb") as fp:
            pickle.dump(video_list, fp)
    else:
        os.remove(bin_folder + "to_download.bin")

if __name__ == "__main__":
    main()
