import pickle
import sys
import os

def main(all_downloaded_path, bin_folder="../bin/"):

    with open(bin_folder + "video_list.bin", "rb") as fp:
        video_list = pickle.load(fp)

    downloaded = []
    with open(all_downloaded_path, "r") as fp:
        for line in fp:
            downloaded.append(line.replace("\n", "").replace(".mp4", ""))

    new_video_list = []
    for i in range(len(video_list)):
        entry_id, url, title, name = video_list[i]
        if title not in downloaded:
            new_video_list.append((entry_id, url, title, name))
    with open(bin_folder + "video_list.bin", "wb") as fp:
        pickle.dump(new_video_list, fp)

if __name__ == "__main__":
    if os.path.exists(sys.argv[1]):
        main(sys.argv[1])
