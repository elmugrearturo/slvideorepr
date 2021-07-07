import pickle
import sys
import os

from unidecode import unidecode
import re

from collections import OrderedDict

template = "https://www.youtube.com/watch?v="

def main(json_path, bin_folder="../bin/"):
    try:
        os.makedirs(bin_folder)
    except:
        pass
    video_list = []
    with open(json_path, "r") as fp:
        for line in fp:
            entry = eval(line)
            video_title = entry["title"]
            file_name = unidecode(re.sub("\s+", "_", video_title))
            video_list.append((entry["id"], 
                               template + entry["id"], 
                               video_title, 
                               file_name))
    with open(bin_folder + "video_list.bin", "wb") as fp:
        pickle.dump(video_list, fp)

if __name__ == "__main__":
    if os.path.exists(sys.argv[1]):
        main(sys.argv[1])
