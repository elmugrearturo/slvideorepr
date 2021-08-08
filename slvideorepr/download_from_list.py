import sys
import os
import pickle
import requests

def main(current_list_path, results_folder="../downloads/"):
    with open(current_list_path, "rb") as fp:
        url_list = pickle.load(fp)
    current_url = url_list.pop(0)
    filename = current_url.split("/")[-1]
    response = requests.get(current_url)
    with open(results_folder + filename, "wb") as fp:
        fp.write(response.content)
    with open(current_list_path, "wb") as fp:
        pickle.dump(url_list, fp)

if __name__ == "__main__":
    if os.path.exists(sys.argv[1]):
        if sys.argv[1].endswith(".bin"):
            main(sys.argv[1])
