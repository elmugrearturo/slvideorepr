# Cleans the Special:FileLists page from wikisign
# and saves the hyperlink list to a pickle
import pickle
import sys
import os
import re
import urllib

def main(filepath, bin_folder="../bin/"):
    with open(filepath, "r") as fp:
        all_text = fp.read()
    urlpattern = re.compile(r'href="http(.*)"')
    origin = "http" + urlpattern.search(all_text)[1]
    prefix = "http://" + urllib.parse.urlparse(origin).netloc
    partial_results = re.findall(r'[(][<]a href="(.*)"[>]archivo[<][/][a][>][)]', all_text)
    
    # Changing (1) to (2)
    #http://lsc.wikisign.org/upload/7/7d/Carboni.flv (1)
    #http://lsc.wikisign.org/upload/transcoded/7/7d/Carboni.flv/Carboni.flv.480p.mp4 (2)
    results = []
    for suffix in partial_results:
        splitted_suffix = [x for x in suffix.split("/") if len(x) > 0]

        splitted_url = [prefix] + [splitted_suffix[0]]
        splitted_url += ["transcoded"] + splitted_suffix[1:]
        splitted_url += [splitted_suffix[-1] + ".480p.mp4"]

        results.append("/".join(splitted_url))

    with open(bin_folder + urllib.parse.urlparse(origin).netloc + ".bin", "wb") as fp:
        pickle.dump(results, fp)

if __name__ == "__main__" :
    if os.path.exists(sys.argv[1]):
        main(sys.argv[1])
