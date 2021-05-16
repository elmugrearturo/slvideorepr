import sys
import os
import pathlib

from extraction import process_corpora

def main(corpus_dir, measure):
    # Decide output folder
    results_dir = pathlib.Path(__file__).parent.parent.absolute() / "results"
    results_dir /= measure

    # Start video processing
    process_corpora(corpus_dir, results_dir, measure)

if __name__ == "__main__":
    has_path = False
    if len(sys.argv) > 1:
        if os.path.exists(sys.argv[1]):
            if os.path.isdir(sys.argv[1]):
                has_path = True
    
    try:
        measure = sys.argv[2]
        if measure != "mse":
            measure = "mse"
    except:
        measure = "ssim"

    print("Selected measure: %s" % measure)
    if has_path:
        main(sys.argv[1], measure)
    else:
        print("Usage: %s CORPUS_DIR" % sys.argv[0])

