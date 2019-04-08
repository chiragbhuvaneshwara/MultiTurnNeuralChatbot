"""
Grid search for searcher parameters in chatbot conversation
"""

import logging
import os
import subprocess


SEARCHERS = ["beam", "mmi-antilm", "mmi-bidi", "idf-beam"]
PENALTY_START = 5  # \ 10.0
PENALTY_STEP = 1
PENALTY_END = 20

LENGTH_START = -1
LENGTH_STEP = -1
LENGTH_END = -3

BEAM_START = 0
BEAM_STEP = 5
BEAM_END = 50

CUTOFF_START = 1
CUTOFF_STEP = 1
CUTOFF_END = 10


def write_config(filename, searcher, penalty, length_penalty, beam_size, language_model, lm_order, cutoff, corpus):
    """Write a config file with the specified parameters."""
    with open(filename, "w") as f:
        f.write("[CHOICE]\n")
        f.write(f"Searcher = {searcher}\n")
        f.write(f"[{searcher.upper()}]\n")
        f.write(f"Penalty = {penalty}\n")
        f.write(f"LengthPenalty = {length_penalty}\n")
        f.write(f"BeamSize = {beam_size}\n")
        f.write(f"LanguageModel = {language_model}\n")
        f.write(f"LMOrder = {lm_order}\n")
        f.write(f"CutOff = {cutoff}\n")
        f.write(f"Corpus = {corpus}\n")


def create_all_configs(file_path):
    directory = os.path.dirname(file_path)
    logging.debug(f"dir {directory}")
    if os.path.exists(directory):
        raise OSError(f"{file_path} exists already")
    os.makedirs(directory)
    searcher = "beam"
    new_dir = os.path.join(directory, "beam")
    logging.debug(f"new_dir {new_dir}")
    os.makedirs(new_dir)
    for beam_size in range(BEAM_START, BEAM_END, BEAM_STEP):
        filename = os.path.join(new_dir, f"beam_size_{beam_size}.ini")
        logging.debug(f"filename {filename}")
        write_config(filename, searcher, None, None, beam_size, None, None, None, None)

    # mmi-antilm
    searcher = "mmi-antilm"
    language_model = "token_MLE_2_lm.pkl"
    lm_order = 2
    new_dir = os.path.join(directory, "mmi-antilm")
    os.makedirs(new_dir)
    for penalty in [x / 10 for x in range(PENALTY_START, PENALTY_END, PENALTY_STEP)]:
        for length_penalty in range(LENGTH_START, LENGTH_END, LENGTH_STEP):
            for cutoff in range(CUTOFF_START, CUTOFF_END, CUTOFF_STEP):
                filename = os.path.join(new_dir, f"penalty_{penalty}_length_penalty_{length_penalty}_cutoff_{cutoff}.ini")
                write_config(filename, searcher, penalty, length_penalty, None, language_model, lm_order, cutoff, None)

    # mmi-bidi
    searcher = "mmi-bidi"
    new_dir = os.path.join(directory, "mmi-bidi")
    os.makedirs(new_dir)
    for penalty in [x / 10 for x in range(PENALTY_START, PENALTY_END, PENALTY_STEP)]:
        for length_penalty in range(LENGTH_START, LENGTH_END, LENGTH_STEP):
            for beam_size in range(BEAM_START, BEAM_END, BEAM_STEP):
                filename = os.path.join(new_dir, f"penalty_{penalty}_length_penalty_{length_penalty}_beam_size_{beam_size}.ini")
                write_config(filename, searcher, penalty, length_penalty, beam_size, None, None, None, None)

    # idf-beam
    searcher = "idf-beam"
    corpus = "/home/ca/Documents/Uni/Master_3/Sopro_Neural_Networks/sopro-chatbot/data/cornell movie-dialogs corpus/formatted_movie_lines.txt"
    new_dir = os.path.join(directory, "idf_beam")
    os.makedirs(new_dir)
    for penalty in [x / 10 for x in range(PENALTY_START, PENALTY_END, PENALTY_STEP)]:
        for length_penalty in range(LENGTH_START, LENGTH_END, LENGTH_STEP):
            for beam_size in range(BEAM_START, BEAM_END, BEAM_STEP):
                filename = os.path.join(new_dir, f"penalty_{penalty}_length_penalty_{length_penalty}_beam_size_{beam_size}.ini")
                write_config(filename, searcher, penalty, length_penalty, beam_size, None, None, None, corpus)


def run_all_configs(root_dir, input_file, model_file):
    """Run a conversation with all config files in root_dir."""
    for subdir, dirs, files in os.walk(root_dir):
        for f in files:
            if not f.endswith(".ini"):
                # skip log files, etc.
                continue
            file_path = os.path.join(subdir, f)
            log_file = os.path.join(subdir, f"log_{f[:-4]}.txt")
            args = ["python", "interface_conversation.py", f"--input", input_file, f"--log", log_file, f"--searcher_config", file_path, model_file]
            print(args)
            subprocess.run(args)

def main():
    file_path = "./grid_search/"
    # create_all_configs(file_path)
    model_file = "../4000_checkpoint.tar"
    input_file = "example_sentences.txt"
    run_all_configs(file_path, input_file, model_file)



if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    main()
    print("finished")
    # parser = argparse.ArgumentParser(description="Gridsearch for chatbot.")
    # parser.add_argument("--beam_start", type=int, default=1)
    # parser.add_argument("--beam_step", type=int, default=2)
    # parser.add_argument("--beam_end", type=int, default=30)
