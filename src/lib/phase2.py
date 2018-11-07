import logging
import os
import shutil
import subprocess
import tempfile
from os.path import join

from gensim.models import KeyedVectors
from gensim.models.fasttext import FastText
from gensim.models.word2vec import LineSentence, Word2Vec
from gensim.scripts.glove2word2vec import glove2word2vec

from lib.general_helpers import prepare_output_dir, find_cached_output

# Phase 2: Embedding generation
# Depending on: Phase 1

def run_phase2(_config):
    _output_dir = prepare_output_dir(_config)
    try:
        run_phase2_core(_config["input"], _config["params"], _output_dir)
        return _output_dir
    except Exception as e:
        shutil.rmtree(_output_dir)
        raise e


def run_phase2_core(_input, params, _output_dir):
    if params["embeddings_method"] in ["fasttext", "word2vec"]:
        if params["embeddings_method"] == "fasttext":
            model_type = FastText
        elif params["embeddings_method"] == "word2vec":
            model_type = Word2Vec
        else:
            raise Exception
        logging.info(params["embeddings_method"])

        corpus = LineSentence(join(_input, "main.txt"))
        model = model_type(size=params["options"]["size"], window=params["options"]["window"], workers=os.cpu_count(),
                           iter=params["options"]["iter"])
        model.build_vocab(corpus)

        model.train(corpus, total_examples=model.corpus_count, epochs=model.epochs)
    elif params["embeddings_method"] == "glove":
        logging.info(params["embeddings_method"])
        model = train_glove_model(_input, params)
    else:
        logging.info(params["embeddings_method"])
        raise Exception

    model.init_sims(replace=True)  # normalizing
    model.wv.save_word2vec_format(join(_output_dir, "main.txt"), binary=False)


def train_glove_model(_input, params):
    try:
        vocab_file_tmp = tempfile.NamedTemporaryFile(mode="w+")
        cooccurrence_file_tmp = tempfile.NamedTemporaryFile(mode="w+b")
        cooccurrence_shuf_file_tmp = tempfile.NamedTemporaryFile(mode="w+b")
        save_file_tmp = tempfile.NamedTemporaryFile(mode="w+", suffix=".txt")
        word2vec_file_tmp = tempfile.NamedTemporaryFile(mode="w+")

        CORPUS = join(_input, "main.txt")
        VOCAB_FILE = vocab_file_tmp.name
        COOCCURRENCE_FILE = cooccurrence_file_tmp.name
        COOCCURRENCE_SHUF_FILE = cooccurrence_shuf_file_tmp.name
        BUILDDIR = params["options"]["glove_builddir"]
        SAVE_FILE = save_file_tmp.name[:-4] # glove will add ".txt" manually to file name
        VERBOSE = 1
        MEMORY = 8.0
        VOCAB_MIN_COUNT = 5
        VECTOR_SIZE = params["options"]["size"]
        MAX_ITER = params["options"]["iter"]
        WINDOW_SIZE = params["options"]["window"]
        NUM_THREADS = os.cpu_count()
        run_cmd("{}/vocab_count -min-count {} -verbose {} < {} > {}".format(
            BUILDDIR,
            VOCAB_MIN_COUNT,
            VERBOSE,
            CORPUS,
            VOCAB_FILE
        ))
        run_cmd("{}/cooccur -memory {} -vocab-file {} -verbose {} -window-size {} < {} > {}".format(
            BUILDDIR,
            MEMORY,
            VOCAB_FILE,
            VERBOSE,
            WINDOW_SIZE,
            CORPUS,
            COOCCURRENCE_FILE
        ))
        run_cmd("{}/shuffle -memory {} -verbose {} < {} > {}".format(
            BUILDDIR,
            MEMORY,
            VERBOSE,
            COOCCURRENCE_FILE,
            COOCCURRENCE_SHUF_FILE
        ))
        run_cmd(
            "{}/glove -save-file {} -threads {} -input-file {} -iter {} -vector-size {} -vocab-file {} -verbose {}".format(
                BUILDDIR,
                SAVE_FILE,
                NUM_THREADS,
                COOCCURRENCE_SHUF_FILE,
                MAX_ITER,
                VECTOR_SIZE,
                VOCAB_FILE,
                VERBOSE
            ))

        glove2word2vec(save_file_tmp.name, word2vec_file_tmp.name)
        model = KeyedVectors.load_word2vec_format(word2vec_file_tmp.name)
    finally:
        vocab_file_tmp.close()
        cooccurrence_file_tmp.close()
        cooccurrence_shuf_file_tmp.close()
        save_file_tmp.close()
        word2vec_file_tmp.close()
    return model


def run_cmd(cmd):
    logging.info(cmd)
    p = subprocess.Popen(
        cmd,
        shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    while True:
        line = p.stdout.readline().decode().rstrip()
        logging.info(line)
        if p.poll() is not None:
            break


def phase2(_config):
    logging.info("Phase 2 starting")
    _cached_output_dir = find_cached_output(_config["output"], _config["input"], _config["params"])

    if _cached_output_dir is not None:
        return _cached_output_dir

    return run_phase2(_config)
