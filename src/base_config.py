BASE_CONFIG = {
    "phase1": {  # Corpus preprocessing
        "input": "data/RW40jsons",
        "params": {
            "actions": [
                "remove_newline_and_carriage_returns",
                "replace_sz_in_muss", # or leave out, or "replace_all_sz"
                "replace_paragraphsign_with_word",
                "replace_non_german_chars_with_space",
                "discard_words_less_two_characters",
                "transform_to_lowercase",
                "save_as_single_line"  # or "save_as_multiple_lines"
            ],
            "DEV_max_texts": None
        },
        "output": "output/01_corpus_preprocessed"
    },
    "phase2": {  # Embedding generation
        "input": "output/01_corpus_preprocessed",
        "params": {
            "embeddings_method": "fasttext",  # or "word2vec" or "glove"
            "options": {
                "size": 100,
                "iter": 5,
                "window": 5,
                "glove_builddir": "/home/markus/glove/build" # only needed for "glove"
            }
        },
        "output": "output/02_word_embeddings"
    },
    "phase3": {  # Graph generation
        "input": "output/02_word_embeddings",
        "params": {
            "variant": "knn", # or "radius"
            "options": {
                "n_neighbors": 3,  # only needed for variant "knn"
                "radius": 0.1,  # only needed for variant "radius"
                "mode": "connectivity",  # or "distance"
                "include_self": False,
                "force_symmetric": True
            }
        },
        "output": "output/03_corpus_graph",
    },
    "phase4": {  # Thesaurus preprocessing
        "inputs": {
            "thesaurus": "data/german_relat_pretty-20180605.json",
            "word_embeddings": "output/02_word_embeddings"
        },
        "params": {
            "actions": [
                "transform_to_lowercase",
                "remove_keys_with_space_or_hyphen",
                "create_n_1_mapping_largest_sysnet_wins_key",
                "remove_keys_not_in_corpus",
                "remove_synsets_with_less_than_two_keys"
            ]
        },
        "output": "output/04_thesaurus_preprocessed"
    },
    "phase5": {  # Thesaurus sampling
        "input": "output/04_thesaurus_preprocessed",
        "params": {
            "variant": "sample_each_concept_frac", # or "all_as_training", or "sample_each_concept_n"
            "options": {
                "frac": 0.5, # only needed for "sample_each_concept_frac"
                "n_test": 1  # only needed for "sample_each_concept_n"
            },
            "rng_num": 1
        },
        "output": "output/05_thesaurus_sampled",
    },
    "phase6": {  # Graph labeling (with training data)
        "inputs": {
            "corpus_graph": "output/03_corpus_graph",
            "thesaurus": "output/05_thesaurus_sampled"
        },
        "params": {
            "variant": "normal"
        },
        "output": "output/06_corpus_graph_training_labeled"
    },
    "phase7": {  # Label propagation
        "input": "output/06_corpus_graph_training_labeled",
        "params": {
            "variant": "label_propagation", # or "label_spreading"
            "options": {
                "iter": 3,
                "alpha": 0.2 # only needed for "label_spreading"
            }
        },
        "output": "output/07_corpus_graph_labels_propagated"
    },
    "phase8": {  # Evaluation
        "inputs": {
            "thesaurus": "output/05_thesaurus_sampled",
            "predictions": "output/07_corpus_graph_labels_propagated"
        },
        "params": {
            "variant": "normal"
        },
        "output": "output/08_propagation_evaluation"
    }
}
