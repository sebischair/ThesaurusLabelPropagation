## Notes
- Before running any of these commands, make sure you have activated this project's virtual environment by running `pipenv shell`.
- `<PROJECT_DIR>` should be the project's absolute path (the path that is printed when executing `pwd` in the project directory).

## Doing a pipeline run
#### Simplest possible pipeline run (using `main.py`)
This is the simplest possible pipeline run. The default hyper-parameters from [base_config.py](../src/base_config.py) are used:
```bash
./main.py -p <PROJECT_DIR>
```
[main.py](../src/main.py) splits the thesaurus into training and test set, generates predictions via the training set, and evaluates the predictions on the test set. It stores the run's configuration and output in `output/XX_runs/<DATE>`.

#### Simplest pipeline run with just 1000 texts (e.g. for development purposes)
```bash
./main.py -p <PROJECT_DIR> -c 'phase1.params.DEV_max_texts=1000'
```
This run sets a hyper-parameter in phase 1 that makes it process just the first 1000 text corpus files. This is useful for development purposes to test that all phases complete successfully. Then, even including word embedding generation, a whole run takes just 1-2 minutes instead of hours.

### Setting multiple parameters at once
```bash
./main.py -p <PROJECT_DIR> -c 'phase1.params.DEV_max_texts=1000' 'phase2.params.embeddings_method="word2vec"' 'phase3.params.options.include_self=True'
```

#### Optimized configuration with with multiple training/test splits (using `multi_run.py`)
This command runs the optimized configuration from our thesis (with fastText embeddings):
```bash
./multi_runs.py -p <PROJECT_DIR> -c \
'phase1.params.actions=["remove_newline_and_carriage_returns","replace_sz_in_muss","replace_paragraphsign_with_word","replace_punctuation_with_space_except_hyphen","discard_words_less_two_characters","transform_to_lowercase","save_as_single_line"]' \
'phase4.params.actions=["transform_to_lowercase","remove_keys_with_space_keep_hyphen","create_n_1_mapping_largest_sysnet_wins_key","remove_keys_not_in_corpus","remove_synsets_with_less_than_two_keys"]' \
'phase2.params.embeddings_method="fasttext"' \
'phase2.params.options.size=400' \
'phase2.params.options.iter=40' \
'phase3.params.options.n_neighbors=12' \
'phase3.params.options.mode="distance"' \
'phase7.params.variant="label_spreading"' \
'phase7.params.options.iter=15' \
&& ntfy send "Best runs attempt successful" \
|| ntfy send "Error"
```
[multi_runs.py](../src/multi_runs.py) generates three different training/test splits of the thesaurus, and trains and evaluates on each of the splits (by calling `main.py` repeatedly) to prevent overfitting on a specific split. All prediction statistics are stored in `output/XX_runs/multi_runs_<DATE>`, we each `main.py` result as a subfolder. The `main.py` evaluation results are combined into an `all_stats.json` file.

**Note**: Here, the Python package `ntfy` ([GitHub](https://github.com/dschep/ntfy)) is used to automatically notify the user via e.g. the Telegram messenger that the run was successful. It is not installed by default. 

## Doing a "Synset Vector" Baseline run
This runs the baseline approach using a specific set of word embeddings and sampled thesaurus (training and test set) and saves the evaluation results into a new `<DATE>-stats-baselineSYNSET_VECTOR.json` file:
```bash
# Note: We **need** to switch to the folder `./src/baselines` for this script to work! Calling the script from outside the folder will result in a ModuleNotFoundError
./purew2v_parameter_studies.py \
-e <PROJECT_DIR>/output/02_word_embeddings/<DATE>/main.txt \
-p1 <PROJECT_DIR>/output/05_thesaurus_sampled/<DATE> \
-m SYNSET_VECTOR \
&& ntfy send "Baseline run successful" \
|| ntfy send "Error"
```
The `k` parameter has to be set manually by editing [purew2v_parameter_studies.py](/src/baselines/purew2v_parameter_studies.py) before the run. By adding the parameter `-s`, the `df_evaluation` table gets stored as well. To prevent overfitting, other thesaurus/test splits can be specified via `-p2` and `-p3`.
