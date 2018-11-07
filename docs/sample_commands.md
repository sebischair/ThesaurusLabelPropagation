## Doing a pipeline run
#### Simplest possible pipeline run (using `main.py`)
This is the simplest possible pipeline run. The default hyper-parameters from [base_config.py](../src/base_config.py) are used:
```bash
./src/main.py -p <PROJECT_DIR>
```
[main.py](../src/main.py) splits the thesaurus into training and test set, generates predictions via the training set, and evaluates the predictions on the test set. It stores the run's configuration and output in `output/XX_runs/<DATE>`.

#### Optimized configuration with with multiple training/test splits (using `multi_run.py`)
This command runs the optimized configuration from our thesis (with fastText embeddings):
```bash
./src/multi_runs.py -p <PROJECT_DIR> -c \
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
[multi_runs.py](../src/multi_runs.py) generates three different training/test splits of the thesaurus, and trains and evaluates on each of the splits (by calling `main.py` repeatedly) to prevent oferfitting on a specific split. All prediction statistics are stored in `output/XX_runs/multi_runs_<DATE>`, we each `main.py` result as a subfolder. The `main.py` evaluation results are combined into an `all_stats.json` file.

**Note**: Here, the Python package `ntfy` ([GitHub](https://github.com/dschep/ntfy)) is used to automatically notify the user via e.g. the Telegram messenger that the run was successful. It is not installed by default. 

## Doing a "Synset Vector" Baseline run
This runs the baseline approach using a specific set of word embeddings and sampled thesaurus (training and test set):
```bash
./src/baselines/purew2v_parameter_studies.py \
-e <PROJECT_DIR>/output/02_word_embeddings/<DATE>/main.txt \
-p1 <PROJECT_DIR>/output/05_thesaurus_sampled/<DATE> \
-m SYNSET_VECTOR \
&& ntfy send "Baseline run successful" \
|| ntfy send "Error"
```
The `k` parameter has to be set manually by editing [purew2v_parameter_studies.py](/src/baselines/purew2v_parameter_studies.py) before the run.