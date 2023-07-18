# hidden_folk_repository

This is the code used for the following paper:

Manex Agirrezabal, Sidsel Boldsen, Nora Hollenstein (2023), The Hidden Folk: Linguistic Properties encoded in Multilingual Contextual Character Representations, Proceedings of the Workshop on Computation and Written Language (CAWL 2023), pages 6–13, Association for Computational Linguistics

### Utils

This script downloads a word list from the OPUS collection[1]. You need to specify the language code, e.g. `da`, `fi`, `fo`, `en`, and so on.

`sh 01_get_freq_list.sh <LANGCODE>`

This script gets the relevant pronunciation dictionaries from Wikipron. These dictionaries are obtained through the Wikipron Github repository[2].

`sh 02_get_prons.sh`

The next script uses m2m-aligner to align letters and phones from the Wikipron dictionary. We need to specify the pronunciation dictionaries name and the language code. The command shows how this is done for the Danish language.

`sh 03_align_charsphones.sh dan_latn_narrow.tsv da`

In the "ipa" directory, we have to run the following command, which will obtains the phonetic features from phones using the package Ipapy. The command shows how this is done for the Danish language.

`python3 get_ipa_embeddings_bogstavlyd_ipapy.py da`

In the "canine" directory, we will run the following command, which obtains the Canine embeddings of the letters in a list of words.

`python3 get_canine_embeddings_general.py da`

To conclude these two commands are used to run the experiments using a Train/test validation mechanism and also the Leave-One-Letter -Out (LOLO).

`python3 run_experiment_lolo_extra.py da`

`python3 run_experiment.py da`


[1] https://opus.nlpl.eu/
[2] https://github.com/CUNY-CL/wikipron/tree/master/data/scrape/tsv
