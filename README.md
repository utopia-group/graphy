![Graphy](logo.png)

# Code artifact for Graphy

This repository contains a snapshot of the source code for Graphy, a data visualization synthesis tool from natural language.

## Installation

Requirement: Python3 (version **3.6**), we suggest using the conda environment.

0. first install pip: `conda install pip`
1. run requirements `pip install -r requirements.txt`
2. To run the NL4DV baseline, please install the stanford nlp packages including the files `stanford-english-corenlp-2018-10-05
-models.jar`, `stanford-parser.jar`
3. To run draco-related baseline, please install clingo-cffi using the command `python3 -m pip install --user --upgrade --extra-index-url https://test.pypi.org/simple/ clingo-cffi`.
4. To run `preprocess.py`, spacy english library is required using `python -m spacy download en_core_web_trf` (you need to first install spacy)
5. To run the neural_parser, trained models are required. We provided all the models necessary to reproduce the evaluation here: https://drive.google.com/file/d/1UkbPoS-cueZ4J5MFMwAOaRje3woVNus_/view?usp=sharing. After downloading the zip file, just directly unzip the file and it should work. 
  
If there are more installation necessary to run this package, please update `README` and `requirements.txt`.

## Running evaluation

Command to run the evaluation script is:

`python run_eval.py --eval_dataset $eval_dataset --top_k $k` where `$eval_dataset` can be `cars`, `movies` or `superstore`. Setting the `$k` will get you top-k synthesized results.

The above command should be good to run the Graphy mode. To run the baselines, enable the following additional flags:

- nl4dv (Sec 8.1): `--nl4dv`
- draco-nl (Sec 8.1): `--enum`
- base-only (Sec 8.2): `--no_qualifier`
- prov-only (Sec 8.2): `--no_table`
- table-only (Sec 8.2): `--no_prov`

Please contact the author or submit an issue if interested in the author's implementation of NcNet (Sec 8.1) and Bart-Vis (Sec 8.1).

## Running user query

To utilize graphy to run additional queries on the supporting dataset, please look at `run_example.py` for reference. 

## Disclaimer 

This is a prototype tool and would recommend caution when building on top of it :) 
