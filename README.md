# xRAG4TS
This repo contains code of xRAG4TS.

## Dependencies for Backbone (TransGTR)
- Python 3.8.5
- Pytorch 1.9.0


### Step 0, Pre-process data. 
We provide data pre-processing scripts in `data_scripts/`. You should run the following data preprocessing scripts. 

`python3 data_scripts/generate_training_data_SG.py --history_seq_len 288 --future_seq_len 12`

`python3 data_scripts/generate_training_data_SG.py --history_seq_len 12 --future_seq_len 12`

`python3 data_scripts/generate_training_Nottingham.py --history_seq_len 288 --future_seq_len 12`

`python3 data_scripts/generate_training_Nottingham.py --history_seq_len 12 --future_seq_len 12`

where `--history_seq_len 288` is used to train the node feature network, and `--history_seq_len 12` is used to train the forecasting model. 

### Step 1, Train a Backbone (TransGTR) to obtain the prediction tokens. 
 

### Step 2, Run spatialencoder.py to obtain embeddings


### Step 3, Run retriever.py for source semantic mapping

### Step 4, Run correlation_extract.py to obtain reasonle_hint for LLM module

### Step 5, Use LLM_predictor and LLM_eva to train xRAG4TS
