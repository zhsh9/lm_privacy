#!/bin/bash

## language modeling, data = "data/wikitext-2-add10b"
# SDPSGD, turn both `-dp` and `--partial` on,
python -u main.py -bs 7 --lr 0.1 -dp --cuda cuda:0 -partial -norm 1e-3  --sigma 0.5 --seed 0 2>&1 | tee logs/partial_dp/20210421/1021/nohidden_lr0.1_norm1e-3_sigma0.5_seed0

# DPSGD, turn only `-dp` on
python -u main.py --epochs 100 -bs 7 --lr 0.05 -dp --cuda cuda:0 -norm 0.1 --seed 1111 2>&1 | tee logs/dp/20210424/repeat/lr0.05_sigma0.5_norm0.1_seed1111

# No DP
python -u main.py -bs 16 --lr 20 --data data/wikitext-2-add10b --cuda cuda:0 --seed 0 2>&1 | tee logs/nodp/20210416/2354/bs16_see0.log

## dialogue system, data = "data/simdial"
# SDPSGD, set `--data data/simdial --data_type dial`
python -u main.py --lr 0.1 --data data/simdial --data_type dial --cuda cuda:0 -dp -partial -bs 3 --sigma 0.7 -norm 5e-3 --epochs 50 2>&1 | tee logs/partial_dp/dialog/20210430/sigma0.7_norm5e-3

# DPSGD
python -u main.py --lr 0.1 --data data/simdial --data_type dial --cuda cuda:0 -dp -bs 1 --sigma 0.6 -norm 1e-2 --epochs 100 2>&1 | tee logs/dp/dialog/20210430/sigma0.6_norm1e-2_bs1_100epochs

# No DP
python -u main.py -bs 16 --lr 20 --data data/simdial --data_type dial --cuda cuda:0 --log-interval 10 --seed 123 2>&1 | tee logs/nodp/dialog/20210501/dialog_bs16_seed123.log


# missed experiments
# data sanitization, missed the secret in the canary
python -u main.py -bs 16 --lr 20 --data data/wikitext-2-add10b-normalized/missing_digits --cuda cuda:0 2>&1 | tee logs/nodp/normalized/20210426/lstm.log

# SDPSGD, turn on `-missing_digits `
python -u main.py -bs 7 --lr 0.1 -dp --cuda cuda:0 -partial -norm 1e-3  --sigma 0.5 -missing_digits --data data/wikitext-2-add10b --epochs 100 --seed 1111 2>&1 | tee logs/partial_dp/missed/20210426/lr0.1_sigm0.5_norm1e-3_seed1111_miss10.log
