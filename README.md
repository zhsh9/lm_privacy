# Selective Differential Privacy for Language Modeling
This is the repo for the paper [Selective Differential Privacy for Language Modeling](https://arxiv.org/pdf/2108.12944.pdf]. You can cite the paper using the following bibtex.

```
@article{shi2021selective,
  title={Selective Differential Privacy for Language Modeling},
  author={Shi, Weiyan and Cui, Aiqi and Li, Evan and Jia, Ruoxi and Yu, Zhou},
  journal={arXiv preprint arXiv:2108.12944},
  year={2021}
}
```

The language modeling is based on code here, https://github.com/pytorch/examples/tree/master/word_language_model

At the time of implementation, `privacy_engine.detach()` didn't work (please refer to this issue, https://github.com/pytorch/opacus/issues/163), so we had to manually pass the `privacy_engine` into `train_partialdp_rnn` in main.py for the private update step in SDPSGD (attach the engine), and pass `privacy_engine=None` for the non-private update step in SDPSGD (detach the engine). Now the detach bug is fixed, and we should be able to directly detach and attach without the hack.

Also, multi-GPU was not supported, so the SDP training took about 48 hours for 50 epochs on one single GPU, but now multi-GPU is supported by opacus. Please refer to https://github.com/pytorch/opacus/pull/166, so speedup is possible although I haven't tried it out.  

## How to install
original_privacy_engine.py is the original code for privacy engine
pe.py is the to-do my own privacy engine
```
# git clone the repo first
# create an environment called "privacy"
# conda env create -f environment.yml
conda env create -f nlp.yaml

# for installing cudatoolkit if cudatoolkit has something wrong with it.
conda activate privacy
conda conda install pytorch==1.7.1 cudatoolkit=11.0 -c pytorch

# just in case, permission to execute the file
chmod +x env_transfer.sh

# create the folders using this script
./env_transfer.sh

# install the correct torch version
# pip uninstall torch
# pip install torch==1.7.1
# nlp.yaml does not need this.

# As a quick fix to achieve the selective dp, simply copy privacy_engine.py in this git repo to the privacy_engine in opacus (usually it's in ~/anaconda3/envs/privacy/lib/python3.8/site-packages/opacus/privacy_engine.py), should import in the future.
# first make a copy of the original privacy_engine.py
mv ~/miniconda3/envs/privacy/lib/python3.8/site-packages/opacus/privacy_engine.py ~/miniconda3/envs/privacy/lib/python3.8/site-packages/opacus/privacy_engine_original.py
# then copy the privacy_engine.py
cp privacy_engine.py ~/miniconda3/envs/privacy/lib/python3.8/site-packages/opacus/privacy_engine.py
# use miniconda3 instead of anaconda3
```

## For the dialogue system task: how to get the CUSTOMERSIM dataset
```
cd simdial_privacy/
python multiple_domains.py --domain track_package --complexity mix --train_size 10000 --test_size 1000 --valid_size 1000 --save_dir output
```

## The dataset
* data/wikitext-2: original wikitext from https://github.com/pytorch/examples/tree/master/word_language_model/data/wikitext-2
* data/wikitext-2-add10b: wikitext + 10 inserted canary "My ID is 341752." in the train.txt
* data/wikitext-2-add10b-normalized/missing_digits: normalized wikitext+ 10 inserted canary "My ID is 341752." in the train.txt.  All the digits in wikitext are replaced with <num>, except for the "341752" in the canary, to mimic the situation where the data sanitization misses some secrets. 
* data/wikitext-2-add10b-normalized/not_missing_digits: normalized wikitext+ 10 inserted canary "My ID is 341752." in the train.txt.  All the digits in wikitext are replaced with <num>, including the "341752" in the canary, to mimic the situation where the data sanitization properly protects all the secrets. 

* data/simdial: simdial + 10 inserted canary "My ID is 341752." in the training data


## how to train the SDP models
```
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
```

## how to run the canary insertion attack
```
python attacks/canary_insertion.py --cuda cuda:0 -bs 256 --checkpoint model/nodp/20210515/122306 --outputf attacks/canary_insertion/nodp_normalized_nomiss/nodp_seed300.csv
```

## how to run the member inference attack
```
python attacks/mem_inference.py --cuda cuda:0 --data_type doc --data data/wikitext-2-add10b -bs 64 --N 1000 --checkpoint model/nodp/20210515/122306 --outputf attacks/membership_inference/nodp_normalized_nomiss/nodp_seed300.csv
```
