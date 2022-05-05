### This is the online repository of smart contract vulnerability detection and localization.
## Task Definition

Detect and Localize reentrancy vulnerabilities in smart contract.

## Dataset

The dataset we use is [SmartBugs Wild Dataset](https://github.com/smartbugs/smartbugs-wild/tree/master/contracts) and filtered following the paper [Empirical Review of Automated Analysis Tools on 47,587 Ethereum Smart Contracts](https://arxiv.org/abs/1910.10601).

The tools analysis results we use is [Vulnerability Analysis of Smart Contracts using SmartBugs](https://github.com/smartbugs/smartbugs-results) and filtered following the paper [Empirical Review of Automated Analysis Tools on 47,587 Ethereum Smart Contracts](https://arxiv.org/abs/1910.10601).

### Data Format

1. dataset/data.jsonl is stored in jsonlines format. Each line in the uncompressed file represents one contract.  One row is illustrated below.

   - **contract:** the smart contract

   - **idx:** index of the contract
  
   - **address:** the smart contract's address

2. train.txt/valid.txt/test.txt provide examples, stored in the following format:    **idx	label**

## Example
We provide an example,line 40 is the vulneralbility statement and ranked 1st in the ranking list
```shell
cd demo
cd dataset
unzip data.zip
cd ..
python detect.py dev
cd saved_models
vim predictions.txt
vim programs.txt
```

## Dependency

- python version: python3.7.6
- pip install torch
- pip install transformers
- sudo apt install build-essential
- pip install tree_sitter
- pip install sklearn


## Vulnerability localization

```shell
cd contract_all
cd dataset
unzip dataset.zip
cd ..
python detect.py dev
```
### Get the result
We provide the result ranking list in txt form
```shell
cd contract_all
cd saved_models
vim predictions.txt
```

### Docker implementation
We also provide docker implementation
```shell
docker pull zz8477/ubuntu18.04:IA_contract
docker run -it --gpus all zz8477/ubuntu18.04:IA_contract /bin/bash

#run demo
cd /home/IA_contract_github/demo/
python detect.py dev
cd saved_models/
vim predictions.txt
vim programs.txt

#run whole project
cd /home/IA_contract_github/contract_all/
python detect.py dev
cd saved_models/
vim predictions.txt
```

## Result

Recall,Precision and F1 on the test set are shown as below:

| Method      |  Recall   | Precision |    F1     |
| ----------- | :-------: | :-------: | :-------: |
| Honeybadger |   0.51  |   0.87  |   0.51  |
| Manticore   |   0.50  |   0.50  |   0.50  |
| Mythril     |   0.52  |   0.50  |   0.50  |
| osiris      |   0.54  |   0.59  |   0.55  |
| Oyente      |   0.54  |   0.66  |   0.57  |
| securify    |   0.55  |   0.53  |   0.53  |
| Slither     |   0.66  |   0.52  |   0.53  |
| smartcheck  |   0.71  |   0.79  |   0.74  |
| DR-GCN      |   0.81  |   0.72  |   0.76  |
| TMP         |   0.83  |   0.74  |   0.78  |
| Vanilla-RNN |   0.59  |   0.50  |   0.51  |
| LSTM        |   0.68  |   0.52  |   0.59  |
| GRU         |   0.71  |   0.53  |   0.61  |
| GCN         |   0.78  |   0.71  |   0.74  |
| CGE         |   0.87  |   0.85  |   0.86  |
| ReVulDL     | **0.93**| **0.92**| **0.93**|

Ranking list on the test set are shown as below:
| Method      |   top-1   |   top-3   |   top-5   |   top_10  |   top_20  |    MFR    |    MAR    |
| ----------- | :-------: | :-------: | :-------: | :-------: | :-------: | :-------: | :-------: |
| ReVulDL     |   20.40%  |   44.05%  |   58.99%  |   70.38%  |   84.05%  |   4.87    |   5.70    |
