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


## Output
<p style="text-align:justify;">The output of ReVulDL is not just a reported line number. Instead the result is a convergent sub-propagation chain or several sub-propagation chains from interpretable machine learning, which could be mapped to a context including a set of statements in descending order of suspiciousness. The context could reflect propagation relationships between suspicious statements may causing reentrancy vulnerability that engineers need to care about. These relationships can facilitate their understanding and analysis the root causes of the vulnerability. As is illustrated in the Figure below, the final result is a context including four statements corresponding the convergent sub-propagation chain. This context with statements in descending order of suspiciousness could help developers understand and locate the vulnerability. In the testing data set, ReVulDL detects 790 vulnerable contracts in total. Among these vulnerable contracts, the number of contracts in which statements having relationships with reentrancy are included in the output context is 684(accounting for 86.58%). Although 106(accounting for 13.42%) of these 790 vulnerable contracts are detected by ReVulDL, the output context could not successfully cover the suspicious statements. When the ranking metric is calculated in the experimental study, we suppose localization of these 13.42% vulnerable contracts as well as the vulnerable contracts could not be detected by ReVulDL is failed and set their location result to be null. For example, in Top-1 of ReVulDL, 20.40% is the number of ReVulDL can locate the vulnerable statements at the first place in the 86.58% vulnerable contracts over the number of all vulnerable contracts in the test data set.</p>

<img src="https://github.com/toolstemp/IAcontract/blob/master/img/fig1.png" width="600"/><br/>

Ranking list on the test set are shown as below:
| Method      |   top-1   |   top-3   |   top-5   |   top_10  |   top_20  |    MFR    |    MAR    |
| ----------- | :-------: | :-------: | :-------: | :-------: | :-------: | :-------: | :-------: |
| ReVulDL     |   20.40%  |   44.05%  |   58.99%  |   70.38%  |   84.05%  |   4.87    |   5.70    |
