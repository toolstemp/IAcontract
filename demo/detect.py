from __future__ import absolute_import, division, print_function
import sys
import logging.config
from configparser import ConfigParser
from tree_sitter import Language, Parser
from parser import (remove_comments_and_docstrings,
                    tree_to_token_index,
                    index_to_code_token,
                    tree_to_variable_index)
from parser import DFG_solidity

import argparse
import glob
import logging
import os
import pickle
import random
import re
import shutil
import json
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from transformers import (WEIGHTS_NAME, AdamW, get_linear_schedule_with_warmup,
                          RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer)
from tqdm import tqdm, trange
import multiprocessing
from model import Model

from sklearn import metrics
from torch.nn.functional import cross_entropy
cpu_cont = 16


dfg_function = {
    'solidity': DFG_solidity
}

# load parsers
parsers = {}
for lang in dfg_function:
    LANGUAGE = Language('parser/my-languages.so', lang)
    parser = Parser()
    parser.set_language(LANGUAGE)
    parser = [parser, dfg_function[lang]]
    parsers[lang] = parser


# initialize logger
logging.config.fileConfig("logging.cfg")
logger = logging.getLogger("root")


class RuntimeContext(object):
    """ runtime enviroment
    """

    def __init__(self):
        """ initialization
        """
        # configuration initialization
        config_parser = ConfigParser()
        config_file = self.get_config_file_name()
        config_parser.read(config_file, encoding="UTF-8")
        sections = config_parser.sections()

        file_section = sections[0]
        self.train_data_file = config_parser.get(
            file_section, "train_data_file")
        self.output_dir = config_parser.get(file_section, "output_dir")
        self.eval_data_file = config_parser.get(file_section, "eval_data_file")
        self.test_data_file = config_parser.get(file_section, "test_data_file")

        base_section = sections[1]
        self.model_name_or_path = config_parser.get(
            base_section, "model_name_or_path")
        self.config_name = config_parser.get(base_section, "config_name")
        self.tokenizer_name = config_parser.get(base_section, "tokenizer_name")

        parameters_section = sections[2]
        self.code_length = int(config_parser.get(
            parameters_section, "code_length"))
        self.data_flow_length = int(config_parser.get(
            parameters_section, "data_flow_length"))
        self.train_batch_size = int(config_parser.get(
            parameters_section, "train_batch_size"))
        self.eval_batch_size = int(config_parser.get(
            parameters_section, "eval_batch_size"))
        self.gradient_accumulation_steps = int(config_parser.get(
            parameters_section, "gradient_accumulation_steps"))
        self.learning_rate = float(config_parser.get(
            parameters_section, "learning_rate"))
        self.weight_decay = float(config_parser.get(
            parameters_section, "weight_decay"))
        self.adam_epsilon = float(config_parser.get(
            parameters_section, "adam_epsilon"))
        self.max_grad_norm = float(config_parser.get(
            parameters_section, "max_grad_norm"))
        self.max_steps = int(config_parser.get(
            parameters_section, "max_steps"))
        self.warmup_steps = int(config_parser.get(
            parameters_section, "warmup_steps"))
        self.seed = int(config_parser.get(parameters_section, "seed"))
        self.epochs = int(config_parser.get(parameters_section, "epochs"))
        self.n_gpu = torch.cuda.device_count()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device

    def get_config_file_name(self):
        """ get the configuration file name according to the command line parameters
        """
        argv = sys.argv
        config_type = "dev"  # default configuration type
        if None != argv and len(argv) > 1:
            config_type = argv[1]
        config_file = config_type + ".cfg"
        logger.info("get_config_file_name() return : " + config_file)
        return config_file

def JSONLength(obj): 
    size = 0;
    for key in obj:
       size = size + 1 
    return size
 
# remove comments, tokenize code and extract dataflow
def extract_dataflow(url,code, parser, lang):
    # remove comments
    try:
        code = remove_comments_and_docstrings(code, lang)
    except:
        pass
    # obtain dataflow
    if lang == "php":
        code = "<?php"+code+"?>"
    try:
        tree = parser[0].parse(bytes(code, 'utf8'))
        root_node = tree.root_node
        tokens_index = tree_to_token_index(root_node)
        code = code.split('\n')
        code_tokens = [index_to_code_token(x, code) for x in tokens_index]
        index_to_code = {}
        for idx, (index, code) in enumerate(zip(tokens_index, code_tokens)):
            index_to_code[index] = (idx, code)
        try:
            DFG, _ = parser[1](root_node, index_to_code, {})
        except:
            DFG = []
        DFG = sorted(DFG, key=lambda x: x[1])
        DFG_original = DFG
        
        f1 = open('temp/temp_DFG.jsonl','w')
        
        file_DFG_original = {}
        for i in range(len(DFG)):
            file_DFG_original[i] = tuple(DFG[i])
        json.dump(file_DFG_original,f1)
        f1.write('\n')
        #mask process
        chain_num_list = []
        for i in range(len(DFG)):
            if len(DFG[i][-1]) != 0:
                chain_num_list.append(i)
        dfg_all = []
        dfg_all.append(DFG)  #add the original DFG into the list
        
        for i in range(len(chain_num_list)):
            dfg_mask = []
            for j in range(len(DFG)):
                if not j == chain_num_list[i]:
                    dfg_mask.append(DFG[j])
            
            file_DFG_mask = {}
            for i in range(len(dfg_mask)):
                file_DFG_mask[i] = tuple(dfg_mask[i])
            json.dump(file_DFG_mask,f1)
            f1.write('\n')
            
            dfg_all.append(dfg_mask)#add the masked DFG into the list
        
        f1.close()
        
        
        dfg_all_chain = []
        dfg_all_chain.append(DFG_original)
        f2 = open('temp/temp_DFG.jsonl','r') 
        f3 = open('temp/'+str(url)+'_min_DFG_all.jsonl','w')
        file_DFG_original = {}
        for i in range(len(DFG_original)):
            file_DFG_original[i] = tuple(DFG_original[i])
        json.dump(file_DFG_original,f3)
        f3.write('\n')
        for line in tqdm(f2,desc="reading:"):
            js = json.loads(line)
            DFG = []
            for i in range(JSONLength(js)):
                DFG.append(tuple(js[str(i)]))
            # identify critical node in DFG,search vulnerability propagation chain
            critical_idx = []
            for id, e in enumerate(DFG):
                if e[0] == "call" and DFG[id+1][0] == "value":
                    critical_idx.append(DFG[id-1][1])
                    critical_idx.append(DFG[id+2][1])
            lines = []
            for index, code in index_to_code.items():
                if code[0] in critical_idx:
                    line = index[0][0]
                    lines.append(line)
            lines = list(set(lines))
            for index, code in index_to_code.items():
                if index[0][0] in lines:
                    critical_idx.append(code[0])
            critical_idx = list(set(critical_idx))
            max_nums = 0
            cur_nums = -1
            while cur_nums != max_nums and cur_nums != 0:
                max_nums = len(critical_idx)
                for id, e in enumerate(DFG):
                    if e[1] in critical_idx:
                        critical_idx += e[-1]
                    for i in e[-1]:
                        if i in critical_idx:
                            critical_idx.append(e[1])
                            break
                critical_idx = list(set(critical_idx))
                cur_nums = len(critical_idx)
            dfg = []
            for id, e in enumerate(DFG):
                if e[1] in critical_idx:
                    dfg.append(e)
            dfg = sorted(dfg, key=lambda x: x[1])
    
            # Removing independent points
            indexs = set()
            for d in dfg:
                if len(d[-1]) != 0:
                    indexs.add(d[1])
                for x in d[-1]:
                    indexs.add(x)
            new_DFG = []
            for d in dfg:
                if d[1] in indexs:
                    new_DFG.append(d)
            dfg = new_DFG 
#            dfg_all_chain.append(dfg)
            file_DFG_original = {}
            for i in range(len(dfg)):
                file_DFG_original[i] = tuple(dfg[i])
            json.dump(file_DFG_original,f3)
            f3.write('\n')                      
        f2.close()
        f3.close()

    except:
        dfg_all_chain = []
#    return code_tokens, dfg
    return code_tokens, dfg_all_chain, chain_num_list


class InputFeatures(object):
    """A single training/test features for a example."""

    def __init__(self,
                 input_tokens_1,
                 input_ids_1,
                 position_idx_1,
                 dfg_to_code_1,
                 dfg_to_dfg_1,
                 label,
                 url1,
                 chain_num_list,
                 code_tokens_orig,
                 DFG_orig,
                 func

                 ):
        # The code function
        self.input_tokens_1 = input_tokens_1
        self.input_ids_1 = input_ids_1
        self.position_idx_1 = position_idx_1
        self.dfg_to_code_1 = dfg_to_code_1
        self.dfg_to_dfg_1 = dfg_to_dfg_1

        
        self.label = label
        self.url1 = url1
        
        self.chain_num_list = chain_num_list
        self.code_tokens_orig = code_tokens_orig
        self.DFG_orig = DFG_orig
        self.func = func
        


def convert_examples_to_features(item):
    # source
    url1, label, tokenizer, args, cache, url_to_code = item
    parser = parsers['solidity']

    for url in [url1]:
        if url not in cache:
            func = url_to_code[url]
            # extract data flow
            code_tokens_orig, dfg_all, chain_num_list = extract_dataflow(url,func, parser, 'solidity')
 
            for dfg in dfg_all:
                code_tokens = code_tokens_orig
                code_tokens = [tokenizer.tokenize(
                    '@ '+x)[1:] if idx != 0 else tokenizer.tokenize(x) for idx, x in enumerate(code_tokens)]
                ori2cur_pos = {}
                ori2cur_pos[-1] = (0, 0)
                for i in range(len(code_tokens)):
                    ori2cur_pos[i] = (ori2cur_pos[i-1][1],
                                      ori2cur_pos[i-1][1]+len(code_tokens[i]))
                code_tokens = [y for x in code_tokens for y in x]
    
                # truncating
                code_tokens = code_tokens[:args.code_length+args.data_flow_length -
                                          3-min(len(dfg), args.data_flow_length)][:512-3]
                source_tokens = [tokenizer.cls_token] + \
                    code_tokens+[tokenizer.sep_token]
                source_ids = tokenizer.convert_tokens_to_ids(source_tokens)
                position_idx = [i+tokenizer.pad_token_id +
                                1 for i in range(len(source_tokens))]
                dfg = dfg[:args.code_length +
                          args.data_flow_length-len(source_tokens)]
                source_tokens += [x[0] for x in dfg]
                position_idx += [0 for x in dfg]
                source_ids += [tokenizer.unk_token_id for x in dfg]
                padding_length = args.code_length + \
                    args.data_flow_length-len(source_ids)
                position_idx += [tokenizer.pad_token_id]*padding_length
                source_ids += [tokenizer.pad_token_id]*padding_length
    
                # reindex
                reverse_index = {}
                for idx, x in enumerate(dfg):
                    reverse_index[x[1]] = idx
                for idx, x in enumerate(dfg):
                    dfg[idx] = x[:-1]+([reverse_index[i]
                                        for i in x[-1] if i in reverse_index],)
                dfg_to_dfg = [x[-1] for x in dfg]
                dfg_to_code = [ori2cur_pos[x[1]] for x in dfg]
                length = len([tokenizer.cls_token])
                dfg_to_code = [(x[0]+length, x[1]+length) for x in dfg_to_code]
                cache[url] = source_tokens, source_ids, position_idx, dfg_to_code, dfg_to_dfg
                InputFeatures_result=InputFeatures(source_tokens, source_ids, position_idx, dfg_to_code, dfg_to_dfg,label, url1, chain_num_list,code_tokens_orig,dfg_all[0],func)
    
    return InputFeatures_result


class TextDataset(Dataset):
    def __init__(self, tokenizer, args, file_path='train'):
        self.examples_all = []

        self.args = args
        index_filename = file_path

        # load index
        logger.info("Creating features from index file at %s ", index_filename)
        url_to_code = {}
        with open('/'.join(index_filename.split('/')[:-1])+'/data.jsonl') as f:
            for line in f:
                line = line.strip()
                js = json.loads(line)
                url_to_code[js['idx']] = js['contract']

        # load code function according to index
        data = []
        cache = {}
        f = open(index_filename)
        with open(index_filename) as f:
            for line in f:
                line = line.strip()
                url1, label = line.split('\t')
                if url1 not in url_to_code:
                    continue  # jump out of for
                if label == '0':
                    label = 0
                else:
                    label = 1
                data.append((url1, label, tokenizer, args, cache, url_to_code))

        # only use 10% valid data to keep best model
        if 'valid' in file_path:
            data = random.sample(data, int(len(data)*0.1))

        # convert example to input features
        self.examples_all = [convert_examples_to_features(
            x) for x in tqdm(data, total=len(data))]
#        print (self.examples_all[0][0].position_idx_1)
    def __len__(self):
        return len(self.examples_all)

    def __getitem__(self, item):
                chain = self.examples_all[item]
                # calculate graph-guided masked function
                attn_mask_1 = np.zeros((self.args.code_length+self.args.data_flow_length,
                                        self.args.code_length+self.args.data_flow_length), dtype=np.bool)
                # calculate begin index of node and max length of input
                node_index = sum([i > 1 for i in chain.position_idx_1])
                max_length = sum([i != 1 for i in chain.position_idx_1])
                # sequence can attend to sequence
                attn_mask_1[:node_index, :node_index] = True
                # special tokens attend to all tokens
                for idx, i in enumerate(chain.input_ids_1):
                    if i in [0, 2]:
                        attn_mask_1[idx, :max_length] = True
                # nodes attend to code tokens that are identified from
                for idx, (a, b) in enumerate(chain.dfg_to_code_1):
                    if a < node_index and b < node_index:
                        attn_mask_1[idx+node_index, a:b] = True
                        attn_mask_1[a:b, idx+node_index] = True
                # nodes attend to adjacent nodes
                for idx, nodes in enumerate(chain.dfg_to_dfg_1):
                    for a in nodes:
                        if a+node_index < len(chain.position_idx_1):
                            attn_mask_1[idx+node_index, a+node_index] = True
                
                return (torch.tensor(chain.input_ids_1),
                    torch.tensor(chain.position_idx_1),
                    torch.tensor(attn_mask_1),
                    torch.tensor(chain.label),
                    torch.tensor(chain.chain_num_list),
                    chain.code_tokens_orig,
                    chain.DFG_orig,
                    chain.func,
                    chain.url1)

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def train(args, train_dataset, model, tokenizer):
    """ Train the model """

    # build dataloader
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(
        train_dataset, sampler=train_sampler, batch_size=args.train_batch_size, num_workers=4)

    args.max_steps = args.epochs*len(train_dataloader)
    args.save_steps = len(train_dataloader)//10
    args.warmup_steps = args.max_steps//5
    model.to(args.device)

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(
            nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters,
                      lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps,
                                                num_training_steps=args.max_steps)

    # multi-gpu training
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.epochs)
    logger.info("  Instantaneous batch size per GPU = %d",
                args.train_batch_size//args.n_gpu)
    logger.info("  Total train batch size = %d",
                args.train_batch_size*args.gradient_accumulation_steps)
    logger.info("  Gradient Accumulation steps = %d",
                args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", args.max_steps)

    global_step = 0
    tr_loss, logging_loss, avg_loss, tr_nb, tr_num, train_loss = 0.0, 0.0, 0.0, 0, 0, 0
    best_f1 = 0

    model.zero_grad()

    for idx in range(args.epochs):
        bar = tqdm(train_dataloader, total=len(train_dataloader))
        tr_num = 0
        train_loss = 0
        for step, batch in enumerate(bar):
            (inputs_ids_1, position_idx_1, attn_mask_1,
             labels) = [x.to(args.device) for x in batch]
            model.train()
            loss, logits = model(
                inputs_ids_1, position_idx_1, attn_mask_1, labels)

            if args.n_gpu > 1:
                loss = loss.mean()

            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), args.max_grad_norm)

            tr_loss += loss.item()
            tr_num += 1
            train_loss += loss.item()
            if avg_loss == 0:
                avg_loss = tr_loss

            avg_loss = round(train_loss/tr_num, 5)
            bar.set_description("epoch {} loss {}".format(idx, avg_loss))

            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
                global_step += 1
                output_flag = True
                avg_loss = round(
                    np.exp((tr_loss - logging_loss) / (global_step - tr_nb)), 4)

                if global_step % args.save_steps == 0:
                    results = evaluate(args, model, tokenizer,
                                       eval_when_training=True)

                    # Save model checkpoint
                    if results['eval_f1'] > best_f1:
                        best_f1 = results['eval_f1']
                        logger.info("  "+"*"*20)
                        logger.info("  Best f1:%s", round(best_f1, 4))
                        logger.info("  "+"*"*20)

                        checkpoint_prefix = 'checkpoint-best-f1'
                        output_dir = os.path.join(
                            args.output_dir, '{}'.format(checkpoint_prefix))
                        if not os.path.exists(output_dir):
                            os.makedirs(output_dir)
                        model_to_save = model.module if hasattr(
                            model, 'module') else model
                        output_dir = os.path.join(
                            output_dir, '{}'.format('model.bin'))
                        torch.save(model_to_save.state_dict(), output_dir)
                        logger.info(
                            "Saving model checkpoint to %s", output_dir)


def evaluate(args, model, tokenizer, eval_when_training=False):
    # build dataloader
    eval_dataset = TextDataset(tokenizer, args, file_path=args.eval_data_file)
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(
        eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size, num_workers=4)

    # multi-gpu evaluate
    if args.n_gpu > 1 and eval_when_training is False:
        model = torch.nn.DataParallel(model)

    # Eval!
    logger.info("***** Running evaluation *****")
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)

    eval_loss = 0.0
    nb_eval_steps = 0
    model.eval()
    logits = []
    y_trues = []
    for batch in eval_dataloader:
        (inputs_ids_1, position_idx_1, attn_mask_1,
         labels) = [x.to(args.device) for x in batch]
        with torch.no_grad():
            lm_loss, logit = model(
                inputs_ids_1, position_idx_1, attn_mask_1, labels)
            eval_loss += lm_loss.mean().item()
            logits.append(logit.cpu().numpy())
            y_trues.append(labels.cpu().numpy())
        nb_eval_steps += 1

    # calculate scores
    logits = np.concatenate(logits, 0)
    y_trues = np.concatenate(y_trues, 0)
    best_threshold = 0.5
    best_f1 = 0

    y_preds = logits[:, 1] > best_threshold
    from sklearn.metrics import recall_score
    recall = recall_score(y_trues, y_preds, average='macro')
    from sklearn.metrics import precision_score
    precision = precision_score(y_trues, y_preds, average='macro')
    from sklearn.metrics import f1_score
    f1 = f1_score(y_trues, y_preds, average='macro')
    result = {
        "eval_recall": float(recall),
        "eval_precision": float(precision),
        "eval_f1": float(f1),
        "eval_threshold": best_threshold,

    }

    logger.info("***** Eval results *****")
    for key in sorted(result.keys()):
        logger.info("  %s = %s", key, str(round(result[key], 4)))

    return result

def DFG_format(DFG_orig):
    DFG = []
    for i in range(len(DFG_orig)):
            DFG_line = []
            for j in range(len(DFG_orig[i])):
                if j == 0 or j == 2:
                    DFG_line.append(DFG_orig[i][j][0])
                if j == 1:
                    DFG_line.append(DFG_orig[i][j].item())
                if j == 3:
                    if len(DFG_orig[i][j]) == 0:
                        DFG_line.append(DFG_orig[i][j])
                    else:
                        DFG_line.append(list(DFG_orig[i][j][0]))
                if j == 4:
                    if len(DFG_orig[i][j]) == 0:
                        DFG_line.append(DFG_orig[i][j])
                    else:
                        temp = []
                        temp.append(DFG_orig[i][j][0].item())
                        DFG_line.append(temp)        
            DFG.append(tuple(DFG_line))
    return DFG
    
def convert_DFG_to_attn_mask(args,tokenizer,code_tokens,dfg):
    
            cache = []
            code_tokens = [tokenizer.tokenize(
                '@ '+x)[1:] if idx != 0 else tokenizer.tokenize(x) for idx, x in enumerate(code_tokens)]
            ori2cur_pos = {}
            ori2cur_pos[-1] = (0, 0)
            for i in range(len(code_tokens)):
                ori2cur_pos[i] = (ori2cur_pos[i-1][1],
                                  ori2cur_pos[i-1][1]+len(code_tokens[i]))
            code_tokens = [y for x in code_tokens for y in x]

            # truncating
            code_tokens = code_tokens[:args.code_length+args.data_flow_length -
                                      3-min(len(dfg), args.data_flow_length)][:512-3]
            source_tokens = [tokenizer.cls_token] + \
                code_tokens+[tokenizer.sep_token]
            source_ids = tokenizer.convert_tokens_to_ids(source_tokens)
            position_idx = [i+tokenizer.pad_token_id +
                            1 for i in range(len(source_tokens))]
            dfg = dfg[:args.code_length +
                      args.data_flow_length-len(source_tokens)]
            source_tokens += [x[0] for x in dfg]
            position_idx += [0 for x in dfg]
            source_ids += [tokenizer.unk_token_id for x in dfg]
            padding_length = args.code_length + \
                args.data_flow_length-len(source_ids)
            position_idx += [tokenizer.pad_token_id]*padding_length
            source_ids += [tokenizer.pad_token_id]*padding_length

            # reindex
            reverse_index = {}
            for idx, x in enumerate(dfg):
                reverse_index[x[1]] = idx
            for idx, x in enumerate(dfg):
                dfg[idx] = x[:-1]+([reverse_index[i]
                                    for i in x[-1] if i in reverse_index],)
            dfg_to_dfg = [x[-1] for x in dfg]
            dfg_to_code = [ori2cur_pos[x[1]] for x in dfg]
            length = len([tokenizer.cls_token])
            dfg_to_code = [(x[0]+length, x[1]+length) for x in dfg_to_code]
            cache.append(source_tokens)
            cache.append(source_ids)
            cache.append(position_idx)
            cache.append(dfg_to_code)
            cache.append(dfg_to_dfg)
            # calculate graph-guided masked function
            attn_mask_1 = np.zeros((args.code_length+args.data_flow_length,
                                args.code_length+args.data_flow_length), dtype=np.bool)
            # calculate begin index of node and max length of input
            node_index = sum([i > 1 for i in position_idx])
            max_length = sum([i != 1 for i in position_idx])
            # sequence can attend to sequence
            attn_mask_1[:node_index, :node_index] = True
            # special tokens attend to all tokens
            for idx, i in enumerate(source_ids):
                if i in [0, 2]:
                    attn_mask_1[idx, :max_length] = True
                    # nodes attend to code tokens that are identified from
            for idx, (a, b) in enumerate(dfg_to_code):
                if a < node_index and b < node_index:
                    attn_mask_1[idx+node_index, a:b] = True
                    attn_mask_1[a:b, idx+node_index] = True
            # nodes attend to adjacent nodes
            for idx, nodes in enumerate(dfg_to_dfg):
                for a in nodes:
                    if a+node_index < len(position_idx):
                        attn_mask_1[idx+node_index, a+node_index] = True
            # return the convert result
            attn_mask_1_result = []
            attn_mask_1_result.append(attn_mask_1)
            source_id_result = []
            source_id_result.append(source_ids)
            position_idx_result = []
            position_idx_result.append(position_idx)

            return (torch.tensor(source_id_result),
                torch.tensor(position_idx_result),
                torch.tensor(attn_mask_1_result))

def DFG_cut(y_preds,dfg,chain_num_list_train):
    retain_indexes = []
    for i in range(len(y_preds)-2):
        if (y_preds[i+2]) != y_preds[1]:
            retain_indexes.append(i)
    chain_num_list_cut = []
    for index in retain_indexes:
        chain_num_list_cut.append(chain_num_list_train[index])
    dfg_cut = []
    for index in chain_num_list_cut:
        dfg_cut.append(dfg[index])
    return dfg_cut

def get_mask_DFG(mask_vector,dfg_cut):
    mask_dfg = []
    mask_vector = mask_vector.sigmoid()
    for i in range(len(dfg_cut)):
        if mask_vector[0][i] >= 0.5:
            mask_dfg.append(dfg_cut[i])
            
    return mask_dfg
            
coeffs = {
        'edge_size': 0.005,
        'node_feat_size': 1.0,
        'edge_ent': 1.0,
        'node_feat_ent': 0.1,
    }
EPS = 1e-15
def __loss__(args,loss_func, mask_vector, preds, label):
    loss = loss_func(preds, label)
    m = mask_vector.sigmoid().to(args.device)
    loss = loss + coeffs['edge_size'] * m.sum()
    ent = -m * torch.log(m + EPS) - (1 - m) * torch.log(1 - m + EPS)
    loss = loss + coeffs['edge_ent'] * ent.mean()
    return loss


def construct_tokens_to_line(func):
    code = func
    tree = parser[0].parse(bytes(code, 'utf8'))
    root_node = tree.root_node
    tokens_index = tree_to_token_index(root_node)
    token_num_to_line = {}
    for i in range(len(tokens_index)):
        token_num_to_line[i] = tokens_index[i][0][0]
    return token_num_to_line


def sort_count(_list):
    list2 = []
    result = []
    for i in set(_list):
        list2.append([_list.count(i),i])

    list2.sort(reverse=True)
    for count,num in list2:
        for i in range(count):
            result.append(num)
    return result
    
def test(args, model, tokenizer, best_threshold=0):
    # build dataloader
    eval_dataset = TextDataset(tokenizer, args, file_path=args.test_data_file)
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=1, num_workers=4)
    # Eval!
    logger.info("***** Running Test *****")
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    # multi-gpu evaluate
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)
    model.eval()
    f = open(os.path.join(args.output_dir, "predictions.txt"), 'w')
    f2 = open(os.path.join(args.output_dir, "programs.txt"), 'w')
    for item_eval_dataset in eval_dataset:
        chain_num_list_train = []
        eval_loss = 0.0
        nb_eval_steps = 0
        logits = []
        y_trues = []
        
        (inputs_ids_1, position_idx_1, attn_mask_1,labels,chain_num_list,code_tokens_orig,DFG_orig,func,url) = [x for x in item_eval_dataset]
            
        code_tokens = code_tokens_orig
       
        f4 = open('temp/'+str(url)+'_min_DFG_all.jsonl','r')
        for line in tqdm(f4,desc="processing-"+str(url)+":"):
                js = json.loads(line)
                dfg = []
                for i in range(JSONLength(js)):
                    dfg.append(tuple(js[str(i)]))
                (inputs_ids, position_idx, attn_mask) = convert_DFG_to_attn_mask(args,tokenizer,code_tokens,dfg)
                inputs_ids = inputs_ids.to(args.device)
                position_idx = position_idx.to(args.device)
                attn_mask = attn_mask.to(args.device)
                labels = labels.to(args.device)
                labels = torch.tensor([1], device='cuda:0')
                with torch.no_grad():
                    lm_loss, logit = model(inputs_ids, position_idx, attn_mask, labels)
                    eval_loss += lm_loss.mean().item()
                    logits.append(logit[:,1][0].cpu().numpy())
                    y_trues.append(labels.cpu().numpy())
                nb_eval_steps += 1     
        f4.close()
        
        """ perform the Intelligence Assistant process
        """
        original_url = url #original url
        original_code = func  #original code with '\n' in In tuple form
        y_preds = logits    #y_preds[0] is the DFG_orig without propogation chain,y_preds[1] is the DFG_orig with propagationchain
        chain_num_list_train = chain_num_list #the number of each row in the original of DFG
        
        dfg = DFG_orig #original DFG
        dfg_cut = DFG_cut(y_preds,dfg,chain_num_list_train)#cut the edges which have no relationship with masked prediction results
        
        mask_vector = []
        
        node_feat_vector = torch.nn.Parameter(torch.randn((1,len(dfg_cut)), requires_grad = True))
#        node_feat_vector=node_feat_vector.to(args.device)
        mask_vector = torch.nn.Parameter(torch.randn((1,len(dfg_cut)), requires_grad = True))
#        mask_vector=mask_vector.to(args.device)
        
        optimizer = torch.optim.Adam([mask_vector], lr = 0.1)
#        loss_func=torch.nn.MSELoss()
        loss_func=cross_entropy
#        print (node_feat_vector)
        record = mask_vector.sigmoid()
        epochs = 100
        for epoch in range(1, epochs+1):
            mask_dfg = get_mask_DFG(mask_vector,dfg_cut)
            (inputs_ids, position_idx, attn_mask) = convert_DFG_to_attn_mask(args,tokenizer,code_tokens,mask_dfg)
            inputs_ids = inputs_ids.to(args.device)
            position_idx = position_idx.to(args.device)
            attn_mask = attn_mask.to(args.device)
            label = labels
            lm_loss, logit = model(inputs_ids, position_idx, attn_mask, label)
            
            preds = torch.stack((logit.reshape(2,1)[1],logit.reshape(2,1)[1]))
            preds = preds.reshape(1,2)

            loss = __loss__(args, loss_func, mask_vector, preds, label)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            logger.info("epoch {} loss {}".format(epoch, loss))
        
        result_dfg = get_mask_DFG(mask_vector,dfg_cut) 
        token_num_to_line = construct_tokens_to_line(original_code)
                
        
        fl_result = []
        for chain_line in result_dfg:
            fl_result.append(token_num_to_line[chain_line[1]])
            fl_result = sort_count(fl_result)
        final_result = []
        for item in fl_result:
            if item not in final_result:
                final_result.append(item)
        for chain_line in dfg_cut:
            if token_num_to_line[chain_line[1]] not in final_result:
                final_result.append(token_num_to_line[chain_line[1]])
        
        f.write(original_url+'\t')
        for item in final_result:
            f.write('['+str(item)+']'+' ')
        f.write('\n')
        
        f2.write('----------------------------------'+str(original_url)+'---------------------------------------\n')
        codewrite = original_code.split('\n')
        for i in range(len(codewrite)):
            f2.write(str(i)+':\t')
            f2.write(codewrite[i]+'\n')
        
    f.close()
    f2.close()
    
def main():

    args = RuntimeContext()

    # Setup logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)
    logger.warning("device: %s, n_gpu: %s", args.device, args.n_gpu,)
    # Set seed
    set_seed(args)
    config = RobertaConfig.from_pretrained(
        args.config_name if args.config_name else args.model_name_or_path)
    config.num_labels = 1
    tokenizer = RobertaTokenizer.from_pretrained(args.tokenizer_name)
    model = RobertaForSequenceClassification.from_pretrained(
        args.model_name_or_path, config=config)
    model = Model(model, config, tokenizer, args)

    checkpoint_prefix = 'checkpoint-best-f1/model.bin'
    output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))
    model.load_state_dict(torch.load(output_dir))
    model.to(args.device)
    test(args, model, tokenizer, best_threshold=0.5)


if __name__ == "__main__":
    main()
