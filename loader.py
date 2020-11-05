"""
Data loader for TACRED json files.
"""

import json
import random
import torch
import numpy as np

from utils import constant, helper, vocab

from torch_geometric.data import Data, DataLoader

from collections import defaultdict

class BatchLoader(object):
    """
    Load data from json files, preprocess and prepare batches.
    """
    def __init__(self, filename, batch_size, opt, vocab, mappings, evaluation=False):
        self.batch_size = batch_size
        self.vocab = vocab
        self.eval = evaluation
        self.opt = opt
        self.mappings = mappings

        with open(filename) as infile:
            data = json.load(infile)
        data, data_r = self.preprocess(data, vocab, opt)
        data = sorted(data, key=lambda d: len(d[0]), reverse=True)
        data_r = sorted(data_r, key=lambda d: len(d[0]), reverse=True)
        
        id2label = dict([(v,k) for k,v in constant.LABEL_TO_ID.items()])
        self.labels = [id2label[d[6]] for d in data_r]
        self.num_examples = len(data_r)

        datalist = self.chuck_batch(data, False)
        datalist_r = self.chuck_batch(data_r, True)

        self.data = DataLoader(datalist, batch_size=batch_size)
        self.data_r = DataLoader(datalist_r, batch_size=batch_size)

        print("{} batches created for {}".format(len(data), filename))

    def chuck_batch(self, data, rule):
        # chunk into batches
        data     = [data[i:i+self.batch_size] for i in range(0, len(data), self.batch_size)]
        self.raw = data
        datalist = list()
        for i, batch in enumerate(data):
            batch = list(zip(*batch))
            if rule:
                assert len(batch) == 10
            else:
                assert len(batch) == 9
            
            # word dropout
            if not self.eval:
                words = [word_dropout(sent,  self.opt['word_dropout']) for sent in batch[0]]
            else:
                words = batch[0]

            # convert to tensors
            words = get_long_tensor(words)
            pos = get_long_tensor(batch[1])
            ner = get_long_tensor(batch[2])
            deprel = get_long_tensor(batch[3])
            subj_masks = get_long_tensor(batch[4])
            obj_masks = get_long_tensor(batch[5])
            e_masks = get_long_tensor(batch[8])
            if rule:
                rules = get_long_tensor(batch[9])
                for i in range(len(words)):
                    datalist += [Data(words=words[i], mask=torch.eq(words[i], 0), e_mask = torch.eq(e_masks[i], 0), pos=pos[i], 
                        ner=ner[i], deprel=deprel[i], d_mask=torch.eq(deprel[i], 0), 
                        subj_mask=torch.eq(subj_masks[i], 0), obj_mask=torch.eq(obj_masks[i], 0), 
                        edge_index=torch.LongTensor(batch[7][i]),
                        rel=torch.LongTensor([batch[6][i]]), rule=rules[i])]
            else:
                for i in range(len(words)):
                    datalist += [Data(words=words[i], mask=torch.eq(words[i], 0), e_mask = torch.eq(e_masks[i], 0), pos=pos[i], 
                        ner=ner[i], deprel=deprel[i], d_mask=torch.eq(deprel[i], 0), 
                        subj_mask=torch.eq(subj_masks[i], 0), obj_mask=torch.eq(obj_masks[i], 0), 
                        edge_index=torch.LongTensor(batch[7][i]),
                        rel=torch.LongTensor([batch[6][i]]))]

        return datalist

    def preprocess(self, data, vocab, opt):
        """ Preprocess the data and convert to ids. """
        processed      = []
        processed_rule = []
        rule_counts = defaultdict(int)
        with open(self.mappings) as f:
            mappings = f.readlines()
        with open('tacred/rules.json') as f:
            rules = json.load(f)
        for c, d in enumerate(data):
            tokens = d['token']
            if opt['lower']:
                tokens = [t.lower() for t in tokens]
            l = len(tokens)
            # anonymize tokens
            ss, se = d['subj_start'], d['subj_end']
            os, oe = d['obj_start'], d['obj_end']
            tokens[ss:se+1] = ['SUBJ-'+d['subj_type']] * (se-ss+1)
            tokens[os:oe+1] = ['OBJ-'+d['obj_type']] * (oe-os+1)
            tokens = map_to_ids(tokens, vocab.word2id)
            pos = map_to_ids(d['stanford_pos'], constant.POS_TO_ID)
            ner = map_to_ids(d['stanford_ner'], constant.NER_TO_ID)
            if self.opt['gat']:
                deprel = map_to_ids(d['stanford_deprel'], constant.DEPREL_TO_ID)
            else:
                deprel = map_to_ids([d for d in d['stanford_deprel'] if d!='ROOT' and d!='root'], constant.DEPREL_TO_ID)
            
            if opt['prune_k'] < 0:
                edge_index = [[h-1 for h in d['stanford_head'] if h != 0], 
                [i for i, h in enumerate(d['stanford_head']) if h != 0]]
            else:
                edge_index = prune_tree(l, d['stanford_head'], opt['prune_k'], list(range(ss, se+1)), list(range(os, oe+1)))
                deprel = map_to_ids([d['stanford_deprel'][i] for i in edge_index[1]], constant.DEPREL_TO_ID)
                if deprel[-1] == 2:
                    deprel = deprel[:-1]
                    edge_index = [edge_index[0][:-1], edge_index[1][:-1]]
                edge_index = [edge_index[0]+edge_index[1], edge_index[1]+edge_index[0]]
            edge_mask = [1 if i in edge_index[1] else 0 for i in range(l)]
            relation = constant.LABEL_TO_ID[d['relation']]

            if opt['pattn']:
                subj_positions = get_positions(d['subj_start'], d['subj_end'], l)
                obj_positions = get_positions(d['obj_start'], d['obj_end'], l)
                if 't_' in mappings[c] or 's_' in mappings[c]:
                    rule = helper.word_tokenize(rules[eval(mappings[c])[0][1]])
                    rule = map_to_ids(rule, vocab.rule2id) 
                    rule = [constant.SOS_ID] + rule + [constant.EOS_ID]
                    processed_rule += [(tokens, pos, ner, deprel, subj_positions, obj_positions, relation, edge_index, rule)]
                else:
                    processed += [(tokens, pos, ner, deprel, subj_positions, obj_positions, relation, edge_index)]
            else:
                subj_mask = [1 if (i in range(ss, se+1) and i in edge_index[0]+edge_index[1]) else 0 for i in range(len(tokens))]
                obj_mask = [1 if (i in range(os, oe+1) and i in edge_index[0]+edge_index[1]) else 0 for i in range(len(tokens))]
                
                if 't_' in mappings[c] or 's_' in mappings[c]:
                    rule_counts[rules[eval(mappings[c])[0][1]]] += 1
                    rule = helper.word_tokenize(rules[eval(mappings[c])[0][1]])
                    rule = map_to_ids(rule, vocab.rule2id) 
                    rule = [constant.SOS_ID] + rule + [constant.EOS_ID]
                    processed_rule += [(tokens, pos, ner, deprel, subj_mask, obj_mask, relation, edge_index, edge_mask, rule)]
                else:
                    processed += [(tokens, pos, ner, deprel, subj_mask, obj_mask, relation, edge_index, edge_mask)]
        print (json.dumps(rule_counts))
        exit()
        return processed, processed_rule

    def gold(self):
        """ Return gold labels as a list. """
        return self.labels

    def __len__(self):
        #return 50
        return len(self.data)+len(self.data_r)

def get_positions(start_idx, end_idx, length):
    """ Get subj/obj position sequence. """
    return list(range(-start_idx, 0)) + [0]*(end_idx - start_idx + 1) + \
            list(range(1, length-end_idx))
            
def map_to_ids(tokens, vocab):
    ids = [vocab[t] if t in vocab else constant.UNK_ID for t in tokens]
    return ids

def get_long_tensor(tokens_list):
    """ Convert list of list of tokens to a padded LongTensor. """
    token_len = max(len(x) for x in tokens_list)
    tokens = list()
    for i, s in enumerate(tokens_list):
        token = torch.LongTensor(token_len).fill_(constant.PAD_ID)
        token[:len(s)] = torch.LongTensor(s)
        tokens += [token]
    return tokens

def word_dropout(tokens, dropout):
    """ Randomly dropout tokens (IDs) and replace them with <UNK> tokens. """
    return [constant.UNK_ID if x != constant.UNK_ID and np.random.random() < dropout \
            else x for x in tokens]

def prune_tree(len_, head, prune, subj_pos, obj_pos):
    cas = None
    subj_ancestors = set()
    for s in subj_pos:
        if head[s]-1 not in subj_pos:
            subj_ancestors.add(s)
    subj_pos = list(subj_ancestors)
    for s in subj_pos:
        h = head[s]
        tmp = [s]
        while h > 0:
            tmp += [h-1]
            subj_ancestors.add(h-1)
            h = head[h-1]

        if cas is None:
            cas = set(tmp)
        else:
            cas.intersection_update(tmp)
    obj_ancestors = set()
    for o in obj_pos:
        if head[o]-1 not in obj_pos:
            obj_ancestors.add(o)
    obj_pos = list(obj_ancestors)
    for o in obj_pos:
        h = head[o]
        tmp = [o]
        while h > 0:
            tmp += [h-1]
            obj_ancestors.add(h-1)
            h = head[h-1]
        cas.intersection_update(tmp)

    # find lowest common ancestor
    if len(cas) == 1:
        lca = list(cas)[0]
    else:
        child_count = {k:0 for k in cas}
        for ca in cas:
            if head[ca] > 0 and head[ca] - 1 in cas:
                child_count[head[ca] - 1] += 1

        # the LCA has no child in the CA set
        for ca in cas:
            if child_count[ca] == 0:
                lca = ca
                break

    path_nodes = subj_ancestors.union(obj_ancestors).difference(cas)
    path_nodes.add(lca)
    # compute distance to path_nodes
    dist = [-1 if i not in path_nodes else 0 for i in range(len_)]

    for i in range(len_):
        if dist[i] < 0:
            stack = [i]
            while stack[-1] >= 0 and stack[-1] not in path_nodes:
                stack.append(head[stack[-1]] - 1)

            if stack[-1] in path_nodes:
                for d, j in enumerate(reversed(stack)):
                    dist[j] = d
            else:
                for j in stack:
                    if j >= 0 and dist[j] < 0:
                        dist[j] = int(1e4) # aka infinity
    edge_index = [[],[]]
    for i in range(len_):
        if dist[i] <= prune:
            h = head[i]
            if h >= 0 and i != lca:
                edge_index[0].append(h-1)
                edge_index[1].append(i)
    return edge_index
