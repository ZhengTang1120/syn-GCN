"""
A rnn model for relation extraction, written in pytorch.
"""
import math
import numpy as np
import torch
from torch import nn
from torch.nn import init
import torch.nn.functional as F

from utils import constant, torch_utils

from torch_geometric.nn import GCNConv, RGCNConv, GATConv

class RelationModel(object):
    """ A wrapper class for the training and evaluation of models. """
    def __init__(self, opt, emb_matrix=None):
        self.opt = opt
        self.model = SynGCN(opt, emb_matrix)
        self.criterion = nn.CrossEntropyLoss()
        self.parameters = [p for p in self.model.parameters() if p.requires_grad]
        if opt['cuda']:
            self.model.cuda()
            self.criterion.cuda()
        self.optimizer = torch_utils.get_optimizer(opt['optim'], self.parameters, opt['lr'])
    
    def update(self, batch):
        """ Run a step of forward and backward model update. """
        inputs = list()
        if self.opt['cuda']:
            for b in constant.KEYS:
                inputs += [batch[b].cuda()]
            labels = batch.rel.cuda()
        else:
            for b in constant.KEYS:
                inputs += [batch[b]]
            labels = batch.rel
        batch_size = labels.size(0)
        # step forward
        self.model.train()
        self.optimizer.zero_grad()
        logits, pooling_output = self.model(inputs, batch_size)
        loss = self.criterion(logits, labels)
        if self.opt.get('conv_l2', 0) > 0:
            loss += self.model.conv_l2() * self.opt['conv_l2']
        if self.opt.get('pooling_l2', 0) > 0:
            loss += self.opt['pooling_l2'] * (pooling_output ** 2).sum(1).mean()
        # backward
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.opt['max_grad_norm'])
        self.optimizer.step()
        loss_val = loss.data.item()
        return loss_val

    def predict(self, batch):
        """ Run forward prediction. If unsort is True, recover the original order of the batch. """
        inputs = list()
        if self.opt['cuda']:
            for b in constant.KEYS:
                inputs += [batch[b].cuda()]
            labels = batch.rel.cuda()
        else:
            for b in constant.KEYS:
                inputs += [batch[b]]
            labels = batch.rel

        batch_size = labels.size(0)

        # forward
        self.model.eval()
        logits, _ = self.model(inputs, batch_size)
        loss = self.criterion(logits, labels)
        probs = F.softmax(logits, dim=1).data.cpu().numpy().tolist()
        predictions = np.argmax(logits.data.cpu().numpy(), axis=1).tolist()

        return predictions, probs, loss.data.item()

    def update_lr(self, new_lr):
        torch_utils.change_lr(self.optimizer, new_lr)

    def save(self, filename, epoch):
        params = {
                'model': self.model.state_dict(),
                'config': self.opt,
                'epoch': epoch
                }
        try:
            torch.save(params, filename)
            print("model saved to {}".format(filename))
        except BaseException:
            print("[Warning: Saving failed... continuing anyway.]")

    def load(self, filename):
        try:
            checkpoint = torch.load(filename)
        except BaseException:
            print("Cannot load model from {}".format(filename))
            exit()
        self.model.load_state_dict(checkpoint['model'])
        self.opt = checkpoint['config']

class SynGCN(nn.Module):
    """ A sequence model for relation extraction. """

    def __init__(self, opt, emb_matrix=None):
        super(SynGCN, self).__init__()
        self.drop = nn.Dropout(opt['dropout'])
        self.emb = nn.Embedding(opt['vocab_size'], opt['emb_dim'], padding_idx=constant.PAD_ID)
        if opt['pos_dim'] > 0:
            self.pos_emb = nn.Embedding(len(constant.POS_TO_ID), opt['pos_dim'],
                    padding_idx=constant.PAD_ID)
        if opt['ner_dim'] > 0:
            self.ner_emb = nn.Embedding(len(constant.NER_TO_ID), opt['ner_dim'],
                    padding_idx=constant.PAD_ID)

        input_size = opt['emb_dim'] + opt['pos_dim'] + opt['ner_dim']
        if opt['trans']:
            encoder_layer = nn.TransformerEncoderLayer(d_model=input_size, nhead=8)
            self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)
            self.tlinear = nn.Linear(input_size, opt['hidden_dim'])
        else:
            self.rnn = nn.LSTM(input_size, opt['hidden_dim'], opt['num_layers'], batch_first=True,\
                dropout=opt['dropout'], bidirectional=True)

        if opt['sgcn']:
            self.deprel_emb = nn.Embedding(len(constant.DEPREL_TO_ID), opt['deprel_dim'],
                    padding_idx=constant.PAD_ID)
            self.attn = Attention(opt['deprel_dim'], 2*opt['hidden_dim'])
            self.sgcn2 = GCNConv(2*opt['hidden_dim'], opt['hidden_dim'])
        if opt['rgcn']:
            self.rgcn = RGCNConv(2*opt['hidden_dim'], opt['hidden_dim'], len(constant.DEPREL_TO_ID)-1, num_bases=len(constant.DEPREL_TO_ID)-1)
        if opt['gcn']:
            self.gcn = GCNConv(2*opt['hidden_dim'], opt['hidden_dim'])
        if opt['gat']:
            self.deprel_emb = nn.Embedding(len(constant.DEPREL_TO_ID), opt['deprel_dim'],
                    padding_idx=constant.PAD_ID)
            self.gat = GATConv((2*opt['hidden_dim'], 2*opt['hidden_dim']+opt['deprel_dim']), opt['hidden_dim'])

        # output mlp layers
        in_dim = opt['hidden_dim']*3
        layers = [nn.Linear(in_dim, opt['hidden_dim']), nn.ReLU()]
        for _ in range(opt['mlp_layers']-1):
            layers += [nn.Linear(opt['hidden_dim'], opt['hidden_dim']), nn.ReLU()]
        self.out_mlp = nn.Sequential(*layers)
        self.linear = nn.Linear(opt['hidden_dim'], opt['num_class'])

        self.opt = opt
        self.topn = self.opt.get('topn', 1e10)
        self.use_cuda = opt['cuda']
        self.emb_matrix = emb_matrix
        self.init_weights()
    
    def conv_l2(self):
        # conv_weights = []
        conv_weights = [self.sgcn.weight, self.sgcn.bias]
        conv_weights += [self.sgcn2.weight, self.sgcn2.bias]
        return sum([x.pow(2).sum() for x in conv_weights])

    def init_weights(self):
        if self.emb_matrix is None:
            self.emb.weight.data[1:,:].uniform_(-1.0, 1.0) # keep padding dimension to be 0
        else:
            self.emb_matrix = torch.from_numpy(self.emb_matrix)
            self.emb.weight.data.copy_(self.emb_matrix)
        if self.opt['pos_dim'] > 0:
            self.pos_emb.weight.data[1:,:].uniform_(-1.0, 1.0)
        if self.opt['ner_dim'] > 0:
            self.ner_emb.weight.data[1:,:].uniform_(-1.0, 1.0)
        if (self.opt['sgcn'] or self.opt['gat']) and self.opt['deprel_dim'] > 0:
            self.deprel_emb.weight.data[1:,:].uniform_(-1.0, 1.0)
        if self.opt['pattn']:
            self.pe_emb.weight.data.uniform_(-1.0, 1.0)

        self.linear.bias.data.fill_(0)
        init.xavier_uniform_(self.linear.weight, gain=1) # initialize linear layer

        # decide finetuning
        if self.topn <= 0:
            print("Do not finetune word embedding layer.")
            self.emb.weight.requires_grad = False
        elif self.topn < self.opt['vocab_size']:
            print("Finetune top {} word embeddings.".format(self.topn))
            self.emb.weight.register_hook(lambda x: \
                    torch_utils.keep_partial_grad(x, self.topn))
        else:
            print("Finetune all embeddings.")

    def zero_state(self, batch_size): 
        state_shape = (2*self.opt['num_layers'], batch_size, self.opt['hidden_dim'])
        h0 = c0 = torch.zeros(*state_shape, requires_grad=False)
        if self.use_cuda:
            return h0.cuda(), c0.cuda()
        else:
            return h0, c0
    
    def forward(self, inputs, batch_size):
        for i in range(len(inputs)-1):
            inputs[i] = inputs[i].view(batch_size, -1)
        words, masks, e_masks, pos, ner, deprel, d_masks, subj_mask, obj_mask, edge_index = inputs # unpack
        s_len = words.size(1)
        seq_lens = list(masks.data.eq(constant.PAD_ID).long().sum(1).squeeze())
        # embedding lookup
        word_inputs = self.emb(words)
        inputs = [word_inputs]
        if self.opt['pos_dim'] > 0:
            inputs += [self.pos_emb(pos)]
        if self.opt['ner_dim'] > 0:
            inputs += [self.ner_emb(ner)]
        inputs = self.drop(torch.cat(inputs, dim=2)) # add dropout to input
        input_size = inputs.size(2)
        if self.opt['trans']:
            # transformer
            inputs  = inputs.transpose(1,0)
            outputs = self.transformer_encoder(inputs, src_key_padding_mask=masks)
            outputs = self.tlinear(outputs)
        else:
            # rnn
            h0, c0 = self.zero_state(batch_size)
            inputs = nn.utils.rnn.pack_padded_sequence(inputs, seq_lens, batch_first=True)
            outputs, (ht, ct) = self.rnn(inputs, (h0, c0))
            outputs, output_lens = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)
        # hidden = self.drop(ht[-1,:,:]) # get the outmost layer h_n
        outputs = self.drop(outputs)

        if self.opt['sgcn']:
            deprel = self.deprel_emb(deprel)

            pool_type = self.opt['pooling']
            
            h_out    = pool(outputs, e_masks.unsqueeze(2), type=pool_type)
            weights = self.attn(deprel, d_masks, h_out).view(-1)
            weights = weights[weights.nonzero()].squeeze(1)
            outputs = outputs.reshape(s_len*batch_size, -1)
            outputs = self.sgcn2(outputs, edge_index, weights)
            outputs = outputs.reshape(batch_size, s_len, -1)

            h_out    = pool(outputs, e_masks.unsqueeze(2), type=pool_type)
            subj_out = pool(outputs, subj_mask.unsqueeze(2), type=pool_type)
            obj_out  = pool(outputs, obj_mask.unsqueeze(2), type=pool_type)

            final_hidden = self.drop(torch.cat([h_out, subj_out, obj_out] , dim=1))


        elif self.opt['gat']:
            deprel    = self.deprel_emb(deprel)
            outputs_t = torch.cat([outputs, deprel], dim=2)
            outputs   = outputs.reshape(s_len*batch_size, -1)
            outputs_t = outputs_t.reshape(s_len*batch_size, -1)
            outputs   = self.gat((outputs, outputs_t), edge_index)
            outputs   = outputs.reshape(batch_size, s_len, -1)

            h_out    = pool(outputs, e_masks.unsqueeze(2), type=pool_type)
            subj_out = pool(outputs, subj_mask.unsqueeze(2), type=pool_type)
            obj_out  = pool(outputs, obj_mask.unsqueeze(2), type=pool_type)

            final_hidden = self.drop(torch.cat([h_out, subj_out, obj_out] , dim=1))

        elif self.opt['gcn']:
            pool_type = self.opt['pooling']
            
            outputs = outputs.reshape(s_len*batch_size, -1)
            outputs = self.gcn(outputs, edge_index)
            outputs = outputs.reshape(batch_size, s_len, -1)

            h_out    = pool(outputs, e_masks.unsqueeze(2), type=pool_type)
            subj_out = pool(outputs, subj_mask.unsqueeze(2), type=pool_type)
            obj_out  = pool(outputs, obj_mask.unsqueeze(2), type=pool_type)

            final_hidden = self.drop(torch.cat([h_out, subj_out, obj_out] , dim=1))
        
        elif self.opt['rgcn']:
            deprel  = deprel.reshape(-1)
            deprel  = (deprel[deprel.nonzero()] - 1).reshape(-1)

            pool_type = self.opt['pooling']
            
            h_out    = pool(outputs, e_masks.unsqueeze(2), type=pool_type)
            weights = self.attn(deprel, d_masks, h_out).view(-1)
            weights = weights[weights.nonzero()].squeeze(1)
            outputs = outputs.reshape(s_len*batch_size, -1)
            outputs = self.rgcn(outputs, edge_index, deprel)
            outputs = outputs.reshape(batch_size, s_len, -1)

            h_out    = pool(outputs, e_masks.unsqueeze(2), type=pool_type)
            subj_out = pool(outputs, subj_mask.unsqueeze(2), type=pool_type)
            obj_out  = pool(outputs, obj_mask.unsqueeze(2), type=pool_type)

            final_hidden = self.drop(torch.cat([h_out, subj_out, obj_out] , dim=1))

        final_hidden = self.out_mlp(final_hidden)
        logits = self.linear(final_hidden)
        return logits, h_out

class Attention(nn.Module):
    """
    A GCN layer with attention on deprel as edge weights.
    """
    
    def __init__(self, input_size, query_size):
        super(Attention, self).__init__()
        self.input_size = input_size
        self.query_size = query_size
        # self.ulinear = nn.Linear(input_size, attn_size)
        # self.vlinear = nn.Linear(query_size, attn_size, bias=False)
        # self.tlinear = nn.Linear(attn_size, 1)
        self.weight = nn.Parameter(torch.Tensor(input_size, query_size))
        self.init_weights()

    def init_weights(self):
        # self.ulinear.weight.data.normal_(std=0.001)
        # self.vlinear.weight.data.normal_(std=0.001)
        # self.tlinear.weight.data.zero_() # use zero to give uniform attention at the beginning
        self.weight.data.normal_(std=0.001)

    def forward(self, x, x_mask, q):
        """
        x : batch_size * seq_len * input_size
        q : batch_size * query_size
        f : batch_size * seq_len * feature_size
        """
        batch_size, seq_len, _ = x.size()

        # x_proj = self.ulinear(x.contiguous().view(-1, self.input_size)).view(
        #     batch_size, seq_len, self.attn_size)
        # q_proj = self.vlinear(q.view(-1, self.query_size)).contiguous().view(
        #     batch_size, self.attn_size).unsqueeze(1).expand(
        #         batch_size, seq_len, self.attn_size)
        # projs = [x_proj, q_proj]
        # scores = self.tlinear(torch.tanh(sum(projs)).view(-1, self.attn_size)).view(
        #     batch_size, seq_len)

        x_proj = torch.matmul(x, self.weight)
        scores = torch.bmm(x_proj, q.view(batch_size, self.query_size, 1)).view(batch_size, seq_len)

        # mask padding
        scores.data.masked_fill_(x_mask.data, -float('inf'))
        weights = F.softmax(scores, dim=1)

        return weights

class PositionAwareAttention(nn.Module):
    """
    A position-augmented attention layer where the attention weight is
    a = T' . tanh(Ux + Vq + Wf)
    where x is the input, q is the query, and f is additional position features.
    """

    def __init__(self, input_size, query_size, feature_size, attn_size):
        super(PositionAwareAttention, self).__init__()
        self.input_size = input_size
        self.query_size = query_size
        self.feature_size = feature_size
        self.attn_size = attn_size
        self.ulinear = nn.Linear(input_size, attn_size)
        self.vlinear = nn.Linear(query_size, attn_size, bias=False)
        if feature_size > 0:
            self.wlinear = nn.Linear(feature_size, attn_size, bias=False)
        else:
            self.wlinear = None
        self.tlinear = nn.Linear(attn_size, 1)
        self.init_weights()

    def init_weights(self):
        self.ulinear.weight.data.normal_(std=0.001)
        self.vlinear.weight.data.normal_(std=0.001)
        if self.wlinear is not None:
            self.wlinear.weight.data.normal_(std=0.001)
        self.tlinear.weight.data.zero_() # use zero to give uniform attention at the beginning

    def forward(self, x, x_mask, q, f):
        """
        x : batch_size * seq_len * input_size
        q : batch_size * query_size
        f : batch_size * seq_len * feature_size
        """
        batch_size, seq_len, _ = x.size()

        x_proj = self.ulinear(x.contiguous().view(-1, self.input_size)).view(
            batch_size, seq_len, self.attn_size)
        q_proj = self.vlinear(q.view(-1, self.query_size)).contiguous().view(
            batch_size, self.attn_size).unsqueeze(1).expand(
                batch_size, seq_len, self.attn_size)
        if self.wlinear is not None:
            f_proj = self.wlinear(f.view(-1, self.feature_size)).contiguous().view(
                batch_size, seq_len, self.attn_size)
            projs = [x_proj, q_proj, f_proj]
        else:
            projs = [x_proj, q_proj]
        scores = self.tlinear(torch.tanh(sum(projs)).view(-1, self.attn_size)).view(
            batch_size, seq_len)

        # mask padding
        scores.data.masked_fill_(x_mask.data, -float('inf'))
        weights = F.softmax(scores, dim=1)
        # weighted average input vectors
        outputs = weights.unsqueeze(1).bmm(x).squeeze(1)
        return outputs

def pool(h, mask, type='max'):
    if type == 'max':
        h = h.masked_fill(mask, -constant.INFINITY_NUMBER)
        return torch.max(h, 1)[0]
    elif type == 'avg':
        h = h.masked_fill(mask, 0)
        return h.sum(1) / (mask.size(1) - mask.float().sum(1))
    else:
        h = h.masked_fill(mask, 0)
        return h.sum(1)