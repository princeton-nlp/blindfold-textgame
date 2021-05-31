import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from os.path import join as pjoin
from memory import * 
from model import DRRN
from util import *
import logger
from transformers import BertTokenizer
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DRRN_Agent:
    def __init__(self, args):
        self.gamma = args.gamma
        self.batch_size = args.batch_size
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.network = DRRN(len(self.tokenizer), args.embedding_dim, args.hidden_dim, args.fix_rep, args.hash_rep, args.act_obs).to(device)
        self.network.tokenizer = self.tokenizer
        self.memory = ABReplayMemory(args.memory_size, args.memory_alpha) 
        self.save_path = args.output_dir
        self.clip = args.clip
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=args.learning_rate)

        self.type_inv = args.type_inv
        self.type_for = args.type_for
        self.w_inv = args.w_inv
        self.w_for = args.w_for
        self.w_act = args.w_act
        self.perturb = args.perturb
        
        self.act_obs = args.act_obs

    def observe(self, transition, is_prior=False):
        self.memory.push(transition, is_prior)

    
    def build_state(self, ob, info):
        """ Returns a state representation built from various info sources. """
        if self.act_obs:
            acts = self.encode(info['valid'])
            obs_ids, look_ids, inv_ids = [], [], []
            for act in acts: obs_ids += act
            return State(obs_ids, look_ids, inv_ids) 
        obs_ids = self.tokenizer.encode(ob)
        look_ids = self.tokenizer.encode(info['look'])
        inv_ids = self.tokenizer.encode(info['inv'])
        return State(obs_ids, look_ids, inv_ids) 
    

    def build_states(self, obs, infos):
        return [self.build_state(ob, info) for ob, info in zip(obs, infos)]


    def encode(self, obs_list):
        """ Encode a list of observations """
        return [self.tokenizer.encode(o) for o in obs_list]


    def act(self, states, poss_acts, sample=True, eps=0.1):
        """ Returns a string action from poss_acts. """
        idxs, values = self.network.act(states, poss_acts, sample, eps=eps)
        act_ids = [poss_acts[batch][idx] for batch, idx in enumerate(idxs)]
        return act_ids, idxs, values
    
    
    def q_loss(self, transitions, need_qvals=False):
        batch = Transition(*zip(*transitions))

        # Compute Q(s', a') for all a'
        # TODO: Use a target network???
        next_qvals = self.network(batch.next_state, batch.next_acts)
        # Take the max over next q-values
        next_qvals = torch.tensor([vals.max() for vals in next_qvals], device=device)
        # Zero all the next_qvals that are done
        next_qvals = next_qvals * (1-torch.tensor(batch.done, dtype=torch.float, device=device))
        targets = torch.tensor(batch.reward, dtype=torch.float, device=device) + self.gamma * next_qvals

        # Next compute Q(s, a)
        # Nest each action in a list - so that it becomes the only admissible cmd
        nested_acts = tuple([[a] for a in batch.act])
        qvals = self.network(batch.state, nested_acts)
        # Combine the qvals: Maybe just do a greedy max for generality
        qvals = torch.cat(qvals)
        loss = F.smooth_l1_loss(qvals, targets.detach()) 

        return (loss, qvals) if need_qvals else loss 

    def update(self):
        if len(self.memory) < self.batch_size:
            return None 

        transitions = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))
        nested_acts = tuple([[a] for a in batch.act])
        terms, loss = {}, 0

        # Compute Q learning Huber loss
        terms['Loss_q'], qvals = self.q_loss(transitions, need_qvals=True)
        loss += terms['Loss_q']
        
        # Compute Inverse dynamics loss
        if self.w_inv > 0:
            if self.type_inv == 'decode':
                terms['Loss_id'], terms['Acc_id'] = self.network.inv_loss_decode(batch.state, batch.next_state, nested_acts, hat=True) 
            elif self.type_inv == 'ce':
                terms['Loss_id'], terms['Acc_id'] = self.network.inv_loss_ce(batch.state, batch.next_state, nested_acts, batch.acts)
            else:
                raise NotImplementedError
            loss += self.w_inv * terms['Loss_id']
        
        # Compute Act reconstruction loss
        if self.w_act > 0:
            terms['Loss_act'], terms['Acc_act'] = self.network.inv_loss_decode(batch.state, batch.next_state, nested_acts, hat=False) 
            loss += self.w_act * terms['Loss_act']

        # Compute Forward dynamics loss
        if self.w_for > 0:
            if self.type_for == 'l2':
                terms['Loss_fd'] = self.network.for_loss_l2(batch.state, batch.next_state, nested_acts)
            elif self.type_for == 'ce':
                terms['Loss_fd'], terms['Acc_fd'] = self.network.for_loss_ce(batch.state, batch.next_state, nested_acts, batch.acts)
            elif self.type_for == 'decode':
                terms['Loss_fd'], terms['Acc_fd'] = self.network.for_loss_decode(batch.state, batch.next_state, nested_acts, hat=True)
            elif self.type_for == 'decode_obs':
                terms['Loss_fd'], terms['Acc_fd'] = self.network.for_loss_decode(batch.state, batch.next_state, nested_acts, hat=False)

            loss += self.w_for * terms['Loss_fd']

        # Backward
        terms.update({'Loss': loss, 'Q': qvals.mean()}) 
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.network.parameters(), self.clip)
        self.optimizer.step()
        return {k: float(v) for k, v in terms.items()} 


    def load(self, path=None):
        if path is None:
            return
        try:
            # self.memory = pickle.load(open(pjoin(path, 'memory.pkl'), 'rb'))
            network = torch.load(pjoin(path, 'model.pt'))
            parts = ['embedding', 'encoder'] # , 'hidden', 'act_scorer']
            state_dict = network.state_dict() 
            state_dict = {k: v for k, v in state_dict.items() if any(part in k for part in parts)}
            # print(state_dict.keys())
            self.network.load_state_dict(state_dict, strict=False)

        except Exception as e:
            print("Error saving model.")
            logging.error(traceback.format_exc())


    def save(self, step=''):
        try:
            os.makedirs(pjoin(self.save_path, step), exist_ok=True)
            pickle.dump(self.memory, open(pjoin(self.save_path, step, 'memory.pkl'), 'wb'))
            torch.save(self.network, pjoin(self.save_path, step, 'model.pt'))
        except Exception as e:
            print("Error saving model.")
            logging.error(traceback.format_exc())
