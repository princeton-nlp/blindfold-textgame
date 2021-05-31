import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import itertools
from util import pad_sequences
from memory import State


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DRRN(torch.nn.Module):
    """
        Deep Reinforcement Relevance Network - He et al. '16

    """
    def __init__(self, vocab_size, embedding_dim, hidden_dim, fix_rep=0, hash_rep=0, act_obs=0):
        super(DRRN, self).__init__()
        self.hidden_dim = hidden_dim
        self.embedding    = nn.Embedding(vocab_size, embedding_dim)
        self.obs_encoder  = nn.GRU(embedding_dim, hidden_dim)
        self.look_encoder = nn.GRU(embedding_dim, hidden_dim)
        self.inv_encoder  = nn.GRU(embedding_dim, hidden_dim)
        self.act_encoder  = nn.GRU(embedding_dim, hidden_dim)
        self.hidden       = nn.Linear(2 * hidden_dim, hidden_dim)
        # self.hidden       = nn.Sequential(nn.Linear(2 * hidden_dim, 2 * hidden_dim), nn.Linear(2 * hidden_dim, hidden_dim), nn.Linear(hidden_dim, hidden_dim))
        self.act_scorer   = nn.Linear(hidden_dim, 1)
        
        self.state_encoder = nn.Linear(3 * hidden_dim, hidden_dim)
        self.inverse_dynamics = nn.Sequential(nn.Linear(2 * hidden_dim, 2 * hidden_dim), nn.ReLU(), nn.Linear(2 * hidden_dim, hidden_dim)) 
        self.forward_dynamics = nn.Sequential(nn.Linear(2 * hidden_dim, 2 * hidden_dim), nn.ReLU(), nn.Linear(2 * hidden_dim, hidden_dim)) 
        
        self.act_decoder = nn.GRU(hidden_dim, embedding_dim)
        self.act_fc = nn.Linear(embedding_dim, vocab_size)
        
        self.obs_decoder = nn.GRU(hidden_dim, embedding_dim)
        self.obs_fc = nn.Linear(embedding_dim, vocab_size)
        
        self.fix_rep = fix_rep
        self.hash_rep = hash_rep
        self.act_obs = act_obs
        self.hash_cache = {}
    
    def packed_hash(self, x):
        y = []
        for data in x:
            data = hash(tuple(data))
            if data in self.hash_cache:
                y.append(self.hash_cache[data])
            else:
                a = torch.zeros(self.hidden_dim).normal_(generator=torch.random.manual_seed(data))
                # torch.random.seed()
                y.append(a)
                self.hash_cache[data] = a
        y = torch.stack(y, dim=0).to(device)
        return y

    def packed_rnn(self, x, rnn):
        """ Runs the provided rnn on the input x. Takes care of packing/unpacking.

            x: list of unpadded input sequences
            Returns a tensor of size: len(x) x hidden_dim
        """
        if self.hash_rep: return self.packed_hash(x)
        lengths = torch.tensor([len(n) for n in x], dtype=torch.long, device=device)
        # Sort this batch in descending order by seq length
        lengths, idx_sort = torch.sort(lengths, dim=0, descending=True)
        _, idx_unsort = torch.sort(idx_sort, dim=0)
        idx_sort = torch.autograd.Variable(idx_sort)
        idx_unsort = torch.autograd.Variable(idx_unsort)
        padded_x = pad_sequences(x)
        x_tt = torch.from_numpy(padded_x).type(torch.long).to(device)
        x_tt = x_tt.index_select(0, idx_sort)
        # Run the embedding layer
        embed = self.embedding(x_tt).permute(1,0,2) # Time x Batch x EncDim
        # Pack padded batch of sequences for RNN module
        packed = nn.utils.rnn.pack_padded_sequence(embed, lengths)
        # Run the RNN
        out, _ = rnn(packed)
        # Unpack
        out, _ = nn.utils.rnn.pad_packed_sequence(out)
        # Get the last step of each sequence
        idx = (lengths-1).view(-1,1).expand(len(lengths), out.size(2)).unsqueeze(0)
        out = out.gather(0, idx).squeeze(0)
        # Unsort
        out = out.index_select(0, idx_unsort)
        return out
    

    def state_rep(self, state_batch):
        # Zip the state_batch into an easy access format
        state = State(*zip(*state_batch))
        # Encode the various aspects of the state
        with torch.set_grad_enabled(not self.fix_rep):
            obs_out = self.packed_rnn(state.obs, self.obs_encoder)
            if self.act_obs: return obs_out
            look_out = self.packed_rnn(state.description, self.look_encoder)
            inv_out = self.packed_rnn(state.inventory, self.inv_encoder)
            state_out = self.state_encoder(torch.cat((obs_out, look_out, inv_out), dim=1))
        return state_out


    def act_rep(self, act_batch):
        # This is number of admissible commands in each element of the batch
        act_sizes = [len(a) for a in act_batch]
        # Combine next actions into one long list
        act_batch = list(itertools.chain.from_iterable(act_batch))
        with torch.set_grad_enabled(not self.fix_rep):
            act_out = self.packed_rnn(act_batch, self.act_encoder)
        return act_sizes, act_out
    
    
    def for_predict(self, state_batch, acts):
        _, act_out = self.act_rep(acts)
        state_out = self.state_rep(state_batch)
        next_state_out =  state_out + self.forward_dynamics(torch.cat((state_out, act_out), dim=1))
        return next_state_out


    def inv_predict(self, state_batch, next_state_batch):
        state_out = self.state_rep(state_batch)
        next_state_out = self.state_rep(next_state_batch)
        act_out = self.inverse_dynamics(torch.cat((state_out, next_state_out - state_out), dim=1))
        return act_out
    

    def inv_loss_l1(self, state_batch, next_state_batch, acts):
        _, act_out = self.act_rep(acts)
        act_out_hat = self.inv_predict(state_batch, next_state_batch)
        return F.l1_loss(act_out, act_out_hat)
    

    def inv_loss_l2(self, state_batch, next_state_batch, acts):
        _, act_out = self.act_rep(acts)
        act_out_hat = self.inv_predict(state_batch, next_state_batch)
        return F.mse_loss(act_out, act_out_hat)


    def inv_loss_ce(self, state_batch, next_state_batch, acts, valids, get_predict=False):
        act_sizes, valids_out = self.act_rep(valids)
        _, act_out = self.act_rep(acts)
        act_out_hat = self.inv_predict(state_batch, next_state_batch)
        now, loss, acc = 0, 0, 0
        if get_predict: predicts = []
        for i, j in enumerate(act_sizes):
            valid_out = valids_out[now: now + j]
            now += j
            values = valid_out.matmul(act_out_hat[i])
            label = valids[i].index(acts[i][0]) 
            loss += F.cross_entropy(values.unsqueeze(0), torch.LongTensor([label]).to(device))
            predict = values.argmax().item()
            acc += predict == label
            if get_predict: predicts.append(predict)
        return (loss / len(act_sizes), acc / len(act_sizes), predicts) if get_predict else (loss / len(act_sizes), acc / len(act_sizes))
            
    
    def inv_loss_decode(self, state_batch, next_state_batch, acts, hat=True, reduction='mean'):
        # hat: use rep(o), rep(o'); not hat: use rep(a)
        _, act_out = self.act_rep(acts)
        act_out_hat = self.inv_predict(state_batch, next_state_batch)
       
        acts_pad = pad_sequences([act[0] for act in acts])
        acts_tensor = torch.from_numpy(acts_pad).type(torch.long).to(device).transpose(0, 1)
        l, bs = acts_tensor.size()
        vocab = self.embedding.num_embeddings
        outputs = torch.zeros(l, bs, vocab).to(device)
        input, z = acts_tensor[0].unsqueeze(0), (act_out_hat if hat else act_out).unsqueeze(0)
        for t in range(1, l):
            input = self.embedding(input)
            output, z = self.act_decoder(input, z)
            output = self.act_fc(output)
            outputs[t] = output
            top = output.argmax(2)
            input = top
        outputs, acts_tensor = outputs[1:], acts_tensor[1:]
        loss = F.cross_entropy(outputs.reshape(-1, vocab), acts_tensor.reshape(-1), ignore_index=0, reduction=reduction)
        if reduction == 'none':  # loss for each term in batch
            lens = [len(act[0]) - 1 for act in acts]
            loss = loss.reshape(-1, bs).sum(0).cpu() / torch.tensor(lens) 
        nonzero = (acts_tensor > 0)
        same = (outputs.argmax(-1) == acts_tensor)
        acc_token = (same & nonzero).float().sum() / (nonzero).float().sum()  # token accuracy
        acc_action = (same.int().sum(0) == nonzero.int().sum(0)).float().sum() / same.size(1)  # action accuracy
        return loss, acc_action


    def for_loss_l2(self, state_batch, next_state_batch, acts):
        next_state_out = self.state_rep(next_state_batch)
        next_state_out_hat = self.for_predict(state_batch, acts)
        return F.mse_loss(next_state_out, next_state_out_hat) # , reduction='sum')


    def for_loss_ce_batch(self, state_batch, next_state_batch, acts):
        # consider duplicates in next_state_batch
        next_states, labels = [], []
        for next_state in next_state_batch:
            if next_state not in next_states:
                labels.append(len(next_states))
                next_states.append(next_state)
            else:
                labels.append(next_states.index(next_state))
        labels = torch.LongTensor(labels).to(device)
        next_state_out = self.state_rep(next_states)
        next_state_out_hat = self.for_predict(state_batch, acts)
        logits = next_state_out_hat.matmul(next_state_out.transpose(0, 1))
        loss = F.cross_entropy(logits, labels) 
        acc = (logits.argmax(1) == labels).float().sum() / len(labels)
        return loss, acc


    def for_loss_ce(self, state_batch, next_state_batch, acts, valids):
        # classify rep(o') from predict(o, a1), predict(o, a2), ...
        act_sizes, valids_out = self.act_rep(valids)
        _, act_out = self.act_rep(acts)
        next_state_out = self.state_rep(next_state_batch)
        now, loss, acc = 0, 0, 0
        for i, j in enumerate(act_sizes):
            valid_out = valids_out[now: now + j]
            now += j
            next_states_out_hat = self.for_predict([state_batch[i]] * j, [[_] for _ in valids[i]])
            values = next_states_out_hat.matmul(next_state_out[i])
            label = valids[i].index(acts[i][0]) 
            loss += F.cross_entropy(values.unsqueeze(0), torch.LongTensor([label]).to(device))
            predict = values.argmax().item()
            acc += predict == label
        return (loss / len(act_sizes), acc / len(act_sizes)) 


    def for_loss_decode(self, state_batch, next_state_batch, acts, hat=True):
        # hat: use rep(o), rep(a); not hat: use rep(o')
        next_state_out = self.state_rep(next_state_batch)
        next_state_out_hat = self.for_predict(state_batch, acts)
        
        import pdb; pdb.set_trace()
        next_state_pad = pad_sequences(next_state_batch)
        next_state_tensor = torch.from_numpy(next_state_batch).type(torch.long).to(device).transpose(0, 1)
        l, bs = next_state_tensor.size()
        vocab = self.embedding.num_embeddings
        outputs = torch.zeros(l, bs, vocab).to(device)
        input, z = next_state_tensor[0].unsqueeze(0), (next_state_out_hat if hat else next_state_out).unsqueeze(0)
        for t in range(1, l):
            input = self.embedding(input)
            output, z = self.obs_decoder(input, z)
            output = self.obs_fc(output)
            outputs[t] = output
            top = output.argmax(2)
            input = top
        outputs, next_state_tensor = outputs[1:].reshape(-1, vocab), next_state_tensor[1:].reshape(-1)
        loss = F.cross_entropy(outputs, next_state_tensor, ignore_index=0)
        nonzero = (next_state_tensor > 0)
        same = (outputs.argmax(1) == next_state_tensor)
        acc = (same & nonzero).float().sum() / (nonzero).float().sum()  # token accuracy
        return loss, acc


    def forward(self, state_batch, act_batch):
        """
            Batched forward pass.
            obs_id_batch: iterable of unpadded sequence ids
            act_batch: iterable of lists of unpadded admissible command ids

            Returns a tuple of tensors containing q-values for each item in the batch
        """
        state_out = self.state_rep(state_batch)
        act_sizes, act_out = self.act_rep(act_batch)
        # Expand the state to match the batches of actions
        state_out = torch.cat([state_out[i].repeat(j,1) for i,j in enumerate(act_sizes)], dim=0)
        z = torch.cat((state_out, act_out), dim=1) # Concat along hidden_dim
        z = F.relu(self.hidden(z))
        act_values = self.act_scorer(z).squeeze(-1)
        # Split up the q-values by batch
        return act_values.split(act_sizes)


    def act(self, states, act_ids, sample=True, eps=0.1):
        """ Returns an action-string, optionally sampling from the distribution
            of Q-Values.
        """
        act_values = self.forward(states, act_ids)
        if sample:
            act_probs = [F.softmax(vals, dim=0) for vals in act_values]
            act_idxs = [torch.multinomial(probs, num_samples=1).item() \
                        for probs in act_probs]
        else:
            act_idxs = [vals.argmax(dim=0).item() if np.random.rand() > eps else np.random.randint(len(vals)) for vals in act_values]
        return act_idxs, act_values
