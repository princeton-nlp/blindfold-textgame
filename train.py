import subprocess
import time
import os
import torch
import logger
import argparse
import jericho
import logging
import json
from os.path import basename, dirname
from drrn import * 
from env import JerichoEnv
from jericho.util import clean
from copy import deepcopy
from vec_env import VecEnv


logging.getLogger().setLevel(logging.CRITICAL)
subprocess.run("python -m spacy download en_core_web_sm".split())

def configure_logger(log_dir, wandb):
    logger.configure(log_dir, format_strs=['log'])
    global tb
    type_strs = ['json', 'stdout']
    if wandb and log_dir != 'logs': type_strs += ['wandb']
    tb = logger.Logger(log_dir, [logger.make_output_format(type_str, log_dir) for type_str in type_strs])
    global log
    log = logger.log


def evaluate(agent, env, nb_episodes=1):
    with torch.no_grad():
        total_score = 0
        for ep in range(nb_episodes):
            log("Starting evaluation episode {}".format(ep))
            score = evaluate_episode(agent, env)
            log("Evaluation episode {} ended with score {}\n\n".format(ep, score))
            total_score += score
        avg_score = total_score / nb_episodes
        return avg_score


def evaluate_episode(agent, env):
    step = 0
    done = False
    ob, info = env.reset()
    state = agent.build_state(ob, info)
    log('Obs{}: {} Inv: {} Desc: {}'.format(step, clean(ob), clean(info['inv']), clean(info['look'])))
    while not done:
        valid_acts = info['valid']
        valid_ids = agent.encode(valid_acts)
        _, action_idx, action_values = agent.act([state], [valid_ids], sample=False)
        action_idx = action_idx[0]
        action_values = action_values[0]
        action_str = valid_acts[action_idx]
        log('Action{}: {}, Q-Value {:.2f}'.format(step, action_str, action_values[action_idx].item()))
        s = ''
        for idx, (act, val) in enumerate(sorted(zip(valid_acts, action_values), key=lambda x: x[1], reverse=True), 1):
            s += "{}){:.2f} {} ".format(idx, val.item(), act)
        log('Q-Values: {}'.format(s))
        ob, rew, done, info = env.step(action_str)
        log("Reward{}: {}, Score {}, Done {}".format(step, rew, info['score'], done))
        step += 1
        log('Obs{}: {} Inv: {} Desc: {}'.format(step, clean(ob), clean(info['inv']), clean(info['look'])))
        state = agent.build_state(ob, info)
    return info['score']


def train(agent, eval_env, envs, max_steps, update_freq, eval_freq, checkpoint_freq, log_freq, r_for):
    start, max_score, max_reward = time.time(), 0, 0
    obs, infos = envs.reset()
    states = agent.build_states(obs, infos)
    valid_ids = [agent.encode(info['valid']) for info in infos]
    transitions = [[] for info in infos] 
    for step in range(1, max_steps+1): 
        action_ids, action_idxs, action_values = agent.act(states, valid_ids, sample=True, eps=0.05 ** (step / max_steps))
        action_strs = [info['valid'][idx] for info, idx in zip(infos, action_idxs)]
        
        # log envs[0] 
        examples = [(action, value) for action, value in zip(infos[0]['valid'], action_values[0].tolist())]
        examples = sorted(examples, key=lambda x: -x[1])
        log('State  {}: {}'.format(step, clean(obs[0] + infos[0]['inv'] + infos[0]['look'])))
        log('Actions{}: {}'.format(step, [action for action, _ in examples]))
        log('Qvalues{}: {}'.format(step, [round(value, 2) for _, value in examples]))
        log('>> Action{}: {}'.format(step, action_strs[0]))
        
        # step
        obs, rewards, dones, infos = envs.step(action_strs)
        next_states = agent.build_states(obs, infos)
        next_valids = [agent.encode(info['valid']) for info in infos]
        if r_for > 0:
            reward_curiosity, _ = agent.network.inv_loss_decode(states, next_states, [[a] for a in action_ids], hat=True, reduction='none')
            rewards = rewards + reward_curiosity.detach().numpy() * r_for
            tb.logkv_mean('Curiosity', reward_curiosity.mean().item())

        for i, (ob, reward, done, info, state, next_state) in enumerate(zip(obs, rewards, dones, infos, states, next_states)):
            transition = Transition(state, action_ids[i], reward, next_state, next_valids[i], done, valid_ids[i])
            transitions[i].append(transition)
            agent.observe(transition)
            if i == 0:
                log("Reward{}: {}, Score {}, Done {}\n".format(step, reward, info['score'], done))
            if done:
                tb.logkv_mean('EpisodeScore', info['score'])
                # obs[i], infos[i] = env.reset()
                # next_states[i] = agent.build_state(obs[i], infos[i])
                # next_valids[i] = agent.encode(infos[i]['valid'])
                if info['score'] >= max_score:  # put in alpha queue
                    if info['score'] > max_score:
                        agent.memory.clear_alpha()
                        max_score = info['score']
                    for transition in transitions[i]:
                        agent.observe(transition, is_prior=True)
                transitions[i] = []

        states, valid_ids = next_states, next_valids
        if step % log_freq == 0:
            tb.logkv('Step', step)
            tb.logkv("FPS", int((step*envs.num_envs)/(time.time()-start)))
            tb.logkv("EpisodeScores100", envs.get_end_scores().mean()) 
            tb.logkv('MaxScore', max_score)
            tb.logkv('Step', step)
            # if envs[0].cache is not None:
            #     tb.logkv('#dict', len(envs[0].cache)) 
            #     tb.logkv('#locs', len(envs[0].cache['loc'])) 
            tb.dumpkvs()
        if step % update_freq == 0:
            res = agent.update()
            if res is not None:
                for k, v in res.items():
                    tb.logkv_mean(k, v)
        if step % checkpoint_freq == 0:
            agent.save(str(step))
            # json_path = envs[0].rom_path.replace('.z5', '.json')
            # if os.path.exists(json_path):
            #     envs[0].cache.update(json.load(open(json_path)))
            # json.dump(envs[0].cache, open(json_path, 'w'))
        if step % eval_freq == 0:
            eval_score = evaluate(agent, eval_env)
            tb.logkv('EvalScore', eval_score)
            tb.dumpkvs()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', default='logs')
    parser.add_argument('--load', default=None)
    parser.add_argument('--spm_path', default='unigram_8k.model')
    parser.add_argument('--rom_path', default='zork1.z5')
    parser.add_argument('--env_step_limit', default=100, type=int)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--num_envs', default=8, type=int)
    parser.add_argument('--max_steps', default=100000, type=int)
    parser.add_argument('--update_freq', default=1, type=int)
    parser.add_argument('--checkpoint_freq', default=10000, type=int)
    parser.add_argument('--eval_freq', default=5000, type=int)
    parser.add_argument('--log_freq', default=100, type=int)
    parser.add_argument('--memory_size', default=10000, type=int)
    parser.add_argument('--memory_alpha', default=.4, type=float)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--gamma', default=.9, type=float)
    parser.add_argument('--learning_rate', default=0.0001, type=float)
    parser.add_argument('--clip', default=5, type=float)
    parser.add_argument('--embedding_dim', default=128, type=int)
    parser.add_argument('--hidden_dim', default=128, type=int)

    parser.add_argument('--wandb', default=1, type=int) 

    parser.add_argument('--type_inv', default='decode') 
    parser.add_argument('--type_for', default='ce') 
    parser.add_argument('--w_inv', default=0, type=float) 
    parser.add_argument('--w_for', default=0, type=float) 
    parser.add_argument('--w_act', default=0, type=float) 
    parser.add_argument('--r_for', default=0, type=float) 

    parser.add_argument('--nor', default=0, type=int, help='no game reward') 
    parser.add_argument('--randr', default=0, type=int, help='random game reward by objects and locations within episode') 
    parser.add_argument('--perturb', default=0, type=int, help='perturb state and action') 

    parser.add_argument('--hash_rep', default=0, type=int, help='hash for representation') 
    parser.add_argument('--act_obs', default=0, type=int, help='action set as state representation') 
    parser.add_argument('--fix_rep', default=0, type=int, help='fix representation') 
    return parser.parse_known_args()[0]


def main():
    args = parse_args()
    print(args)
    configure_logger(args.output_dir, args.wandb)
    agent = DRRN_Agent(args)
    agent.load(args.load) 
    # cache = {'loc': set()} 
    cache = None
    if args.perturb:
        args.en2de = torch.hub.load('pytorch/fairseq', 'transformer.wmt19.en-de.single_model')
        args.de2en = torch.hub.load('pytorch/fairseq', 'transformer.wmt19.de-en.single_model')
        args.en2de.eval()
        args.de2en.eval()
        args.en2de.cuda()
        args.de2en.cuda()
        args.perturb_dict = {}

    env = JerichoEnv(args.rom_path, args.seed, args.env_step_limit, get_valid=True, cache=cache, args=args)
    # envs = [JerichoEnv(args.rom_path, args.seed, args.env_step_limit, get_valid=True, cache=cache, args=args) for _ in range(args.num_envs)]
    envs = VecEnv(args.num_envs, env)
    train(agent, env, envs, args.max_steps, args.update_freq, args.eval_freq, args.checkpoint_freq, args.log_freq, args.r_for)


if __name__ == "__main__":
    main()
