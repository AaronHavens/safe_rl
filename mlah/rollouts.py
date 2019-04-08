import numpy as np
import math
import time
import copy
#from grad_attack import *
from collections import deque
import matplotlib.pyplot as plt

def adv_scope(rew_,new_,vpred1_,vpred2_,last_a):#compute advantage with rolling inputs
    #[(ob,a,r,new),...,(ob,a,r,new)] Computes the advantage for each policy given that they would have acted
    T = len(rew_)
    rew = np.array(rew_)
    new = np.array(new_)
    vpred1 = np.array(vpred1_)
    vpred2 = np.array(vpred2_)
    adv_1 = np.empty(T,'float32')
    adv_2 = np.empty(T,'float32')
    lastgaelam1 = 0
    lastgaelam2 = 0
    gamma = 0.99
    lam = 0.98
    for t in reversed(range(T)):
        if t == T-1:
            nonterminal = 0
            vpred1_next = 0
            vpred2_next = 0
        else:
            nonterminal = 1-new[t+1]
            vpred1_next = vpred1[t+1]
            vpred2_next = vpred2[t+1]

        delta1 = rew[t] + gamma * vpred1_next * nonterminal - vpred1[t]
        delta2 = rew[t] + gamma * vpred2_next * nonterminal - vpred2[t]

        adv_1[t] = lastgaelam1 = delta1 + gamma * lam * nonterminal * lastgaelam1
        adv_2[t] = lastgaelam2 = delta2 + gamma * lam * nonterminal * lastgaelam2


    return np.array([adv_1[0],adv_2[0]],dtype=np.float32) #return advantage of last T_stop states. kind of like s,a,r,pi abstraction

def traj_segment_generator(adv_generator, master_pol,sub_policies, env, horizon, stochastic, args):
    #replay = args.replay
    t = 0
    i = 0
    ac = env.action_space.sample()
    new = True
    rew = 0.0
    ob_ = env.reset()
    cur_subpolicy = 0
    macro_vpred = 0
    #macro_horizon = math.ceil(horizon/macrolen)
    counts = np.array([0,0])
    counts_action = np.array([0,0])
    cur_ep_ret = 0
    cur_ep_len = 0
    ep_rets = []
    ep_lens = []
    adv_space = []
    # Initialize history arrays
    obs = np.array([ob_ for _ in range(horizon)])
    macro_obs = np.array([np.array([0,0],dtype=np.float32) for _ in range(horizon)])
    rews = np.zeros(horizon, 'float32')
    vpreds = np.zeros(horizon, 'float32')
    q_len = 2
    vpreds1 = deque(maxlen=q_len)
    vpreds2 = deque(maxlen=q_len)
    rews_q = deque(maxlen=q_len)
    news_q = deque(maxlen=q_len)
    news = np.zeros(horizon, 'int32')
    acs = np.array([ac for _ in range(horizon)])
    macro_acs = np.zeros(horizon, 'int32')
    pol_choice = np.zeros(horizon, 'int32')
    macro_vpreds = np.zeros(horizon, 'float32')
    last_policy = 0
    cur_pol = 0


    x = 0
    z = 0

    # total = [0,0]
    # tt = 0

    while True:
        '''
        if t % macrolen == 0:
            cur_subpolicy, macro_vpred = pi.act(stochastic, ob)

            if np.random.uniform() < 0.1:
                cur_subpolicy = np.random.randint(0, len(sub_policies))
            if args.force_subpolicy is not None:
                cur_subpolicy = args.force_subpolicy
                z += 1
        '''
        #determined by adversary here
        #class, ob <- perturbation
        #cur_subpolicy <- ob[adv state]
        #ob = ob_
        #latent = env.env.goal
        #latent = env.env.goal
        ob, latent = adv_generator.perturb(ob_,t)
        counts[latent] += 1
        cur_subpolicy = latent

        if args.pretrain < 0:
            ac1,vpred1 = sub_policies[0].act(stochastic, ob)
            ac2,vpred2 = sub_policies[1].act(stochastic, ob)
            rews_q.append(rew)
            news_q.append(new)
            vpreds1.append(vpred1)
            vpreds2.append(vpred2)

        if len(rews_q) >= 1 and args.pretrain < 0:
            macro_ob = adv_scope(rews_q,news_q,vpreds1,vpreds2,last_policy)
            if t%1==0:
                macro_ac, macro_pred = master_pol.act(True,macro_ob)



        elif args.pretrain >= 0:
            macro_ac = 0
            macro_pred = 0
            cur_pol = args.pretrain
            #macro_ob = np.array([args.pretrain],dtype=np.float32)
            macro_ob = np.array([0],dtype=np.float32)

        last_policy = cur_pol
        #cur_pol = macro_ac
        if macro_ac == 1:
            if cur_pol == 0: cur_pol = 1
            else: cur_pol = 0
        #cur_pol = latent
        counts_action[cur_pol] += 1
        #counts[macro_ac] += 1
        #macro_ac = latent
        ac, vpred = sub_policies[cur_pol].act(stochastic, ob)
        # if np.random.uniform(0,1) < 0.05:
            # ac = env.action_space.sample()

        if t > 0 and t % horizon == 0:
            # tt += 1
            # print(total)
            # total = [0,0]
            print('latent objectives',counts)
            dicti = {"ob" : obs, "rew" : rews, "vpred" : vpreds,
                    "new" : news, "ac" : acs, "ep_rets" : ep_rets,
                    "ep_lens" : ep_lens, "macro_ac" : macro_acs, "pol_choice" : pol_choice,
                    "macro_vpred" : macro_vpreds, "macro_ob" : macro_obs,"latent_counts" : counts,"real_counts":counts_action}
            yield {key: np.copy(val) for key,val in dicti.items()}
            ep_rets = []
            ep_lens = []
            counts = np.array([0,0])
            counts_action = np.array([0,0])
            x += 1

        i = t % horizon
        obs[i] = ob
        vpreds[i] = vpred
        news[i] = new
        acs[i] = ac
        macro_acs[i] = macro_ac
        pol_choice[i] = cur_pol
        macro_vpreds[i] = macro_pred
        macro_obs[i] = macro_ob

        '''
        if t % macrolen == 0:
            macro_acs[int(i/macrolen)] = cur_subpolicy
            macro_vpreds[int(i/macrolen)] = macro_vpred
        '''

        ob_, rew, new, info = env.step(ac)
        rews[i] = rew
        ''' we dont want to render ever for now
        if t > 2048*30:
            env.render()        # print(info)
            pass
        '''
        #if x%20 == 0:
        #env.render()
        if cur_ep_len > 20000:
            new = True
        cur_ep_ret += rew
        cur_ep_len += 1

        if new:# and ((t+1) % macrolen == 0):
        # if new:
            ep_rets.append(cur_ep_ret)
            ep_lens.append(cur_ep_len)
            vpreds1 = deque(maxlen=q_len)
            vpreds2 = deque(maxlen=q_len)
            rews_q = deque(maxlen=q_len)
            news_q = deque(maxlen=q_len)
            cur_ep_ret = 0
            cur_ep_len = 0
            ob_ = env.reset()
            last_policy = 0
        t += 1
#probs wont use this for a while
def add_advantage_macro(seg, gamma, lam):
    new = np.append(seg["new"], 0) # last element is only used for last vtarg, but we already zeroed it if last new = 1
    vpred = np.append(seg["macro_vpred"], 0)
    T = int(len(seg["rew"]))
    seg["macro_adv"] = gaelam = np.empty(T, 'float32')
    rew = seg["rew"]
    lastgaelam = 0
    for t in reversed(range(T)):
        nonterminal = 1 - new[t+1]
        #if t < T-1:
            #if seg["macro_ac"][t] != seg["macro_ac"][t+1]: lastgaelam = 0
        delta = rew[t] + gamma * vpred[t+1] * nonterminal - vpred[t]
        gaelam[t] = lastgaelam = delta #+ gamma * lam * nonterminal * lastgaelam
    seg["macro_tdlamret"] = seg["macro_adv"] + seg["macro_vpred"]

    # print(seg["macro_ac"])
    # print(rew)
    # print(seg["macro_adv"])
    #seg["macro_ob"] = seg["ob"][0::macrolen]
#used to prepare each segmented rollout



def prepare_allrolls(allrolls, gamma, lam, num_subpolicies):
    for i in range(len(allrolls) - 1):
        for key,value in allrolls[i + 1].items():
            allrolls[0][key] = np.append(allrolls[0][key], value, axis=0)
    test_seg = allrolls[0]
    # calculate advantages
    new = np.append(test_seg["new"], 0)
    vpred = np.append(test_seg["vpred"], 0)
    T = len(test_seg["rew"])
    test_seg["adv"] = gaelam = np.empty(T, 'float32')
    rew = test_seg["rew"]
    lastgaelam = 0
    indices = []
    for t in reversed(range(T)):
        nonterminal = 1-new[t+1]
        if t < T-1:
            if test_seg["pol_choice"][t] == test_seg["pol_choice"][t+1]:  #Try not to roll in expectation of other MDPs
                delta = rew[t] + gamma * vpred[t+1] * nonterminal - vpred[t]
                gaelam[t] = lastgaelam = delta #+ gamma * lam * nonterminal * lastgaelam
                indices.append(t)
            else: lastgaelam = 0
    test_seg["tdlamret"] = test_seg["adv"][indices] + test_seg["vpred"][indices]

    split_test = split_segments(test_seg, num_subpolicies, indices)
    print(len(split_test[0]['ob']),len(split_test[1]['ob']))
    return split_test

def split_segments(seg, num_subpolicies, indices):
    subpol_counts = []
    for i in range(num_subpolicies):
        subpol_counts.append(0)
    for macro_ac in seg["pol_choice"]:
        subpol_counts[macro_ac] += 1
    subpols = []
    for i in range(num_subpolicies):
        obs = np.array([seg["ob"][0] for _ in range(subpol_counts[i])])
        advs = np.zeros(subpol_counts[i], 'float32')
        tdlams = np.zeros(subpol_counts[i], 'float32')
        acs = np.array([seg["ac"][0] for _ in range(subpol_counts[i])])
        subpols.append({"ob": obs, "adv": advs, "tdlamret": tdlams, "ac": acs})
    subpol_counts = []
    for i in range(num_subpolicies):
        subpol_counts.append(0)
    j = 0
    for i in indices:
        mac = seg["pol_choice"][i]
        subpols[mac]["ob"][subpol_counts[mac]] = seg["ob"][i]
        subpols[mac]["adv"][subpol_counts[mac]] = seg["adv"][i]
        subpols[mac]["tdlamret"][subpol_counts[mac]] = seg["tdlamret"][j]
        subpols[mac]["ac"][subpol_counts[mac]] = seg["ac"][i]
        subpol_counts[mac] += 1
        j += 1
    return subpols
