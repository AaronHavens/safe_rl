import gym
import math
import numpy as np
import copy


def stoch_perturb(ob,space,power,augmented):
    for i in range(len(ob)):
        eps = np.random.uniform(-1,1)
        delta = eps*(space.high[i]-space.low[i])*power
        ob[i] = max(min(ob[i]+delta,space.high[i]),space.low[i])
    return ob

def unbounded_stoch_perturb(ob,space,power,augmented):
    for i in range(len(ob)):
        eps = np.random.uniform(-1,1)
        delta = eps*(1.0)*power
        ob[i] = ob[i]+delta
    return ob

def unbounded_bias_perturb(ob,space,power,augmented):
    for i in range(len(ob)):
        eps = np.random.uniform(0,1)
        delta = eps*(0.4)*power
        ob[i] = ob[i]+delta
    return ob

def symetric_mirror_perturb(ob,space,power,augmented):
    for i in range(len(ob)):
        ob[i] = max(min(ob[i]*-1,space.high[i]),space.low[i])
    return ob

def ip_bias_perturb(ob,space,power,augmented):
    for i in range(len(ob)):
        eps = np.random.uniform(0,1)
        delta = eps*(2.0)*power
        ob[i] = ob[i]+delta
    return ob

def ip_white_perturb(ob,space,power,augmented):
    for i in range(len(ob)):
        eps = np.random.uniform(-1,1)
        delta = eps*(2.0)*power
        ob[i] = ob[i]+delta
    return ob

def stoch_bias(ob,space,power,augmented):
    for i in range(len(ob)):
        eps = np.random.uniform(0,1)
        delta = eps*(space.high[i]-space.low[i])*power
        ob[i] = max(min(ob[i]+delta,space.high[i]),space.low[i])
    return ob

def stoch_bias_grid(ob,space,power,augmented):
    for i in range(len(ob)):
        delta = np.random.uniform(0,2)
        ob[i] = max(min(int(ob[i]-delta),space.high[i]),space.low[i])
    return ob


def grid_reflect(ob, soace,power, augmented):
    ob[0] = (-(ob[0]-11//2)+11//2)%11
    return ob

def l2norm(x,x_):
    l2_square = 0
    for i in range(len(x)):
        l2square += (x[i]-x_[i])**2
    return math.sqrt(l2square)

class adv_gen():
    def __init__(self,w,ob_space,perturb_func=stoch_perturb,intermittent=False,interval = [1024,1024], delay=0,augmented=True):
        self.w = w
        self.perturb_func = perturb_func
        self.space = ob_space
        self.timer_delay = 0
        self.intermittent = intermittent
        self.adv_duration = interval[0]
        self.adv_interval = interval[1]
        self.augmented = augmented
        self.state_value_map = []
        self.until_attack_timer = interval[1]
        self.attack_timer = interval[0]
        self.isattack = False
        self.delay = delay
    def perturb(self,ob_,t,power=1.0):
        ob = copy.deepcopy(ob_)
        self.timer_delay += 1
        if(self.until_attack_timer == 0): self.isattack = True

        if(self.isattack):
            if(self.attack_timer == 0):
                self.attack_timer = self.adv_duration
                self.until_attack_timer = self.adv_interval
                self.isattack = False
            else:
                self.attack_timer -= 1
        self.until_attack_timer -= 1
        if(self.isattack and t!=0 and np.random.uniform(0,1) < self.w and self.timer_delay > self.delay):
            if self.augmented:
                return self.perturb_func(ob,self.space,power,self.augmented),1
            else:
                return self.perturb_func(ob,self.space,power,self.augmented),0
        else:
            return ob, 0




    def sample_action_space(self,pi,env):
        traj = traj_sampler_ran(pi,env,50,True)
        states = np.array(traj["ob"])
        vpred = np.array(traj["new"])
        inds = np.argsort(vpred)
        sorted_states = states[inds]
        sorted_vpred = np.sort(vpred)
        ten_largest_s = sorted_states[len(sorted_states)-10:len(sorted_states)]
        ten_largest_v = sorted_vpred[len(sorted_states)-10:len(sorted_states)]
        tuples = list()
        for i in range(10):
            self.state_value_map.append(tuple((ten_largest_v[i],ten_largest_s[i])))
        sorted(self.state_value_map, key=lambda x: x[0])
    def over_estimator(self,pi,env, current_state,w):
        if np.random.uniform(0,1) < w and self.timer_delay < 10000:
            if len(self.state_value_map) <= 30:
                self.sample_action_space(pi,env)

            s = self.state_value_map[len(self.state_value_map)-1][1]
            if(self.augmented):
                s[len(s)-1] = 0.0
            del self.state_value_map[len(self.state_value_map)-1]
            return s
        else:
            return current_state


def traj_sampler_ran(pi,env_,horizon,stochastic): #Use this function to get a simulated trajectory sample.
    t = 0
    env = copy.deepcopy(env_)
    ac = env.action_space.sample() # not used, just so we have the datatype
    new = True # marks if we're on first timestep of an episode
    ob_ = env.reset()
    cur_ep_ret = 0 # return in current episode
    cur_ep_len = 0 # len of current episode
    ep_rets = [] # returns of completed episodes in this segment
    ep_lens = [] # lengths of ...
    ob = np.concatenate((ob_,np.array([0.0],dtype=np.float32)),axis=0)
    # Initialize history arrays
    obs = np.array([ob for _ in range(horizon)])
    rews = np.zeros(horizon, 'float32')
    vpreds = np.zeros(horizon, 'float32')
    news = np.zeros(horizon, 'int32')
    acs = np.array([ac for _ in range(horizon)])
    prevacs = acs.copy()

    while True:
        prevac = ac
        ac, vpred = pi.act(stochastic, ob)#no adversary

        if t > 0 and t % horizon == 0:
            return {"ob" : obs, "rew" : rews, "vpred" : vpreds, "new" : news,
                    "ac" : acs, "prevac" : prevacs, "nextvpred": vpred * (1 - new),
                    "ep_rets" : ep_rets, "ep_lens" : ep_lens}

            ep_rets = []
            ep_lens = []
        i = t % horizon
        obs[i] = ob
        vpreds[i] = vpred
        news[i] = new
        acs[i] = ac
        prevacs[i] = prevac
        #Introduce adversary here?
        ob_, rew, new, _ = env.step(ac)
        ob = np.concatenate((ob_,np.array([0.0],dtype=np.float32)),axis=0)
        rews[i] = rew

        cur_ep_ret += rew
        cur_ep_len += 1
        if new:
            ep_rets.append(cur_ep_ret)
            ep_lens.append(cur_ep_len)
            cur_ep_ret = 0
            cur_ep_len = 0
            ob_ = env.reset()
            ob = np.concatenate((ob_,np.array([0.0],dtype=np.float32)),axis=0)
        t += 1
