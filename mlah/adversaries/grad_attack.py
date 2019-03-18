import numpy as np
import copy
from dataset import Dataset

class adv_gen():
    def __init__(self,ob_space,grad_func, interval = [5000,5000],delay=0,dummy=False):
        self.grad_func = grad_func
        self.space = ob_space
        self.timer_delay = delay
        self.adv_duration = interval[0]
        self.adv_interval = interval[1]
        self.until_attack_timer = interval[1]
        self.attack_timer = interval[0]
        self.isattack = False
        self.dummy = dummy

    def perturb(self,ob_):
        ob = copy.deepcopy(ob_)
        self.timer_delay -= 1
        if(self.timer_delay < 0 and not self.dummy):
            if(self.until_attack_timer == 0): self.isattack = True

            if(self.isattack):
                if(self.attack_timer == 0):
                    self.attack_timer = self.adv_duration
                    self.until_attack_timer = self.adv_interval
                    self.isattack = False
                else:
                    self.attack_timer -= 1
            self.until_attack_timer -= 1

            if(self.isattack):
                return self.grad_attack(ob),1
            else:
                return ob, 0
        else:
            return ob, 0

    def grad_attack(self, ob):
        ob_attack = np.array(copy.deepcopy(ob),dtype=np.float32)
        for i in range(10):
            grad = self.grad_func(np.expand_dims(ob_attack, axis=0))
            ob_attack -= 0.2*grad[0]
        #ob_attack[0] = min(max(ob_attack[0],0),11)
        #ob_attack[1] = min(max(ob_attack[1],0),11)
        return ob_attack
