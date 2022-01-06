#### This class generate free form functions

import random
import numpy as np
from inspect import signature
from tqdm.auto import tqdm

class FuzzyBooleanGenerator():

    def __init__(self):
        self.options_dict = {'atom':self.atom,'not':self.fuzzy_not, \
        'and':self.fuzzy_and,'or':self.fuzzy_or}
        self.options_scores = [0.3, 1, 1, 1]

    def generate_expression(self, inputs:list, seed:int , max_depth: int):
        seeded_random = random.Random(seed)
        return self._call_generate_expression(inputs, seeded_random, max_depth)
    
    def _call_generate_expression(self, inputs:list, seeded_random, max_depth):
        if max_depth == 0:
            return seeded_random.choice(inputs)
        else:
            option_name = seeded_random.choices(list(self.options_dict.keys()), weights=self.options_scores, k=1)[0]
            option_func = self.options_dict[option_name]
            if option_name == 'atom':
                arg = seeded_random.choice(inputs)
                return option_func(arg)
            else:
                num_args = len(signature(option_func).parameters)
                args = [self._call_generate_expression(inputs, seeded_random, max_depth-1) for _ in range(num_args)]
                return option_func(*args)

    def call_generate_expression_str(inputs:list, seeded_random, max_depth):
        if max_depth == 0:
            return str(seeded_random.choice(inputs))
        else:
            option_name = seeded_random.choices(list(self.options_dict.keys()), weights=self.options_scores, k=1)[0]
            option_func = self.options_dict[option_name]
            if option_name == 'atom':
                arg = seeded_random.choice(inputs)
                return str(option_func(arg))
            else:
                num_args = len(signature(option_func).parameters)
                args = [self.call_generate_expression_str(inputs, seeded_random, max_depth-1) for _ in range(num_args)]
                string = option_name + "(" + ', '.join(args) + ")"
                #string = option_name + ' '.join(args)
                return string
    
    def generate_dataset(self, num_points=163480, input_dim=5, max_depth=5, function_seeds:list=[1,2,3], path='./temp.npy'):
        num_functions = len(function_seeds)
        X = np.random.uniform(low=0.0, high=1.0, size=(num_points, input_dim))
        Y = np.zeros((num_points, num_functions))
        for i in tqdm(range(num_points)):
            for j in range(num_functions):
                Y[i,j] = self.generate_expression(X[i], function_seeds[j], max_depth)
        data = {'X':X, 'Y':Y}
        np.save(path,data)

    def atom(self, a):
        return a

    def fuzzy_not(self, a):
        return 1-a

    def fuzzy_and(self, a, b):
        return a*b

    def fuzzy_or(self, a, b):
        return 1 - (1 - a) * (1 - b)
    



        