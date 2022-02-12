import random
from torch.utils.data import Dataset
import numpy as np
import math

class SOPBooleanDataset(Dataset):
    
    def __init__(self, size_inputs, num_points, func_codes:list):
        
        self.size_inputs = size_inputs
        self.num_points = num_points
        self.func_codes = []

        self.X = np.random.uniform(low=0.0, high=1.0, size=(num_points, size_inputs))
        self.Y = np.empty((num_points, len(func_codes)))

        for func_code in func_codes:
            binary_code = format(func_code, 'b')
            binary_code = binary_code.rjust(math.ceil( len(binary_code) / size_inputs ) * size_inputs, '0')
            self.func_codes.append(binary_code)
        
        for idx, func_code in enumerate(self.func_codes):
            self._calculate_function(idx, func_code)
    
    def _calculate_function(self, func_idx, func_code: str):
        code = func_code
        size_inputs = self.size_inputs
        num_points = self.num_points
        subcodes = [code[i:i+size_inputs] for i in range(0,len(code),size_inputs)]
        for idx in range(num_points):
            sum_result = 0
            for subcode in subcodes:
                product_result = 1
                for term_id in range(size_inputs):
                    if subcode[term_id] == '1':
                        temp = self.X[idx][term_id]
                    else:
                        temp = 1-self.X[idx][term_id]
                    product_result = product_result * temp
                sum_result = 1 - (1-sum_result) * (1-product_result)
                self.Y[idx][func_idx] = sum_result


    
    def __len__(self):
        return self.num_points

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]