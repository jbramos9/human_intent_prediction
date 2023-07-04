import random
from typing import List

class GazeTransform:
    def __init__(self, p=0.1):
        self.p = p
    

class RandomShift(GazeTransform):
    def __init__(self, mean=10, std=5, p=0.1):
        super().__init__(p)
        self.mean = mean
        self.std = std
    
    def __call__(self, x):
        if random.random() < self.p:
            try:
                shift = int(random.gauss(self.mean, self.std))
                # Ensure the shift is within the length of the list and retain sign
                if shift > 0:
                    shift = shift % len(x)
                elif shift < 0:
                    shift = (shift % len(x))*-1

                return x[-shift:] + x[:-shift]
            except:
                print("Something is wrong in RandomShift")
                return x
        else:
            return x

class RandomStretch(GazeTransform):
    def __init__(self, mean=1, std=0.3, p=0.1):
        super().__init__(p)
        self.mean = mean
        self.std = std

    def __call__(self, x) -> List:
        if random.random() < self.p:
            try:
                f = random.gauss(self.mean, self.std)
                new_size = int( f * len(x))
                return self._stretch(x, new_size)
            except:
                print("Something is wrong in RandomStretch")
                return x
        else:
            return x
    
    def _stretch(self, arr, new_size) -> List:
        scaling_factor = len(arr) / new_size
        new_array = []

        # loop through the new array and copy elements from the original array
        for i in range(new_size):
            index = int(i * scaling_factor)
            if index < len(arr):
                new_array.append(arr[index])
        return new_array
    

class RandomBlanks(GazeTransform):
    def __init__(self, mean=10, std=3, p=0.1):
        super().__init__(p)
        self.mean = mean
        self.std = std
    
    def __call__(self, x):
        if random.random() < self.p:
            try:
                space = int(abs(random.gauss(self.mean, self.std))) # random length of the blank
                blank = ["none" for _ in range(space)]
                idx = random.randint(0, len(x) + 1)     # random location

                return x[:idx] + blank + x[idx:]
            except:
                print("Something is wrong in RandomBlanks")
                return x
        return x
