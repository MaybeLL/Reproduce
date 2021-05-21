import numpy as np
from numpy.core.fromnumeric import diagonal



if __name__ == '__main__':
    x = np.array([20,2,8,4])
    xiabiao = np.argsort(x)
    print(xiabiao)
    print(x)