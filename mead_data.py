import numpy as np

def print_array_statistics(x):
    # Print useful array statistics
    from numpy import sqrt
    print('Array statistics')
    n = x.size
    print('size:', n)
    print('sum:', x.sum())
    print('min:', x.min())
    print('max:', x.max())
    mean = x.mean()
    print('mean:', mean)
    std = x.std()
    var = std**2
    #std_bc = x.std(ddof=1) # Avoiding this to prevent multiple similar calculations
    std_bc = std*sqrt(n/(n-1))
    var_bc = std_bc**2
    print('std:', std)
    print('std (Bessel corrected):', std_bc)
    print('variance:', var)
    print('variance (Bessel corrected):', var_bc**2)
    print('<x^2>:', mean**2+var)
    print()

def standardize(x):
         
    '''
    From: https://towardsdatascience.com/pca-with-numpy-58917c1d0391
    This function standardizes an array, its substracts mean value, 
    and then divide the standard deviation.
    
    param 1: array 
    return: standardized array
    '''    
    rows, columns = x.shape
    
    standardizedArray = np.zeros(shape=(rows, columns))
    tempArray = np.zeros(rows)
    
    for col in range(columns):
        mean = np.mean(x[:, col])
        std = np.std(x[:, col])
        tempArray = np.empty(0)
        
        for element in x[:, col]:
            tempArray = np.append(tempArray, (element-mean)/std)
 
        standardizedArray[:, col] = tempArray
    
    return standardizedArray