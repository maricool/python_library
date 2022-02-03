def is_prime(n):
    '''
    Returns True if the number 'n' is prime
    '''
    prime = True
    for i in range(2, n):
        if n%i == 0: 
            prime = False
            break
    return prime

def sum_of_digits(n):
    '''
    Calculates the sum of the digits of a number
    '''
    tot = 0
    for digit in str(n):
        tot += int(digit)
    return tot

def product_of_digits(n):
    '''
    Calculates the product of the digits of a number
    '''
    tot = 1
    for digit in str(n):
        tot *= int(digit)
    return tot