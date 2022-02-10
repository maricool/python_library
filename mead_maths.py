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

def is_even(num):
    '''
    Returns true if integer is even
    '''
    if num%2 == 0:
        return True
    else:
        return False

def is_odd(num):
    '''
    True if integer is odd
    '''
    return not is_even(num)

def ceiling(a, b):
    '''
    Ceiling division a/b
    TODO: Could also use math.ceiling
    '''
    return -(-a // b)

def reverse_digits(old_number):
    '''
    Reverses the digits of integer n
    '''
    new_number = 0
    n = old_number
    while n != 0:
        last_digit = n%10 # Isolate the final digit using modulus
        n = n//10 # Floor division to remove the final digit
        new_number = new_number*10+last_digit # Construct new number
    return new_number
    #return int(str(n)[::-1]) # Easy using strings


