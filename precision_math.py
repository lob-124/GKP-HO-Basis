from decimal import Decimal

"""
Module for doing arbitrary-precision arithmetic using the decimal module

Author: Liam O'Brien
"""

#Useful constants
_pi = Decimal('3.141592653589793')
_sqrt_pi = _pi.sqrt()


fac_dict = {0: Decimal(1), 1: Decimal(1)}
def my_factorial(n):
    """
    Compute n!. Assumes n is an int
    """
    global fac_dict
    if fac_dict.get(n) is not None:
        return fac_dict[n]
    else:
        res = Decimal(n) * my_factorial(n-1)
        fac_dict[n] = res
        return res


fac2_dict = {0: Decimal(1), 1: Decimal(1)}
def my_factorial2(n):
    """
    Compute n!!. Assumes n is an int
    """
    global fac2_dict
    if fac2_dict.get(n) is not None:
        return fac2_dict[n]
    else:
        res = Decimal(n) * my_factorial2(n-2)
        fac2_dict[n] = res
        return res


def my_herm0(n):
    """
    Compute H_n(0) - the nth (physicists) Hermite polynomial evaluated at zero
    """
    if n % 2 == 1:
        return Decimal(0)
    elif n == 0:
        return Decimal(1)
    else:
        return Decimal(-2)**(n//2) * my_factorial2(n-1)


def my_herm(n):
    """
    Compute H_n(x) - the nth (physicists) Hermite polynomial evaluated at x
    """
    if n == 0:
        return lambda x : Decimal(1)
    if n == 1:
        return lambda x: Decimal(2*x)
    else:
        nfac = my_factorial(n)
        zero_val = my_herm0(n)
        def f(x):
            if x == 0:
        	    return zero_val
            _sum = 0
            for k in range(n//2+1):
                _sum += (-1)**k/(my_factorial(k) * my_factorial(n-2*k)) * Decimal(2*x)**(n-2*k)
            return nfac*_sum
        return f


def my_ho_wavefunction(n,sigma=1):
    """
    Implementation of nth QHO wavefunction using exact decimal arithmetic

    NB: This function is currently not vectorized! The resulting function must be called with one 
        value at a time
    """
    Ha = my_herm(n)
    _sig = Decimal(sigma)
    _prefac =  1/(2**n*my_factorial(n)*_sig*_sqrt_pi).sqrt()
    def f(phi):
        _exp_arg = Decimal(-phi**2/(2*sigma**2))
        res = _prefac*_exp_arg.exp()*Ha(phi/sigma)
        return float(res)

    return f