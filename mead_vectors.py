# Check if a square array/matrix is symmetric
def check_symmetric(a, rtol=1e-05, atol=1e-08):
    from numpy import allclose
    return allclose(a, a.T, rtol=rtol, atol=atol)

# Check if a square array/matrix is antisymmetric
def check_antisymmetric(a, rtol=1e-05, atol=1e-08):
    from numpy import allclose
    return allclose(a, -a.T, rtol=rtol, atol=atol)

# Check if a square array/matrix is Hermitian
def check_antisymmetric(a, rtol=1e-05, atol=1e-08):
    from numpy import allclose, conj
    return allclose(a, conj(a.T), rtol=rtol, atol=atol)