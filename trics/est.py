import jax 
import jax.numpy as jnp

def fwl(X, D, Y):
    coeffsD = jnp.linalg.lstsq(X, D, rcond=None)[0]
    dhat = X @ coeffsD
    resD = D - dhat
    coeffsY = jnp.linalg.lstsq(resD, Y, rcond=None)[0][0]
    return coeffsY

def sample(k, p, data, key):
    idx = jax.random.choice(key, jnp.arange(k), shape=(k,), p=p, replace=True)
    selected_X = data.X[idx]
    selected_D = data.D[idx]
    selected_Y = data.Y[idx]
    return selected_X, selected_D, selected_Y

def ols(k, p, data, key):
    X, D, Y = sample(k, p, data, key)
    return fwl(X, D, Y)

def get_residuals(X, D):
    coeffsD = jnp.linalg.lstsq(X, D, rcond=None)[0]
    dhat = X @ coeffsD
    resD = D - dhat
    return resD

def get_fitted(X, D):
    coeffsD = jnp.linalg.lstsq(X, D, rcond=None)[0]
    dhat = X @ coeffsD
    return dhat