import jax 
import jax.numpy as jnp
from trics.data import Data 

def sample(k, p, data, key):
    idx = jax.random.choice(key, jnp.arange(k), shape=(k,), p=p, replace=True)
    selected_X = data.X[idx]
    selected_D = data.D[idx]
    selected_Y = data.Y[idx]
    if data.Z is none:
        return Data(selected_X, selected_D, selected_Y)
    else:
        selected_Z = data.Z[idx]
        return Data(selected_X, selected_D, selected_Y, selected_Z) 

def get_residuals(X, D):
    coeffsD = jnp.linalg.lstsq(X, D, rcond=None)[0]
    dhat = X @ coeffsD
    resD = D - dhat
    return resD
    
def get_fitted(X, D):
    coeffsD = jnp.linalg.lstsq(X, D, rcond=None)[0]
    dhat = X @ coeffsD
    return dhat
    
def fwl(data):
    coeffsD = jnp.linalg.lstsq(data.X, data.D, rcond=None)[0]
    dhat = data.X @ coeffsD
    resD = data.D - dhat
    coeffsY = jnp.linalg.lstsq(resD, data.Y, rcond=None)[0][0]
    return coeffsY
    
def iv(data):
    dhat = get_fitted(jnp.hstack((data.X, data.Z)), data.D) 
    data = Data(data.X, dhat, data.Y)
    return fwl(data)

    # Y = (data.Y == outcome).astype(jnp.float32)
    # new_data = Data(X=data.X, Z=data.Z, D=data.D, Y=Y)
    # sample_data = sample(k, n, new_data, key)
