import jax 
import jax.numpy as jnp 

def cosine(a, b):
    return jnp.dot(a, b)/(jnp.linalg.norm(a)*jnp.linalg.norm(b))