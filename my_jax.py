import jax
import jax.numpy as jnp
from jax import grad, jit, vmap


# Define a simple quadratic loss function: loss(params) = (params[0]^2 + params[1]^2)
def loss(params):
    return params[0] ** 2 + params[1] ** 2


# Compute gradient (automatic differentiation)
grad_loss = grad(loss)

# JIT-compile for speed
jit_grad_loss = jit(grad_loss)

# Vectorize over a batch of parameters (e.g., for parallel computation)
vmap_grad_loss = vmap(jit_grad_loss)

# Example usage
params = jnp.array([1.0, 2.0])  # Single set of params
print("Gradient for single params:", jit_grad_loss(params))  # Output: [2. 4.]

# Batch of params (e.g., simulating multiple runs on TPU)
batch_params = jnp.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
print("Vectorized gradients:\n", vmap_grad_loss(batch_params))
# Output: [[ 2.  4.]
#          [ 6.  8.]
#          [10. 12.]]
