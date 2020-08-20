import jax
import jax.numpy as np
print("jax version {}".format(jax.__version__))
from jax.lib import xla_bridge
print("jax backend {}".format(xla_bridge.get_backend().platform))


from jax import random
key = random.PRNGKey(0)
x = random.normal(key, (5,5))
print(x)
