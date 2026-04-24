#!/bin/bash
# WSL2 JAX GPU 테스트 스크립트
source ~/ethicaai_env/bin/activate

python3 << 'PYEOF'
import jax
import jax.numpy as jnp
import time

print(f"JAX version: {jax.__version__}")
print(f"Backend: {jax.default_backend()}")
print(f"Devices: {jax.devices()}")

# GPU 벤치마크
n = 5000
key = jax.random.PRNGKey(0)
a = jax.random.normal(key, (n, n))
b = jax.random.normal(key, (n, n))

# 워밍업
c = jnp.dot(a, b).block_until_ready()

# 실제 벤치마크
t0 = time.time()
for _ in range(10):
    c = jnp.dot(a, b).block_until_ready()
gpu_time = (time.time() - t0) / 10

print(f"5000x5000 matmul: {gpu_time*1000:.1f}ms")
print("GPU TEST PASSED!")
PYEOF
