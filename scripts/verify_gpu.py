#!/usr/bin/env python3
"""JAX GPU 검증 스크립트"""
import jax
import jax.numpy as jnp

print("JAX version:", jax.__version__)
print("Backend:", jax.default_backend())
print("Devices:", jax.devices())

# GPU 연산 테스트
x = jnp.ones((1000, 1000))
y = jnp.dot(x, x)
print("GPU Compute OK:", y.shape, "sum:", float(y.sum()))
print("GPU READY!" if jax.default_backend() != "cpu" else "CPU ONLY")
