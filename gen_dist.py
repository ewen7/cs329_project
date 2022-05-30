import numpy as np

x = np.random.randint(10, 100, (10,)).astype(float)
x /= x.sum()
x = list(np.round(x * 100))
x[0] += 100 - sum(x)
assert sum(x) == 100
x = [str(y/100) for y in x]
print(' '.join(x))
breakpoint()
