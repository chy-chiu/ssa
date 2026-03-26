
# %%
import random
import string
import numpy as np

n_items = 50

def generate_ground_truth(seed: int = None):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    items = []
    while len(items) < n_items:
        item = "".join(random.choices(string.ascii_uppercase, k=3))
        if item not in items:
            items.append(item)
    return items

gt = generate_ground_truth(42)
q = [f"{gt[i]} > {gt[i+1]}" for i in range(n_items-1)]
np.random.shuffle(q)

print(gt)
for r in q:
    print(r)
# %%
# %%
gt[ix + np.random.randint(1, n_items-ix)]
# %%


NRE, QIC, QKW, GRP, FGN, JOF, JNN, ETH, ERT, ASA

