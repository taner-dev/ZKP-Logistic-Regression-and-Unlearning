import numpy as np
import pandas as pd
from logreg import Gradientenabstieg

print("SISA Unlearning Test\n==================")

data = pd.read_csv(r"data\data.csv")
X = data[["texture_se","area_worst"]].values
data["diagnosis"] = data["diagnosis"].map({"M":1,"B":0})
y = data["diagnosis"].values
X = (X - X.mean(axis=0))/X.std(axis=0)

NUM_SHARDS=2
NUM_SLICES=2
indices=np.random.permutation(len(X))
shards=np.array_split(indices, NUM_SHARDS)

models = []
for shard_id, shard_idx in enumerate(shards):
    slices =np.array_split(shard_idx, NUM_SLICES)
    w = np.zeros(2)
    b = 0
    for slice_id, slice_idx in enumerate(slices):
        w, b =Gradientenabstieg(X[slice_idx], y[slice_idx])
        models.append((shard_id, slice_id, w.copy(), b))

print("Vor Unlearning:")
for m in models:
    print(f"Shard {m[0]} Slice {m[1]}")



# SISA-Unlearning


UNLEARN_SHARD = 0
UNLEARN_SLICE = 1

models_after = []

for shard_id, shard_idx in enumerate(shards):
    slices = np.array_split(shard_idx, NUM_SLICES)

    # Nicht betroffener Shard muss alte Modelle behalten
    if shard_id != UNLEARN_SHARD:
        for m in models:
            if m[0] == shard_id:
                models_after.append(m)
        continue

    # Betroffener Shard muss neu trainieren 
    print(f"\nRetraining Shard {shard_id} ab Slice {UNLEARN_SLICE}...")

    w = np.zeros(2)
    b = 0

    for slice_id, slice_idx in enumerate(slices):

        w, b = Gradientenabstieg(X[slice_idx], y[slice_idx])
        models_after.append((shard_id, slice_id, w.copy(), b))
models = models_after

print("\nNach Unlearning:")
for m in models_after:
    print(f"Shard {m[0]} Slice {m[1]}")

print("\n Unlearning erfolgreich")
