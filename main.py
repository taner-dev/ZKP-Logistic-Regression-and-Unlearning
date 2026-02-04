import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json, os, subprocess
from pathlib import Path
from logreg import Gradientenabstieg, predict


# Daten auslesen

data = pd.read_csv(r"data\data.csv")

X = data[["texture_se", "area_worst"]].values
data["diagnosis"] = data["diagnosis"].map({"M": 1, "B": 0})
y = data["diagnosis"].values

X = (X - X.mean(axis=0)) / X.std(axis=0)


# SISA start

NUM_SHARDS = 2
NUM_SLICES = 2

indices = np.random.permutation(len(X))
shards = np.array_split(indices, NUM_SHARDS)

models = []


# SISA Training

for shard_id, shard_idx in enumerate(shards):
    slices = np.array_split(shard_idx, NUM_SLICES)

    w = np.zeros(2)
    b = 0

    for slice_id, slice_idx in enumerate(slices):
        w, b = Gradientenabstieg(X[slice_idx], y[slice_idx])
        models.append((shard_id, slice_id, w.copy(), b))


# Modell auswählen (Nach unlearning)

models = [m for m in models if not (m[0] == 0 and m[1] >= 1)]

w, b = models[-1][2], models[-1][3]


# Die Erstellung vom Graphen

plt.figure(figsize=(7,5))
plt.scatter(X[y==0,0], X[y==0,1], s=10, label="B")
plt.scatter(X[y==1,0], X[y==1,1], s=10, label="M")

x_vals = np.linspace(X[:,0].min(), X[:,0].max(), 100)
y_vals = -(w[0]*x_vals + b) / w[1]
plt.plot(x_vals, y_vals, "k--")

plt.xlabel("texture_se")
plt.ylabel("area_worst")
plt.legend()
plt.title("Logistische Regression mit SISA für Brustkrebs")
plt.show()


# ZKP Input

n = 3           # Anzahl der erwarteten Merkmale vom circuit
S = 10**6       # Skalar wegen Foalting Point

# 
x_test = np.append(X[0], 0)  # add a dummy 0 feature
w_n = np.append(w, 0)        # add a dummy 0 weight
n = 3

b_val = b


x_scaled = (x_test*S).astype(int)
w_scaled = (w_n*S).astype(int)
b_scaled = int(b_val*S)
z_scaled = b_scaled + sum(x_scaled[i]*w_scaled[i] for i in range(n))

# predicted class (0 or 1)
y_pred = int((b_val + np.dot(w_n, x_test)) >= 0) 


# ZKP Input generieren


# Baut Json
input_data = {
    "x": x_scaled.tolist(),
    "w": w_scaled.tolist(),
    "b": b_scaled,
    "y": y_pred,
    "z": int(z_scaled)
}

# Existiert der Ordner?
os.makedirs("zkp", exist_ok=True)

# Json schreiben
with open("zkp/input.json", "w") as f:
    json.dump(input_data, f, indent=2)

print("input.json generated successfully!")


print("input.json generated successfully!")
os.makedirs("zkp", exist_ok=True)
with open("zkp/input.json", "w") as f:
    json.dump(input_data, f, indent=2)


# ZKP Beweis

os.chdir("circom")
subprocess.run(
    ["circom", "logreg.circom", "--r1cs", "--wasm"],
    check=True,
    cwd="."
)



# Pfade


ROOT = Path(__file__).parent

CIRCOM = ROOT / "tools" / "circom.exe"
SNARKJS = ROOT / "node_modules" / ".bin" / "snarkjs.cmd"
PTAU = ROOT / "powersOfTau28_hez_final_08.ptau"

CIRCUIT = ROOT / "circom" / "logreg.circom"
INPUT = ROOT / "zkp" / "input.json"


# 1. Circuit kompilieren

subprocess.run(
    [
       str(SNARKJS), "groth16", "setup",
       "logreg.r1cs",
       str(PTAU),
       "logreg.zkey"
    ],
    check=True
)


# 2. Groth16 setup




subprocess.run(
    [
       str(SNARKJS), "zkey", "export", "verificationkey",
       "logreg.zkey",
       "verification_key.json"
    ],
    check=True
)


# 3. Witness generieren


subprocess.run(
    [
       "node",
       "logreg_js/generate_witness.js",
       "logreg_js/logreg.wasm",
       str(INPUT),
       "witness.wtns"
    ],
    check=True
)


# 4. Beweis generieren


subprocess.run(
    [
       str(SNARKJS), "groth16", "prove",
       "logreg.zkey",
       "witness.wtns",
       "proof.json",
       "public.json"
    ],
    check=True
)


# 5. Beweis verifizieren


subprocess.run(
    [
       str(SNARKJS), "groth16", "verify",
       "verification_key.json",
       "public.json",
       "proof.json"
    ],
    check=True
)

print("\n ZKP inference proof successfully verified!")

