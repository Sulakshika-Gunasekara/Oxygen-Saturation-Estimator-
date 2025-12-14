import os
import json
import torch

import final  # imports constants + TinyTransformer from final.py


# --- PATHS (your layout) ---
BASE_DIR = r"D:\demo final"
LOG_DIR = os.path.join(BASE_DIR, final.LOG_DIR)
FRONTEND_DIR = os.path.join(BASE_DIR, "frontend")

MODEL_WEIGHTS = os.path.join(LOG_DIR, "best_run.pt")
FEAT_MU_PATH  = os.path.join(LOG_DIR, "feat_mu.pt")
FEAT_STD_PATH = os.path.join(LOG_DIR, "feat_std.pt")
Y_MU_PATH     = os.path.join(LOG_DIR, "y_mu.pt")
Y_STD_PATH    = os.path.join(LOG_DIR, "y_std.pt")

ONNX_PATH = os.path.join(FRONTEND_DIR, "model.onnx")
NORM_JSON_PATH = os.path.join(FRONTEND_DIR, "norm.json")

# --- LOAD NORMALIZATION STATS ---
feat_mu  = torch.load(FEAT_MU_PATH, map_location="cpu")  # [n_feats]
feat_std = torch.load(FEAT_STD_PATH, map_location="cpu") # [n_feats]
y_mu     = float(torch.load(Y_MU_PATH, map_location="cpu").item())
y_std    = float(torch.load(Y_STD_PATH, map_location="cpu").item())

n_feats = int(feat_mu.numel())
print(f"n_feats = {n_feats}")

# --- BUILD MODEL using final.py config ---
model = final.TinyTransformer(
    n_feats=n_feats,
    d=final.D_MODEL,
    h=final.N_HEADS,
    L=final.N_LAYERS,
    drop=final.DROP,
    drop_path=final.DROP_PATH,
    max_t=final.MAX_T,
)

model.load_state_dict(torch.load(MODEL_WEIGHTS, map_location="cpu"))
model.eval()

# --- DUMMY INPUT ---
T = final.MAX_T
dummy_x = torch.randn(1, T, n_feats, dtype=torch.float32)
dummy_mask = torch.zeros(1, T, dtype=torch.bool)

# --- EXPORT ---
os.makedirs(FRONTEND_DIR, exist_ok=True)

torch.onnx.export(
    model,
    (dummy_x, dummy_mask),
    ONNX_PATH,
    input_names=["input", "mask"],
    output_names=["output"],
    opset_version=17,
    do_constant_folding=True,
    dynamic_axes={
        "input":  {0: "batch", 1: "time"},
        "mask":   {0: "batch", 1: "time"},
        "output": {0: "batch"},
    },
)

print(f"Saved ONNX model to: {ONNX_PATH}")

# --- WRITE norm.json ---
norm_data = {
    "feat_mean": feat_mu.tolist(),
    "feat_std":  feat_std.tolist(),
    "y_mean":    float(y_mu),
    "y_std":     float(y_std),
    "max_t":     int(final.MAX_T),
    "n_feats":   int(n_feats),
}

with open(NORM_JSON_PATH, "w") as f:
    json.dump(norm_data, f, indent=4)

print(f"Saved norm.json to: {NORM_JSON_PATH}")
