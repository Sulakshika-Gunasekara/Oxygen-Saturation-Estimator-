import os
import glob
import json
import numpy as np

# === adjust these if your data lives elsewhere ===
VV_DIR   = r"D:/vv/npz"      # same as VV_DEFAULT_DIR in final.py
PURE_DIR = r"D:/pure/npz4"   # same as PURE_DEFAULT_DIR in final.py

BASE_DIR     = r"D:\demo final"
FRONTEND_DIR = os.path.join(BASE_DIR, "frontend")
OUT_JSON     = os.path.join(FRONTEND_DIR, "demo_samples.json")

os.makedirs(FRONTEND_DIR, exist_ok=True)

def collect_samples(root, pattern, dataset_name,
                    max_files=3, max_seqs_per_file=3):
    samples = []
    paths = sorted(glob.glob(os.path.join(root, pattern)))
    for path in paths[:max_files]:
        fname = os.path.basename(path)
        data = np.load(path)
        X = data["features"]   # [N, T, F]
        y = data["labels"]     # [N]
        N, T, F = X.shape
        use_n = min(N, max_seqs_per_file)

        for i in range(use_n):
            samples.append(
                {
                    "id": f"{dataset_name}:{fname}:{i}",
                    "dataset": dataset_name,
                    "file": fname,
                    "index": int(i),
                    "label": float(y[i]),
                    "features": X[i].astype(float).tolist(),  # [T, F]
                }
            )
    return samples

def main():
    all_samples = []
    # VV data
    if os.path.isdir(VV_DIR):
        all_samples.extend(
            collect_samples(VV_DIR, "*.npz", "VV", max_files=3, max_seqs_per_file=3)
        )
    # PURE / test data
    if os.path.isdir(PURE_DIR):
        all_samples.extend(
            collect_samples(PURE_DIR, "*_sequences.npz", "PURE",
                            max_files=3, max_seqs_per_file=3)
        )

    out_obj = {"samples": all_samples}
    with open(OUT_JSON, "w") as f:
        json.dump(out_obj, f)

    print(f"Wrote {OUT_JSON} with {len(all_samples)} samples.")

if __name__ == "__main__":
    main()
