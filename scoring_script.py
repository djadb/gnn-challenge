import pandas as pd
import sys
from sklearn.metrics import f1_score

submission_path = sys.argv[1]
truth_path = sys.argv[2]
output_path = sys.argv[3]

submission = pd.read_parquet(submission_path)
truth = pd.read_csv(truth_path)

score = f1_score(truth["y"], submission["y_pred"], average="macro")

with open(output_path, "w") as f:
    f.write(f"{score:.6f}")
