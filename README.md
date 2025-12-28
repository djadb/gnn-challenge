# GNN Challenge

The AIDS dataset is a benchmark molecular graph dataset commonly used in graph learning and cheminformatics. Each sample represents a molecule as a graph, where nodes correspond to atoms and edges correspond to chemical bonds.
The goal of the challenge is to predict whether a molecule is **active against HIV** (Label = 1) or **inactive** (Label = 0).

---

## Dataset

* The training set consists of **1,600 graphs**, while the test set consists of **400 graphs**.
* Both sets are **imbalanced**, with inactive compounds representing approximately **25%** of the data.
* Each node represents an atom and is encoded using a **38-dimensional one-hot vector** indicating the atom’s chemical element.
* Each edge is encoded using a **one-hot vector representing the bond type** (3 bond types in total).

---

## Evaluation

The evaluation metric is the **macro F1-score**, which computes the F1-score independently for each class and averages them, making it suitable for imbalanced datasets.

---

## Submission

* An example submission file is provided in the `submissions/` folder.
* Submissions must be in **`.parquet` format** and contain two columns:

  * `graph_id`
  * `y_pred` (integer labels: 0 or 1)
* The submission file **must be named `submission.parquet`** to be considered for the leaderboard.
* Participants must fork the repository and place their submission file in the `submissions/` folder.
* Once ready, open a **pull request** to the main repository.
* **Only one submission per pull request** is allowed.

---

## Rules

* External data is **not allowed**.
* The full labeled dataset is **available online**, and any attempt to access it is strictly prohibited.
* Submissions are **unlimited**.

---

## Leaderboard

* The leaderboard is available in `leaderboard.md`.
* Once a pull request is submitted, the submission is evaluated automatically and added to the leaderboard within a few minutes.
* **Only the most recent submission per user is kept** on the leaderboard.
