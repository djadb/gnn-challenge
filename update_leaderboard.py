import sys
import os
from datetime import datetime, timezone

score_file = sys.argv[1]
leaderboard_file = sys.argv[2]
username = sys.argv[3]

with open(score_file, "r") as f:
    score = float(f.read().strip())

timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

# If leaderboard doesn't exist, create header with Rank
try:
    with open(leaderboard_file, "r", encoding="utf-8") as f:
        lines = f.readlines()
except FileNotFoundError:
    lines = [
        "# 🏆 Leaderboard\n\n",
        "| Rank |          User        |  F1 (macro) |        Timestamp        |\n",
        "|------|----------------------|-------------|-------------------------|\n",
    ]

# Remove old entry from same user (keep best only)
new_lines = []
for line in lines:
    if line.startswith(f"|") and username in line:
        continue
    new_lines.append(line)

# Add new entry (without rank yet)
entry = f"| 0    | {username:<21}|{score:>12.4f} | {timestamp:<24}|\n"
new_lines.append(entry)

# Sort by score (descending)
header = new_lines[:3]
rows = new_lines[3:]

rows_sorted = sorted(
    rows,
    key=lambda x: float(x.split("|")[3]),
    reverse=True
)

# Add ranks
rows_ranked = []
for i, row in enumerate(rows_sorted, start=1):
    parts = row.split("|")
    # Replace rank (first column) with current index
    parts[1] = f" {i:<4} "
    rows_ranked.append("|".join(parts))

# Write back
with open(leaderboard_file, "w", encoding="utf-8") as f:
    f.writelines(header + rows_ranked)
    
os.remove(score_file)

