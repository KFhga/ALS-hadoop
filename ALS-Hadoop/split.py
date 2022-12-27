import os
import csv
import random

random.seed(1)
train_percentage = 0.8
probe_percentage = 0.2

data = ["ml-100k", "ml-1m", "ml-10M100K", "ml-20m", "ml-25m"]
for dir_name in data:
    train_lines = []
    probe_lines = []
    with open(os.path.join("processed", dir_name, "ratings.dat")) as f:
        for line in f:
            r = random.random()
            if r <= train_percentage:
                train_lines.append(line.strip())
            elif r > train_percentage and r <= train_percentage + probe_percentage:
                probe_lines.append(line.strip())
    with open(os.path.join("processed", dir_name, "ratings.train"), "w") as f:
        for line in train_lines:
            print(line, file=f)
    with open(os.path.join("processed", dir_name, "ratings.probe"), "w") as f:
        for line in probe_lines:
            print(line, file=f)

