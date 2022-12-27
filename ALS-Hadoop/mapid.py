import os
import csv

data = [
    ("ml-100k", "u.data", None, False),
    ("ml-1m", "ratings.dat", "::", False),
    ("ml-10M100K", "ratings.dat", "::", False),
    ("ml-20m", "ratings.csv", ",", True),
    ("ml-25m", "ratings.csv", ",", True),
]

for dir_name, file_name, delim, skip_first in data:
    print("Processing", dir_name)
    uset = set()
    iset = set()
    records = []
    with open(os.path.join(dir_name, file_name)) as f:
        for _, row in enumerate(f):
            if _ == 0 and skip_first:
                continue
            uid, iid, rating = row.strip().split(delim)[:3]
            uset.add(uid)
            iset.add(iid)
            records.append([uid, iid, rating])

    print(len(uset), "users")
    print(len(iset), "items")
    umap = {k: v for v, k in enumerate(uset)}
    imap = {k: v for v, k in enumerate(iset)}

    out_path = os.path.join("processed", dir_name)
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    with open(os.path.join(out_path, "ratings.dat"), "w") as f:
        csv_writer = csv.writer(f)
        for uid, iid, rating in records:
            csv_writer.writerow([umap[uid], imap[iid], rating])

    with open(os.path.join(out_path, "umap.dat"), "w") as f:
        for k, v in umap.items():
            print(k, v, file=f)

    with open(os.path.join(out_path, "imap.dat"), "w") as f:
        for k, v in imap.items():
            print(k, v, file=f)

