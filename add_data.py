# add_data.py - safely appends a batch of new cars to cars.csv
# Usage: python add_data.py data_batches/batch2_new_cars.csv

import sys
import pandas as pd

batch_file = sys.argv[1]
main = pd.read_csv("cars.csv")
batch = pd.read_csv(batch_file)

if list(batch.columns) != list(main.columns):
    sys.exit(f"Columns do not match! Expected {list(main.columns)}")

combined = pd.concat([main, batch], ignore_index=True)
combined.to_csv("cars.csv", index=False)
print(f"cars.csv: {len(main)} rows -> {len(combined)} rows (added {len(batch)} from {batch_file})")
