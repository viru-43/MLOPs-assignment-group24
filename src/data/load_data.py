from ucimlrepo import fetch_ucirepo
import pandas as pd

# Fetch dataset
heart = fetch_ucirepo(id=45)

X = heart.data.features
y = heart.data.targets

print("Features shape:", X.shape)
print("Targets shape:", y.shape)

# Convert target DataFrame → Series
y = y.iloc[:, 0]

# Convert multi-class target to binary
y = y.apply(lambda val: 1 if val > 0 else 0)

# Combine features + target
df = pd.concat([X, y.rename("target")], axis=1)

# Save to CSV
df.to_csv("data/raw/heart.csv", index=False)

print("Saved cleaned dataset to data/raw/heart.csv")