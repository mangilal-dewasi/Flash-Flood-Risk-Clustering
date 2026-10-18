
import pandas as pd

path = "data/processed/feature_correlation.csv"
corr = pd.read_csv(path, index_col=0)

threshold = 0.80
pairs = []

for i in range(len(corr.columns)):
    for j in range(i + 1, len(corr.columns)):
        value = corr.iloc[i, j]
        if abs(value) >= threshold:
            pairs.append(
                (corr.columns[i], corr.columns[j], round(value, 3))
            )

print("\nHIGH-CORRELATION FEATURE PAIRS")
print("=" * 55)

if pairs:
    for feature_a, feature_b, value in sorted(
        pairs, key=lambda x: abs(x[2]), reverse=True
    ):
        print(f"{feature_a}  <->  {feature_b}: {value}")
else:
    print("No feature pairs reached the 0.80 threshold.")
