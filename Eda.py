import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ==================================================
# CREATE FOLDERS
# ==================================================
os.makedirs("images", exist_ok=True)
os.makedirs("reports", exist_ok=True)

# ==================================================
# LOAD DATASET
# ==================================================
try:
    df = pd.read_csv("data/dataset.csv")
    print("Dataset loaded successfully.")
except FileNotFoundError:
    print("ERROR: data/dataset.csv not found.")
    exit()

# ==================================================
# BASIC INFORMATION
# ==================================================
print("\n" + "="*60)
print("DATASET INFORMATION")
print("="*60)

print("\nFirst 5 Rows:")
print(df.head())

print("\nShape:")
print(df.shape)

print("\nColumns:")
print(df.columns.tolist())

print("\nData Types:")
print(df.dtypes)

# ==================================================
# MISSING VALUES
# ==================================================
print("\nMissing Values:")
print(df.isnull().sum())

plt.figure(figsize=(10, 5))
sns.heatmap(df.isnull(), cbar=False)
plt.title("Missing Values Heatmap")
plt.tight_layout()
plt.savefig("images/missing_values.png")
plt.close()

# ==================================================
# DUPLICATES
# ==================================================
duplicates = df.duplicated().sum()

print("\nDuplicate Rows:", duplicates)

if duplicates > 0:
    df = df.drop_duplicates()

# ==================================================
# STATISTICAL SUMMARY
# ==================================================
print("\nStatistical Summary")
print(df.describe())

# ==================================================
# NUMERICAL AND CATEGORICAL COLUMNS
# ==================================================
numerical_cols = df.select_dtypes(include=np.number).columns
categorical_cols = df.select_dtypes(include="object").columns

# ==================================================
# HISTOGRAMS
# ==================================================
for col in numerical_cols:

    plt.figure(figsize=(8, 5))

    plt.hist(
        df[col].dropna(),
        bins=10
    )

    plt.title(f"Histogram - {col}")

    plt.tight_layout()

    plt.savefig(
        f"images/hist_{col}.png"
    )

    plt.close()

# ==================================================
# BOXPLOTS
# ==================================================
for col in numerical_cols:

    plt.figure(figsize=(8, 5))

    plt.boxplot(
        df[col].dropna()
    )

    plt.title(
        f"Boxplot - {col}"
    )

    plt.tight_layout()

    plt.savefig(
        f"images/box_{col}.png"
    )

    plt.close()

# ==================================================
# BAR CHARTS FOR CATEGORICAL DATA
# ==================================================
for col in categorical_cols:

    plt.figure(figsize=(8, 5))

    df[col].value_counts().plot(
        kind="bar"
    )

    plt.title(
        f"Bar Chart - {col}"
    )

    plt.tight_layout()

    plt.savefig(
        f"images/bar_{col}.png"
    )

    plt.close()

# ==================================================
# CORRELATION HEATMAP
# ==================================================
if len(numerical_cols) > 1:

    corr = df[numerical_cols].corr()

    print("\nCorrelation Matrix:")
    print(corr)

    plt.figure(figsize=(10, 6))

    sns.heatmap(
        corr,
        annot=True,
        cmap="coolwarm"
    )

    plt.title("Correlation Heatmap")

    plt.tight_layout()

    plt.savefig(
        "images/correlation_heatmap.png"
    )

    plt.close()

# ==================================================
# PAIRPLOT
# ==================================================
if len(numerical_cols) > 1:

    pair = sns.pairplot(
        df[numerical_cols]
    )

    pair.savefig(
        "images/pairplot.png"
    )

    plt.close("all")

# ==================================================
# OUTLIERS
# ==================================================
print("\nOutlier Analysis")

outlier_results = []

for col in numerical_cols:

    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)

    iqr = q3 - q1

    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr

    count = len(
        df[
            (df[col] < lower) |
            (df[col] > upper)
        ]
    )

    outlier_results.append(
        f"{col}: {count} outliers"
    )

    print(
        f"{col}: {count} outliers"
    )

# ==================================================
# STRONG CORRELATIONS
# ==================================================
strong_corr = []

if len(numerical_cols) > 1:

    corr = df[numerical_cols].corr()

    for i in range(len(corr.columns)):
        for j in range(i + 1, len(corr.columns)):

            value = corr.iloc[i, j]

            if abs(value) >= 0.70:

                strong_corr.append(
                    f"{corr.columns[i]} <-> {corr.columns[j]} = {value:.2f}"
                )

# ==================================================
# REPORT
# ==================================================
with open(
    "reports/EDA_Report.txt",
    "w",
    encoding="utf-8"
) as report:

    report.write("EXPLORATORY DATA ANALYSIS REPORT\n")
    report.write("="*60 + "\n\n")

    report.write(f"Rows: {df.shape[0]}\n")
    report.write(f"Columns: {df.shape[1]}\n\n")

    report.write("MISSING VALUES\n")
    report.write(str(df.isnull().sum()))
    report.write("\n\n")

    report.write("STATISTICAL SUMMARY\n")
    report.write(str(df.describe()))
    report.write("\n\n")

    report.write("OUTLIER ANALYSIS\n")
    for item in outlier_results:
        report.write(item + "\n")

    report.write("\n")

    report.write("STRONG CORRELATIONS\n")

    if len(strong_corr) == 0:
        report.write("No strong correlations found.\n")
    else:
        for item in strong_corr:
            report.write(item + "\n")

print("\n" + "="*60)
print("EDA COMPLETED SUCCESSFULLY")
print("="*60)

print("\nGenerated:")
print("images/missing_values.png")
print("images/correlation_heatmap.png")
print("images/pairplot.png")
print("reports/EDA_Report.txt")