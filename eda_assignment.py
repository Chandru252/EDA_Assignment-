import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# Load dataset
df = pd.read_csv("dataset.csv")     # Change filename if needed

print("First 5 Rows")
print(df.head())

print("\nData Types")
print(df.dtypes)

print("\nShape")
print(df.shape)

print("\nMissing Value Count")
print(df.isnull().sum())

null_percentage = (df.isnull().sum()/df.shape[0])*100

print("\nMissing Percentage")
print(null_percentage)

print("\nColumns Above 20% Missing")
print(null_percentage[null_percentage>20])

# Select only numeric columns
numeric_cols = df.select_dtypes(include="number").columns

# Fill missing values with median
for col in numeric_cols:
    if null_percentage[col] < 20:
        df[col] = df[col].fillna(df[col].median())

duplicates = df.duplicated().sum()

print("Duplicate Rows =", duplicates)

old_rows = df.shape[0]

df = df.drop_duplicates()

new_rows = df.shape[0]
print("Rows Removed =", old_rows-new_rows)

print("\nNew Missing Percentage")

print((df.isnull().sum()/df.shape[0])*100)


print("\nMemory Before")

before = df.memory_usage(deep=True).sum()

print(before)

# Example
# Convert numeric stored as object

df["Age"] = pd.to_numeric(df["Age"], errors="coerce")

# Convert repetitive string column

df["Gender"] = df["Gender"].astype("category")

after = df.memory_usage(deep=True).sum()

print("\nMemory After")

print(after)

print(df.describe())

numeric_cols = df.select_dtypes(include=np.number).columns

skewness = {}

for col in numeric_cols:
    skewness[col] = df[col].skew()

skew_df = pd.DataFrame(skewness.items(), columns=["Column","Skewness"])

print(skew_df)

highest_skew = skew_df.iloc[
    skew_df["Skewness"].abs().idxmax()
]

print("\nHighest Absolute Skew")

print(highest_skew)

columns = ["Age","Salary"]

for col in columns:

    Q1 = df[col].quantile(0.25)

    Q3 = df[col].quantile(0.75)

    IQR = Q3-Q1

    lower = Q1-1.5*IQR

    upper = Q3+1.5*IQR

    outliers = df[(df[col]<lower)|(df[col]>upper)]

    print(col)

    print("Q1 =",Q1)

    print("Q3 =",Q3)

    print("IQR =",IQR)

    print("Outliers =",len(outliers))

    print("------------------")

    plt.figure(figsize=(8,5))

plt.plot(df["Age"])

plt.title("Age Line Plot")

plt.xlabel("Index")

plt.ylabel("Age")

plt.show()


df.groupby("Gender")["Salary"].mean().plot.bar()

plt.title("Average Salary by Gender")

plt.xlabel("Gender")

plt.ylabel("Mean Salary")

plt.show()

sns.histplot(df["Salary"], bins=20)

plt.title("Salary Distribution")

plt.show()

sns.scatterplot(data=df,
                x="Age",
                y="Salary")

plt.title("Age vs Salary")

plt.show()

sns.boxplot(data=df,
            x="Gender",
            y="Salary")

plt.title("Salary by Gender")

plt.show()

corr = df.corr(numeric_only=True)

print(corr)

plt.figure(figsize=(10,8))

sns.heatmap(corr,
            annot=True,
            cmap="coolwarm")

plt.title("Correlation Heatmap")

plt.show()

corr_abs = corr.abs().copy()

# Make a writable copy
corr_abs_values = corr_abs.to_numpy().copy()

np.fill_diagonal(corr_abs_values, 0)

corr_abs = pd.DataFrame(
    corr_abs_values,
    index=corr_abs.index,
    columns=corr_abs.columns
)

highest = corr_abs.unstack().idxmax()

print("Highest Correlation Pair =", highest)
skew_sorted = skew_df.reindex(
    skew_df["Skewness"].abs().sort_values(ascending=False).index
)

top2 = skew_sorted["Column"].head(2)

for col in top2:

    print("\n",col)

    print("Mean =",df[col].mean())

    print("Median =",df[col].median())

    df[col] = df[col].fillna(df[col].median())

print(df[top2].isnull().sum())

pearson = df.corr(numeric_only=True)

spearman = df.corr(method="spearman",numeric_only=True)

print("\nPearson")

print(pearson)

print("\nSpearman")

print(spearman)

difference = (spearman-pearson).abs()

print("\nDifference")

print(difference)

diff = difference.unstack().reset_index()

diff.columns=["Column1","Column2","Difference"]

diff = diff[diff["Column1"]<diff["Column2"]]

print(diff.sort_values("Difference",ascending=False).head(3))

group = df.groupby("Gender")["Salary"].agg(["mean","std","count"])

print(group)

highest_mean = group["mean"].idxmax()

highest_std = group["std"].idxmax()

print("Highest Mean Group =",highest_mean)

print("Highest Std Group =",highest_std)

ratio = group["mean"].max()/group["mean"].min()

print("Mean Ratio =",ratio)

df.to_csv("cleaned_data.csv",index=False)

print("Dataset Saved Successfully")