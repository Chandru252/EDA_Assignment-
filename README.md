# Data Cleaning and Exploratory Data Analysis (EDA)

## Dataset Description

The dataset used in this project contains employee information with the following columns:

* Name
* Age
* Gender
* Salary

The dataset contains **3 rows** and **4 columns**.

---

## Data Loading

The dataset was loaded into a pandas DataFrame using:

```python
pd.read_csv("dataset.csv")
```

The first five rows, data types, and dataset shape were displayed to understand the structure of the data.

---

## Missing Value Analysis

The dataset was checked for missing values using:

```python
df.isnull().sum()
```

### Result

* No missing values were found.
* All columns contain 0% missing values.
* Since there were no missing values, no imputation was required.

### Why Median?

If missing values had existed in numeric columns, the median would have been used because it is less affected by outliers than the mean.

---

## Duplicate Detection

Duplicate rows were checked using:

```python
df.duplicated().sum()
```

### Result

* Duplicate rows found: **0**
* No rows were removed.

---

## Data Type Conversion

The following conversions were performed:

* Age → Numeric
* Gender → Category

Memory usage before conversion:

* 500 bytes

Memory usage after conversion:

* 450 bytes

The memory usage decreased after converting the Gender column to the category data type.

---

## Descriptive Statistics

Summary statistics were generated using:

```python
df.describe()
```

The statistics included:

* Mean
* Standard Deviation
* Minimum
* Maximum
* Quartiles

---

## Skewness Analysis

Skewness was calculated for all numeric columns.

Results:

* Age = -0.586
* Salary = -0.423

The Age column has the highest absolute skewness.

Negative skewness means that the distribution has a longer tail on the left side.

---

## Outlier Detection

The IQR method was used to detect outliers in:

* Age
* Salary

Very few data points were available, so no significant outliers were identified.

Future handling:
If a larger dataset is used, outliers will be retained initially and evaluated before applying capping or transformation.

---

## Visualizations

The following plots were created:

* Line Plot
* Bar Chart
* Histogram
* Scatter Plot
* Box Plot
* Correlation Heatmap

These visualizations helped understand the distribution and relationships among variables.

---

## Correlation Analysis

Pearson correlation was calculated for all numeric columns.

A correlation heatmap was generated to visualize the relationships.

Correlation does not necessarily imply causation because other hidden variables may influence the relationship.

---

## Spearman Correlation

Spearman correlation was also computed and compared with Pearson correlation.

The comparison helps identify monotonic relationships that may not be perfectly linear.

---

## Group Aggregation

The dataset was grouped by the Gender column.

The following statistics were calculated:

* Mean Salary
* Standard Deviation
* Count

These statistics help compare salary distributions across different groups.

---

## Output

The cleaned dataset was saved as:

```
cleaned_data.csv
```

---

## Conclusion

This project demonstrated the complete process of:

* Loading data
* Cleaning data
* Handling missing values
* Removing duplicates
* Correcting data types
* Detecting outliers
* Performing exploratory data analysis
* Creating visualizations
* Computing correlations
* Saving the cleaned dataset

The dataset was successfully prepared for future machine learning and predictive modeling tasks.
