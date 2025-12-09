import pandas as pd
from statsmodels.stats.proportion import proportions_ztest
from scipy import stats

# --- Configuration ---
# Set the significance level (alpha). Standard practice is 0.05.
ALPHA = 0.05
# The name of the data file, which MUST be in the same directory as this script.
DATA_FILE_NAME = "RenewalData.txt" 

# --- 1. Load Data (Using Relative Path) ---

print("--- A/B Test Renewal Analysis ---")

try:
    # Use pd.read_csv. It can read .csv, .txt (assuming it's comma-separated), etc.
    df = pd.read_csv(DATA_FILE_NAME)
except FileNotFoundError:
    print(f"FATAL ERROR: The data file was not found at: '{DATA_FILE_NAME}'")
    print("Please ensure your data file is named 'RenewalData.txt' (or update DATA_FILE_NAME) and is in the same folder as this script.")
    exit()

# --- 1.5 Data Quality Checks ---
print("\n## 1.5 Data Quality Checks")

# Check for balanced groups
group_counts = df['Test_Group'].value_counts()
print(f"Group Counts:\n{group_counts.to_markdown(numalign='left', stralign='left')}")

# Add a warning if groups are significantly unbalanced (more than 5% difference)
total_obs = len(df)
if total_obs > 0 and abs(group_counts['A'] - group_counts['B']) / total_obs > 0.05:
    print("⚠️ WARNING: Test groups are significantly unbalanced (more than 5% difference). Results should be interpreted with caution.")

print("Data checks complete. Proceeding to analysis.")


# --- 2. Step 2: Descriptive Statistics ---

print("\n## Step 2: Descriptive Statistics (Observed Data)")

# Group the data and calculate key metrics
grouped_stats = df.groupby('Test_Group').agg(
    Total_Customers=('Customer_ID', 'count'),
    Renewal_Rate=('Renewal_Status', 'mean'),  # Mean of 0/1 is the renewal rate
    Avg_Discounted_ARR=('Discounted_ARR', 'mean'),
    Std_Discounted_ARR=('Discounted_ARR', 'std')
)

# Store raw means for later use in interpretation
raw_renewal_A = grouped_stats.loc['A', 'Renewal_Rate']
raw_renewal_B = grouped_stats.loc['B', 'Renewal_Rate']
raw_arr_A = grouped_stats.loc['A', 'Avg_Discounted_ARR']
raw_arr_B = grouped_stats.loc['B', 'Avg_Discounted_ARR']

# Format the results for clean output
grouped_stats_formatted = grouped_stats.copy()
grouped_stats_formatted['Renewal_Rate'] = (grouped_stats_formatted['Renewal_Rate'] * 100).round(2).astype(str) + '%'
grouped_stats_formatted['Avg_Discounted_ARR'] = grouped_stats_formatted['Avg_Discounted_ARR'].map('${:,.2f}'.format)
grouped_stats_formatted['Std_Discounted_ARR'] = grouped_stats_formatted['Std_Discounted_ARR'].map('${:,.2f}'.format)

print(grouped_stats_formatted.to_markdown(numalign='left', stralign='left'))

# --- 3. Step 3: Z-Test for Renewal Rate (Proportions) ---

print("\n## Step 3: Z-Test for Renewal Rate (Renewal Proportions)")

# Display the description of the formula used for the Two-Sample Z-Test for Proportions.
print("\n--- Z-Test Formula ---")
print("Z-Statistic measures the difference between two sample proportions, divided by the pooled standard error.")

# Extract required counts and display them
renewals_A = df[df['Test_Group'] == 'A']['Renewal_Status'].sum()
nobs_A = df[df['Test_Group'] == 'A']['Renewal_Status'].count()
renewals_B = df[df['Test_Group'] == 'B']['Renewal_Status'].sum()
nobs_B = df[df['Test_Group'] == 'B']['Renewal_Status'].count()

print("--- Test Inputs ---")
print(f"Group A (n={nobs_A}): {renewals_A} Renewals | Proportion (p̂A): {raw_renewal_A:.4f}")
print(f"Group B (n={nobs_B}): {renewals_B} Renewals | Proportion (p̂B): {raw_renewal_B:.4f}")
print(f"Difference in Proportions (p̂A - p̂B): {(raw_renewal_A - raw_renewal_B):.4f}")
print("-------------------")

# Perform the Two-Sample Z-Test
stat_prop, p_value_prop = proportions_ztest(
    count=[renewals_A, renewals_B],
    nobs=[nobs_A, nobs_B]
)

print(f"Z-Statistic: {stat_prop:.4f}")
print(f"P-Value: {p_value_prop:.4f}")

# DYNAMIC CONCLUSION FOR Z-TEST
if p_value_prop < ALPHA:
    print(f"Conclusion: The P-value ({p_value_prop:.4f}) is LESS than alpha ({ALPHA}). We **REJECT** the null hypothesis. The difference in renewal rates is **STATISTICALLY SIGNIFICANT**.")
else:
    print(f"Conclusion: The P-value ({p_value_prop:.4f}) is GREATER than or equal to alpha ({ALPHA}). We **FAIL TO REJECT** the null hypothesis. The difference in renewal rates is **NOT statistically significant**.")


# --- 4. Step 4: T-Test for Average Discounted ARR (Means) ---

print("\n## Step 4: T-Test for Average Discounted ARR (ARR Means)")

# Display the description of the formula used for the T-Test for Means (Welch's T-Test).
print("\n--- T-Test Formula (Welch's) ---")
print("T-Statistic measures the difference between two sample means, divided by the standard error of the difference.")

# Extract ARR data and statistics and display them
arr_A = df[df['Test_Group'] == 'A']['Discounted_ARR']
arr_B = df[df['Test_Group'] == 'B']['Discounted_ARR']

std_A = grouped_stats.loc['A', 'Std_Discounted_ARR']
std_B = grouped_stats.loc['B', 'Std_Discounted_ARR']

print("--- Test Inputs ---")
print(f"Group A (n={nobs_A}): Mean (X̄A) = ${raw_arr_A:.2f} | Std Dev (sA) = ${std_A:.2f}")
print(f"Group B (n={nobs_B}): Mean (X̄B) = ${raw_arr_B:.2f} | Std Dev (sB) = ${std_B:.2f}")
print(f"Difference in Means (X̄A - X̄B): ${(raw_arr_A - raw_arr_B):.2f}")
print("-------------------")

# Perform the Independent Two-Sample T-Test (Welch's T-test is safer)
stat_arr, p_value_arr = stats.ttest_ind(
    arr_A,
    arr_B,
    equal_var=False
)

print(f"T-Statistic: {stat_arr:.4f}")
print(f"P-Value: {p_value_arr:.4f}")

# DYNAMIC CONCLUSION FOR T-TEST
if p_value_arr < ALPHA:
    print(f"Conclusion: The P-value ({p_value_arr:.4f}) is LESS than alpha ({ALPHA}). We **REJECT** the null hypothesis. The difference in average discounted ARR is **STATISTICALLY SIGNIFICANT**.")
else:
    print(f"Conclusion: The P-value ({p_value_arr:.4f}) is GREATER than or equal to alpha ({ALPHA}). We **FAIL TO REJECT** the null hypothesis. The difference in average discounted ARR is **NOT statistically significant**.")


# --- 5. Final Business Interpretation (Now referencing key values) ---
print("\n## Step 5: Final Business Interpretation")

renewal_is_significant = p_value_prop < ALPHA
arr_is_significant = p_value_arr < ALPHA

# Create formatted strings for inclusion in the final print statement
format_renewal_A = f"{(raw_renewal_A * 100):.2f}%"
format_renewal_B = f"{(raw_renewal_B * 100):.2f}%"
format_arr_A = f"${raw_arr_A:,.2f}"
format_arr_B = f"${raw_arr_B:,.2f}"

# Synthesize the final recommendation
if renewal_is_significant and arr_is_significant:
    if raw_renewal_A > raw_renewal_B and raw_arr_A > raw_arr_B:
        print(f"FINAL RECOMMENDATION: Test Group A (Lower Discount) is the clear winner across both key metrics.")
        print(f"  > Renewal Rate: {format_renewal_A} vs {format_renewal_B} (Significant)")
        print(f"  > Avg ARR: {format_arr_A} vs {format_arr_B} (Significant)")
        print("ACTION: The lower discount strategy should be implemented.")
    elif raw_renewal_B > raw_renewal_A and raw_arr_B > raw_arr_A:
        print(f"FINAL RECOMMENDATION: Test Group B (Higher Discount) is the clear winner across both key metrics.")
        print(f"  > Renewal Rate: {format_renewal_B} vs {format_renewal_A} (Significant)")
        print(f"  > Avg ARR: {format_arr_B} vs {format_arr_A} (Significant)")
        print("ACTION: The higher discount strategy should be implemented.")
    else:
        print("FINAL RECOMMENDATION: The results are mixed (conflicting winners).")
        print(f"  > Renewal Rate Comparison: Group A ({format_renewal_A}) vs Group B ({format_renewal_B})")
        print(f"  > Avg ARR Comparison: Group A ({format_arr_A}) vs Group B ({format_arr_B})")
        print("ACTION: Business stakeholders need to decide which metric (renewal volume or revenue per customer) is currently more important.")
elif renewal_is_significant or arr_is_significant:
    print("FINAL RECOMMENDATION: Only one key metric showed a significant difference.")
    
    if renewal_is_significant:
        winner = 'A' if raw_renewal_A > raw_renewal_B else 'B'
        print(f"  > Renewal Rate is significant. Winner: Group {winner} ({format_renewal_A} vs {format_renewal_B})")
    else:
        print(f"  > Renewal Rate is NOT significant. (Group A: {format_renewal_A}, Group B: {format_renewal_B})")

    if arr_is_significant:
        winner = 'A' if raw_arr_A > raw_arr_B else 'B'
        print(f"  > Avg ARR is significant. Winner: Group {winner} ({format_arr_A} vs {format_arr_B})")
    else:
        print(f"  > Avg ARR is NOT significant. (Group A: {format_arr_A}, Group B: {format_arr_B})")

    print("ACTION: Further data collection or segmentation analysis is recommended to justify a full strategy change.")
else:
    print("FINAL RECOMMENDATION: Neither key metric showed a statistically significant difference.")
    print(f"  > Renewal Rate: {format_renewal_A} vs {format_renewal_B}")
    print(f"  > Avg ARR: {format_arr_A} vs {format_arr_B}")
    print("ACTION: It is recommended to stick with the current default strategy (Group A) or explore a completely new test (Group C).")