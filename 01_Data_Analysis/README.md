# Phase 1: Data Analysis (EDA)

## Objective
Load all 9 raw CSV files, merge them into unified patron and restaurant DataFrames, and perform exploratory data analysis to understand the structure, distributions, and relationships in the data.

## Key Script
- `analysis_eda.py` — Full EDA script: loads CSVs, merges dataframes, profiles data, visualizes distributions for numerical and categorical columns, generates ydata-profiling HTML reports.

## Inputs
- `00_Raw_Data/*.csv` — 9 raw source files (ratings, user profiles, restaurant attributes)

## Outputs
- `final_data.pkl` — Cleaned, merged DataFrame saved to disk
- `Reports/EDA_Reports/eda_report.html` — Full automated profiling report
- `Reports/EDA_Reports/patrons_report.html` — Patron-specific profile report

## Key Questions Addressed
- What causes a restaurant to be highly rated?
- What is the distribution of ratings, food ratings, and service ratings?
- Which user and restaurant attributes have the most variation?

## Data Merges Performed
| Join | Key |
|------|-----|
| userprofile + userpayment | userID |
| + usercuisine | userID |
| + rating_final | userID |
| geoplaces2 + chefmozaccepts | placeID |
| + chefmozparking | placeID |
| + chefmozcuisine | placeID |
| + chefmozhours4 | placeID |
| patrons + restaurants | placeID |
