# Interactive Road Safety Dashboard — Quick Start

**CST2213 — Business Intelligence Programming 2: Advanced Concepts — Final Project**
Student: Mohamed Samatar | Dataset: City of Ottawa Open Data, 2020 Tabular Transportation Collision Data

This is a **single self-contained file**. Everything — cleaning, feature
engineering, the predictive model, geographic hotspot clustering, and the
dashboard UI — lives inside `Mohamed_Samatar_CST2213_BI_Final_Project_Submission.py`. There is no `src/` folder, no
`train.py`, and nothing to import, so there's nothing to break by running
it from the wrong folder.

[Interactive Road Safety Dashboard](https://mohamed-final-project-interactive-road-safety-dashboard.streamlit.app/)


<<<<<<< HEAD
=======
 [Interactive Road Safety Dashboard Streamlit App](https://mohamed-final-project-interactive-road-safety-dashboard.streamlit.app/) platform.

>>>>>>> 7dc4f7d2ca8107117af7303df929a46e3c2ba697
## Setup (do this once)

1. Create a folder anywhere on your computer, e.g. `Desktop\RoadSafetyDashboard`.
2. Put `Mohamed_Samatar_CST2213_BI_Final_Project_Submission.py` and `2020_Tabular_Transportation_Collision_Data.csv` in that folder.
3. Open Command Prompt / Terminal and navigate into that folder:
   ```
   cd Desktop\RoadSafetyDashboard
   ```
4. Install the required libraries (only needed once):
   ```
   pip install streamlit pandas numpy scikit-learn plotly joblib
   ```

## Run it (every time)

From inside that same folder:
```
streamlit run Mohamed_Samatar_CST2213_BI_Final_Project_Submission.py
```

A browser tab should open automatically at `http://localhost:8501`. If it
doesn't, copy that URL into your browser manually.

## The three CSV files, and which ones the app needs

| File | Rows | What it's for |
|---|---|---|
| `2020_Tabular_Transportation_Collision_Data.csv` | 10,047 | **Required.** The raw source file. Place this next to `Mohamed_Samatar_CST2213_BI_Final_Project_Submission.py` — the app loads it automatically. |
| `cleaned_ottawa_collision_data.csv` | 10,047 | Optional. The cleaned/engineered output, for your report/instructor to inspect separately, or to upload manually via the sidebar. |
| `dashboard_ready_collision_data_no_unknown.csv` | 9,522 | Optional. Same as cleaned, minus rows with an "Unknown" category value. Also uploadable via the sidebar. |

You do **not** need to place all three files next to `Mohamed_Samatar_CST2213_BI_Final_Project_Submission.py` — only the
raw one. The app can also accept the cleaned or dashboard-ready file if you
upload it manually through the sidebar's "Upload a CSV file" button; it
automatically detects which of the three formats it's looking at and skips
re-cleaning if the file has already been processed.

## If something goes wrong

- **"This file doesn't match a format this app recognizes"** — you
  uploaded something other than one of the three files above, or a file
  with renamed/altered columns. The error message lists exactly which
  columns are missing for each of the three accepted formats.
- **"File not found" error about the CSV** — make sure
  `2020_Tabular_Transportation_Collision_Data.csv` is in the same folder
  as `Mohamed_Samatar_CST2213_BI_Final_Project_Submission.py`, or upload a file manually using the sidebar.
- **SyntaxError mentioning `cd`** — a terminal command got pasted into
  `Mohamed_Samatar_CST2213_BI_Final_Project_Submission.py` by mistake. Re-download `Mohamed_Samatar_CST2213_BI_Final_Project_Submission.py` fresh.
- **"missing ScriptRunContext" warning** — harmless; only appears if the
  app was launched with `python Mohamed_Samatar_CST2213_BI_Final_Project_Submission.py` instead of `streamlit run Mohamed_Samatar_CST2213_BI_Final_Project_Submission.py`.
  Use the second command.

## Dashboard tabs

1. **Overview** — dataset preview, missing-values summary, severity breakdown.
2. **Trend Analysis** — collisions by month, day of week, and hour of day.
3. **Condition Analysis** — injury rate by weather, light, and road
   condition; top collision location types; K-Means geographic hotspot map.
4. **Model Performance** — comparison of Logistic Regression, Random
   Forest, and Gradient Boosting; confusion matrix; feature importance.
   Best model is selected by **Injury-class recall**, since missing an
   at-risk condition matters more than a false alarm for a road-safety tool.
5. **Risk Prediction** — pick a hypothetical scenario (month, hour,
   weather, light, road, traffic control, impact type, location type) and
   get a live predicted injury probability and risk tier.

## Sidebar

- **Data Input** — optional CSV upload (accepts any of the three files above).
- **Filters** — Weather, Light Condition, Road Condition (single-select,
  defaults to "All").
