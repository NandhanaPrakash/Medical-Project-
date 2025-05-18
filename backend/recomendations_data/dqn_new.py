import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA

# Load the dataset
df = pd.read_csv("final_ehr_dataset_cleaned.csv")

# --- Step 1: Drop duplicates ---
df = df.drop_duplicates()

# --- Step 2: Average Day 1 and Day 2 nutrient intake ---
day1_cols = [col for col in df.columns if col.startswith("DR1T")]
day2_cols = [col for col in df.columns if col.startswith("DR2T")]

common_nutrients = [col[5:] for col in day1_cols if f'DR2T{col[5:]}' in day2_cols]
for nutrient in common_nutrients:
    df[f'{nutrient}_AVG'] = df[f'DR1T{nutrient}'].fillna(0).add(
                            df[f'DR2T{nutrient}'].fillna(0)) / 2

df.drop(columns=day1_cols + day2_cols, inplace=True)

# --- Step 3: Map Activity_Level to Calories_Burned ---
activity_map = {
    "Sedentary": 1500,
    "Moderate": 2000,
    "Active": 2500
}
df["Calories_Burned"] = df["Activity_Level"].map(activity_map)
df.drop(columns=["Activity_Level"], inplace=True)

# --- Step 4: Encode Takes_Supplements (Yes/No) ---
if df["Takes_Supplements"].dtype == object:
    df["Takes_Supplements"] = df["Takes_Supplements"].map({"Yes": 1, "No": 0})
else:
    df["Takes_Supplements"] = df["Takes_Supplements"].astype(int)

# --- Step 5: Encode Supplement_Name if present ---
if "Supplement_Name" in df.columns:
    df["Supplement_Name"] = df["Supplement_Name"].fillna("None")
    le = LabelEncoder()
    df["Supplement_Name"] = le.fit_transform(df["Supplement_Name"])

# --- Step 6: Drop high-null and textual columns ---
df = df.dropna(thresh=len(df)*0.5, axis=1)
for col in df.select_dtypes(include=["object"]).columns:
    df.drop(columns=[col], inplace=True)

# --- Step 7: Fill remaining NaNs with column mean ---
df.fillna(df.mean(numeric_only=True), inplace=True)

# --- Step 8: Standard Scaling ---
numeric_cols = df.select_dtypes(include=[np.number]).columns
scaler = StandardScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(df[numeric_cols]), columns=numeric_cols)

# --- Step 9: PCA (retain 95% variance) ---
pca = PCA(n_components=0.95, random_state=42)
df_pca = pd.DataFrame(pca.fit_transform(df_scaled))

# --- Step 10: Save cleaned and PCA-transformed dataset ---
df_scaled.to_csv("final_ehr_dataset_scaled.csv", index=False)
df_pca.to_csv("final_ehr_dataset_pca.csv", index=False)

print("✅ Done:")
print("- Scaled data saved as: final_ehr_dataset_scaled.csv")
print("- PCA data saved as: final_ehr_dataset_pca.csv")
