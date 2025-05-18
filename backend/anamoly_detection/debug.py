import pandas as pd

df = pd.read_excel("anamoly.xlsx",header=2)
df.columns = df.columns.str.strip().str.replace(r'\s+', '_', regex=True).str.replace(r'[^\w\s]', '')

print("🔍 Cleaned Column Names:")
print(df.columns.tolist())
