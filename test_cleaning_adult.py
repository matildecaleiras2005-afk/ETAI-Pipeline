import yaml
from src.data import load_data
from src.preprocessing import clean_dataset

with open("config_adult.yaml") as f:
    config = yaml.safe_load(f)

train_path = config["data"].get("train_path") or config["data"].get("path")
df = load_data(train_path)
df_clean = clean_dataset(df, config["diagnostics"])

print("--- Validação da Limpeza (Adult) ---")
print("Missing em workclass:", df_clean["workclass"].isna().sum(), "-> Esperado: 379")
print("Missing em occupation:", df_clean["occupation"].isna().sum(), "-> Esperado: 381")
print("Missing em native_country:", df_clean["native_country"].isna().sum(), "-> Esperado: 86")

total_question_marks = (df_clean == "?").sum().sum()
print("Total de '?' residuais:", total_question_marks, "-> Esperado: 0")