# ============================================
# src/prepare_dataset.py
# Limpieza, selección de columnas y derivación de variables
# ============================================

import os
import re
import numpy as np
import pandas as pd

# =====================
# Configuración de paths
# =====================
RAW_PATH = "data/raw/flights_jfk_mia.csv"
OUT_DIR = "data/processed"
OUT_PATH = os.path.join(OUT_DIR, "flights_clean.parquet")

os.makedirs(OUT_DIR, exist_ok=True)


# =====================
# Función auxiliar
# =====================
def parse_duration(duration_str):
    """
    Convierte formato ISO 8601 (PT#H#M) a minutos totales.
    Ejemplo: 'PT5H30M' → 330
    """
    if pd.isna(duration_str):
        return np.nan
    match = re.match(r"PT(?:(\d+)H)?(?:(\d+)M)?", str(duration_str))
    if not match:
        return np.nan
    hours = int(match.group(1)) if match.group(1) else 0
    minutes = int(match.group(2)) if match.group(2) else 0
    return hours * 60 + minutes


# =====================
# Proceso principal
# =====================
def main():
    print("📥 Cargando dataset...")
    df = pd.read_csv(RAW_PATH)
    print(f"Dataset original: {df.shape[0]:,} filas, {df.shape[1]} columnas")

    # 1️⃣ Selección de columnas relevantes
    cols = [
        "legId", "searchDate", "flightDate",
        "startingAirport", "destinationAirport",
        "travelDuration", "isBasicEconomy", "isRefundable", "isNonStop",
        "seatsRemaining", "totalTravelDistance",
        "segmentsAirlineName", "segmentsCabinCode",
        "baseFare", "totalFare"
    ]
    df = df[cols].copy()

    # 2️⃣ Conversión de fechas
    df["searchDate"] = pd.to_datetime(df["searchDate"], errors="coerce")
    df["flightDate"] = pd.to_datetime(df["flightDate"], errors="coerce")
    df = df.dropna(subset=["searchDate", "flightDate"])

    df["days_to_departure"] = (df["flightDate"] - df["searchDate"]).dt.days
    df = df[df["days_to_departure"] >= 0]

    # 3️⃣ Conversión de duración
    df["duration_min"] = df["travelDuration"].apply(parse_duration)
    df = df[df["duration_min"].notna() & (df["duration_min"] > 0)]
    df.drop(columns=["travelDuration"], inplace=True)

    # 4️⃣ Variables temporales derivadas
    df["flight_month"] = df["flightDate"].dt.month
    df["flight_dayofweek"] = df["flightDate"].dt.day_name()
    df["is_weekend"] = df["flight_dayofweek"].isin(["Saturday", "Sunday"]).astype(int)

    # 5️⃣ Limpieza categóricas
    df["main_airline"] = df["segmentsAirlineName"].astype(str).str.split("|").str[0].str.strip()
    df["main_cabin"] = df["segmentsCabinCode"].astype(str).str.split("|").str[0].str.strip()
    df.drop(columns=["segmentsAirlineName", "segmentsCabinCode"], inplace=True, errors="ignore")

    # 6️⃣ Imputar distancia por ruta
    df["totalTravelDistance"] = df.groupby(
        ["startingAirport", "destinationAirport"]
    )["totalTravelDistance"].transform(lambda s: s.fillna(s.mean()))

    # 7️⃣ Eliminar nulos restantes y guardar
    df = df.dropna(subset=["totalFare", "totalTravelDistance"])

    os.makedirs(OUT_DIR, exist_ok=True)
    df.to_parquet(OUT_PATH, index=False)

    print("✅ Dataset limpio generado correctamente")
    print(f"Archivo guardado en: {OUT_PATH}")
    print(f"Filas finales: {len(df):,}")


if __name__ == "__main__":
    main()
