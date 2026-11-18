"""Süre tahmin modelinin özellik (feature) üretimi.

Notebook'taki build_features ile birebir aynıdır; hem eğitimde hem de
AI servisinde aynı fonksiyonun kullanılması gerekir.
"""
import numpy as np
import pandas as pd

EARTH_RADIUS_KM = 6371.0


def haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(a))


def manhattan_km(lat1, lon1, lat2, lon2):
    return haversine_km(lat1, lon1, lat1, lon2) + haversine_km(lat1, lon1, lat2, lon1)


def bearing_deg(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    dlon = lon2 - lon1
    y = np.sin(dlon) * np.cos(lat2)
    x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
    return np.degrees(np.arctan2(y, x))


def build_features(d: pd.DataFrame, kmeans) -> pd.DataFrame:
    out = pd.DataFrame(index=d.index)
    plat, plon = d["pickup_latitude"].values, d["pickup_longitude"].values
    dlat, dlon = d["dropoff_latitude"].values, d["dropoff_longitude"].values
    out["pickup_latitude"], out["pickup_longitude"] = plat, plon
    out["dropoff_latitude"], out["dropoff_longitude"] = dlat, dlon
    out["haversine_km"] = haversine_km(plat, plon, dlat, dlon)
    out["manhattan_km"] = manhattan_km(plat, plon, dlat, dlon)
    out["bearing"] = bearing_deg(plat, plon, dlat, dlon)

    t = d["pickup_datetime"]
    out["hour"] = t.dt.hour
    out["minute_of_day"] = t.dt.hour * 60 + t.dt.minute
    out["dow"] = t.dt.day_of_week          # 0 = Pazartesi
    out["month"] = t.dt.month
    out["is_weekend"] = (out["dow"] >= 5).astype(int)
    out["is_rush_hour"] = (out["hour"].isin([7, 8, 9, 16, 17, 18, 19]) & (out["is_weekend"] == 0)).astype(int)

    out["pickup_cluster"] = kmeans.predict(np.column_stack([plat, plon]))
    out["dropoff_cluster"] = kmeans.predict(np.column_stack([dlat, dlon]))
    out["vendor_id"] = d["vendor_id"].values
    out["passenger_count"] = d["passenger_count"].values
    return out
