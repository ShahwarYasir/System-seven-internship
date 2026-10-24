"""Creates houses_raw.csv, the practice dataset used from Day 2 onwards.

The data is synthetic (made up with a fixed random seed), so everyone running this
gets the same file. It is intentionally a little messy: missing values, duplicate rows,
inconsistent text and a few impossible areas, so the Day 4 cleaning task has real work to do.
"""
import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
n = 300

area = rng.choice(np.arange(3, 20.5, 0.5), size=n)                    # marla
bedrooms = np.clip(np.round(area / 4 + rng.normal(0, 0.7, n)), 1, 6).astype(int)
bathrooms = np.clip(bedrooms - rng.integers(0, 2, n), 1, 5).astype(int)
age = rng.integers(0, 31, n)                                           # years
zone = rng.choice(["Central", "Suburban", "Outskirts"], size=n, p=[0.25, 0.45, 0.30])
zone_bonus = pd.Series(zone).map({"Central": 4.0, "Suburban": 1.5, "Outskirts": 0.0}).to_numpy()

# price in million PKR: a clear linear signal plus some noise
price = 1.5 + 0.9 * area + 0.7 * bedrooms + 0.4 * bathrooms - 0.08 * age + zone_bonus
price = np.round(np.clip(price + rng.normal(0, 1.5, n), 1.5, None), 2)

df = pd.DataFrame({
    "area_marla": area,
    "bedrooms": bedrooms,
    "bathrooms": bathrooms.astype(float),
    "age_years": age.astype(float),
    "location_zone": zone,
    "price_million_pkr": price,
})

# ---- make it messy on purpose ----
df.loc[rng.choice(n, 15, replace=False), "age_years"] = np.nan
df.loc[rng.choice(n, 10, replace=False), "bathrooms"] = np.nan
df.loc[rng.choice(n, 12, replace=False), "location_zone"] = "central"          # wrong case
df.loc[rng.choice(n, 8, replace=False), "location_zone"] = " Suburban "        # stray spaces
df.loc[rng.choice(n, 3, replace=False), "area_marla"] = 120.0                  # typo-sized areas
df = pd.concat([df, df.sample(5, random_state=1)], ignore_index=True)           # duplicate rows
df = df.sample(frac=1, random_state=7).reset_index(drop=True)

df.to_csv("houses_raw.csv", index=False)
print(df.shape)
