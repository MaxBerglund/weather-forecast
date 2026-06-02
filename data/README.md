---
license: cc-by-4.0
language:
  - en
pretty_name: SMHI Swedish Temperature Stations (Hourly)
tags:
  - time-series
  - forecasting
  - weather
  - temperature
  - sweden
  - smhi
  - benchmark
task_categories:
  - time-series-forecasting
size_categories:
  - 10K<n<100K
configs:
  - config_name: stations
    data_files: swedish-temperatures-stations.parquet
---

# SMHI Swedish Temperature Stations (Hourly)

Hourly air-temperature observations from **15 weather stations across Sweden**,
covering the **last 10 years** at **1-hour resolution**. Built as a
benchmark/evaluation dataset for **autoregressive time-series forecasting**.

- **Source:** [SMHI Open Data](https://opendata.smhi.se/) — Meteorological
  Observations (metobs) API, parameter `1` (air temperature, *momentanvärde,
  1 gång/tim*), `corrected-archive` (quality-controlled) period.
- **Resolution:** 1 hour (the finest available for SMHI temperature).
- **Time zone:** UTC.
- **Unit:** degrees Celsius (°C).
- **Quality:** only quality-controlled observations (codes `G` and `Y`) are kept.

## Files

| File | Description |
|------|-------------|
| `swedish-temperatures.parquet` | Main dataset. **Wide** layout: a regular hourly `DatetimeIndex` (UTC, named `timestamp`) × a 2-level **MultiIndex** of columns `(station_id, station_name)`; values are temperature in °C. Gaps in the regular hourly grid are explicit `NaN`. |
| `swedish-temperatures-stations.parquet` | Station metadata: `station_id`, `station_name`, `city`, `latitude`, `longitude`, `elevation` (m), `first_obs`, `last_obs`, `n_obs`, `pct_missing`. |

## Stations

Stockholm, Göteborg, Malmö, Uppsala, Norrköping, Linköping, Örebro, Karlstad,
Sundsvall, Östersund, Umeå, Luleå, Kiruna, Visby, Jönköping — geographically
diverse, currently-active stations, each with < 5 % missing over the window.

## Loading

The main file uses a pandas MultiIndex column layout, so load it with pandas:

```python
import pandas as pd
from huggingface_hub import hf_hub_download

path = hf_hub_download(
    repo_id="rebase-energy/smhi-stations",
    filename="swedish-temperatures.parquet",
    repo_type="dataset",
)
df = pd.read_parquet(path)                  # restores MultiIndex columns + UTC index
# df.columns.names == ['station_id', 'station_name']; df.index.name == 'timestamp'

# one station's series:
stockholm = df.xs("98230", level="station_id", axis=1)
```

Station metadata:

```python
meta = pd.read_parquet(hf_hub_download(
    repo_id="rebase-energy/smhi-stations",
    filename="swedish-temperatures-stations.parquet",
    repo_type="dataset",
))
```

## Suggested benchmark task

Autoregressive forecasting of hourly temperature per station. A simple
persistence baseline (`ŷ_{t+1} = y_t`) yields a mean absolute error of
~0.66 °C across stations — a reasonable lower bar to beat. The regular hourly
grid with explicit `NaN` gaps makes it straightforward to construct
fixed-horizon train/eval splits.

## License & attribution

The underlying observations are produced by **SMHI** (Swedish Meteorological and
Hydrological Institute) and distributed as open data under
**Creative Commons Attribution 4.0 (CC BY 4.0)**. Please attribute SMHI when
using this dataset. See <https://opendata.smhi.se/> for the source and terms.

## Reproducing

This dataset is built by `scripts/build_swedish_temperatures.py` in the
[emflow](https://github.com/rebase-energy/emflow) repository.
