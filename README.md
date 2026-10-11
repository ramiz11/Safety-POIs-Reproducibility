
# Safety POIs – Reproducibility

Reproducibility repository for **safety-annotated next-POI datasets** (NYC + Chicago). This repo contains:

- The **safety score computation** pipeline (OSRM route → crime exposure → robust normalization → safety score)
- End‑to‑end preprocessing to build trajectories and prompts

> Note:
> This repository generates datasets and textual exports; it does **not** implement our safety-augmented LLM method. Method training and evaluation are maintained in [SafetyIsAllYouNeed](https://github.com/ramiz11/SafetyIsAllYouNeed).

---

## Repository Structure

```
configs/                 # runtime configuration
preprocessing.py         # time normalization, splits, trajectories
run_preprocessing.py     # end‑to‑end preprocessing pipeline
safety.py                # OSRM routing + crime exposure + safety scores
text_utils.py            # coordinate-bearing prompts, with and without Safety
requirements/            # pip requirements
```

---

## Quick Start

### 1) Install dependencies

Preprocessing (geo stack):
```bash
pip install -r requirements/preprocessing.txt
```

### 2) Configure paths

Edit `configs/preprocessing_config.py` or pass `base_dir` at runtime. Place the
prepared input CSVs listed under [Data Requirements](#data-requirements) beneath
that directory. Use a fresh data directory to rebuild numeric trajectories;
existing trajectories and route/Safety caches are reused when present.

### 3) Preprocess data + safety scores

```bash
python run_preprocessing.py \
  --dataset NYC \
  --traj_len 20 \
  --crime_radius 500 \
  --crime_time_weeks 4 \
  --base_dir /absolute/path/to/Safety-POIs-Reproducibility
```

Outputs:

- trajectory pickles (`{train,validation,test}_trajectories.pickle`)
- safety‑score annotations (crime exposure + normalized safety score)
- `textual_{train,validation,test}_trajs.json`
- `safety/safety_textual_{train,validation,test}_trajs.json`

For Chicago, use `--dataset CHICAGO --traj_len 20 --crime_radius 1000 --crime_time_weeks 3`.

Numeric trajectories retain all check-ins and attributes. Text prompts include
each observed check-in's own `<latitude, longitude>` pair (six decimals), timestamp,
POI and optional category. The final POI is the answer; its timestamp, coordinates
and category are not included in the prompt. For length 20, the Safety variant
includes 18 observed-to-observed transition scores, not the transition to the target.

### 4) Regenerate only textual exports

When numeric and Safety-annotated trajectory pickles already exist:

```bash
python run_preprocessing.py \
  --dataset NYC --traj_len 20 --crime_radius 500 --crime_time_weeks 4 \
  --base_dir /absolute/path/to/Safety-POIs-Reproducibility \
  --prompts-only
```

This replaces the six textual JSONs without reading source CSVs, contacting
routing services, or changing numeric data. Both this option and a normal run
regenerate text instead of reusing stale JSONs. Add `--prompt-output-dir /path/to/export`
to write a separate copy. The same option works with the Chicago parameters above.

---

## Safety Score (Core Method)

For each consecutive POI pair `(p_j, p_{j+1})` at time `t_j`:

1. **Route**: compute walking route via OSRM
2. **Crime exposure**: count incidents within a buffer `R` meters and time window `[t_j − Δ, t_j]`
3. **Robust scaling** (train‑only):
   - `median` and `IQR` of train crime counts
   - `r = clip((c − median) / IQR, 0, 1)`
4. **Safety score**:
   - `s = 1 − r` (higher is safer)

The score is **relative exposure**, not a probability of harm.

---

## Data Requirements

The pipeline starts from prepared, POI-mapped check-ins and aligned crime CSVs,
not directly from the upstream raw downloads. These inputs and their source notes
are in the [companion data directory](https://github.com/ramiz11/SafetyIsAllYouNeed/tree/main/data).
Clone that repository and run `git lfs install` and `git lfs pull` there to obtain
the actual data rather than LFS pointers. Copy the required inputs beneath your
`base_dir`, preserving these paths:

- NYC check-ins: `data/NYC_checkins/raw/dataset_w_mapped_poi.csv`
- NYC crime: `data/NYPD_CrimeData/Preprocessed_forsquare_nyc_alligned_subset_data.csv`
- Chicago check-ins: `data/Chicago_checkins/raw/dataset_w_mapped_poi.csv`
- Chicago crime: `data/Chicago_CrimeData/Preprocessed_gowalla_chicago_alligned_subset_data.csv`

Alternatively, use that checkout as `base_dir` with `--prompts-only` to export
from its supplied numeric/Safety trajectories. Use a separate `--prompt-output-dir`
if you want to leave its existing text files untouched. Respect the original data
licenses and attribution requirements described in its data notes.

**Check‑ins** must include:

- `user_id`, `poi_id`, `latitude`, `longitude`
- `local_time` or `checkin_time` (strings; naive or tz‑aware)
- `category` (optional)

**Crime data**:

- NYC: `complaint_date_start`, `complaint_date_end`, `Latitude`, `Longitude`
- Chicago: `Date`, `Latitude`, `Longitude`

---

## Notes

- OSRM public endpoint is used by default (`OSRM_BASE_URL`).
- For heavy use, host your own OSRM instance and update the config.
- Generated prompts are audited against the supplied numeric/Safety trajectories.
  A fresh CSV-to-Safety rebuild is a separate check; live routing and different
  source snapshots can change those values.
- Crime data can be biased; safety scores are proxies, not ground truth.

---

## Acknowledgments

- OSRM routing engine
- NYC & Chicago public crime datasets
