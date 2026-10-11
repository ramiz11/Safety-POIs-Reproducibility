import re
import time
import json
import math
import os
import pickle as pkl
import tempfile
from pathlib import Path
import pandas as pd


def _safe_time_fmt(ts) -> str:
    """
    Cross-platform-ish time formatting.
    Tries %-I (Unix), falls back to %I (Windows) without leading-zero removal.
    """
    try:
        return ts.strftime("%B %d, %Y, at %-I:%M %p")
    except ValueError:
        return ts.strftime("%B %d, %Y, at %I:%M %p")

def _get_category_col(df: pd.DataFrame):
    """
    Prefer NYC column name if present, otherwise generic 'category'.
    Return None if neither exists.
    """
    if "poi_category_name" in df.columns:
        return "poi_category_name"
    if "category" in df.columns:
        return "category"
    return None


def convert_latlon_to_text(lat: float, lon: float) -> str:
    """
    Use geopy Nominatim to reverse-geocode (lat, lon) into an address.
    Returns a string with city, county, road, house_number, and a generic category.
    """
    from geopy.geocoders import Nominatim

    geolocator = Nominatim(user_agent="geo_converter")
    location = geolocator.reverse((lat, lon), exactly_one=True)

    if location and location.raw.get("address"):
        address = location.raw["address"]
        city = address.get("city", address.get("town", address.get("village", "Unknown City")))
        county = address.get("county", "Unknown County")
        road = address.get("road", "Unknown Road")
        house_number = address.get("house_number", "Unknown Number")
        category = address.get("amenity", "General Area")
        return f"{city} {county} {road} {house_number} ({category})"
    else:
        return "Address not found"


def build_coordinates2addresses(checkins_df: pd.DataFrame, out_path: str, throttle_s: float = 1.0):
    """
    For all unique (lat, lon) pairs in checkins_df,
    build a dict mapping (lat, lon) -> textual address.
    If a category column exists, replace the parentheses content with the POI category.
    NOTE: Nominatim usage policy discourages aggressive batching; throttle requests.
    """
    from tqdm import tqdm

    unique_coords = set(zip(checkins_df['latitude'], checkins_df['longitude']))
    location2address = {}
    cat_col = _get_category_col(checkins_df)

    for coords in tqdm(unique_coords, desc="Building coords->address map"):
        lat, lon = coords
        textual_address = convert_latlon_to_text(lat, lon)

        if cat_col is not None:
            subset = checkins_df[(checkins_df['latitude'] == lat) & (checkins_df['longitude'] == lon)]
            if len(subset) > 0:
                poi_cat = subset.iloc[0][cat_col]
                if pd.notna(poi_cat):
                    textual_address = re.sub(r"\([^)]*\)", f"({poi_cat})", textual_address)

        location2address[coords] = textual_address
        if throttle_s:
            time.sleep(throttle_s)  # Reduce latency from geocoding service

    with open(out_path, 'wb') as f:
        pkl.dump(location2address, f)
    return location2address


def load_coordinates2addresses(path: str) -> dict:
    """Load the pickled dictionary of (lat, lon) -> address."""
    with open(path, 'rb') as f:
        return pkl.load(f)


def build_prompt(traj: pd.DataFrame, userid: int, poi_id_range: int,
                 location2address: dict = None, *, include_safety: bool = False) -> str:
    """
    Describe the observed check-ins and use the final POI only as the answer.

    - Uses 'local_time' if present, else falls back to 'event_time_utc'.
    - Uses NYC 'poi_category_name' or generic 'category' if available; otherwise omits.
    - Each observed row retains its own latitude/longitude, even for repeated POIs.
    - Safety text, when requested, covers observed-to-observed transitions only.
    - location2address is retained for caller compatibility; addresses are not injected.
    """
    if len(traj) < 2:
        raise ValueError("A trajectory needs observed history and a target row")
    cat_col = _get_category_col(traj)

    # select a time column for display
    time_col = "local_time" if "local_time" in traj.columns else (
        "event_time_utc" if "event_time_utc" in traj.columns else None
    )
    if time_col is None:
        raise KeyError("Trajectory is missing both 'local_time' and 'event_time_utc'.")

    partial_traj = traj.iloc[:-1]
    final_checkin = traj.iloc[-1]

    # Summaries for partial trajectory
    lines = []
    for position, (_, row) in enumerate(partial_traj.iterrows()):
        poi_id = row.poi_id
        tstamp = _safe_time_fmt(row[time_col])
        try:
            latitude, longitude = float(row['latitude']), float(row['longitude'])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("Observed row is missing valid latitude/longitude") from exc
        if not (math.isfinite(latitude) and math.isfinite(longitude)
                and -90 <= latitude <= 90 and -180 <= longitude <= 180):
            raise ValueError("Observed row has invalid latitude/longitude")
        coordinates = f"<{latitude:.6f}, {longitude:.6f}>"
        if cat_col and pd.notna(row.get(cat_col)):
            cat_text = str(row[cat_col])
            lines.append(f"At {tstamp}, user {userid} visited POI id {poi_id} at {coordinates}, which is a {cat_text}.")
        else:
            lines.append(f"At {tstamp}, user {userid} visited POI id {poi_id} at {coordinates}.")
        if include_safety and position < len(partial_traj) - 1:
            score = row['normalized_safety']
            lines.append(f"The safety score from POI {poi_id} to the next POI is {round(score, 3)}.")

    trajectory_str = "\n".join(lines)

    # IMPORTANT: poi_id_range is the COUNT (max_id + 1). Display 0..(count-1).
    max_id_inclusive = max(0, poi_id_range - 1)

    given = "Given the data (including the route safety scores)," if include_safety else "Given the data,"
    safety_instruction = (
        "Please consider the user’s trajectory and these safety scores when determining the most likely POI.\n"
        if include_safety else ""
    )
    question = (
        f"<question>: The following is a trajectory of user {userid}:\n"
        f"{trajectory_str}\n\n"
        f"{given} which POI id will user {userid} visit next within 24 hours?\n"
        f"{safety_instruction}"
        f"Note that POI id is an integer in the range from 0 to {max_id_inclusive}."
    )
    answer = f"<answer>: POI id {final_checkin.poi_id}."
    return question + "\n" + answer


def write_textual_trajectories(numeric_root, output_root=None):
    """Regenerate six prompt JSONs from trusted numeric and Safety pickles.

    Validate matching rows and construct every prompt before replacing any text
    file. Numeric trajectories, split ordering and Safety values are never changed.
    """
    numeric_root = Path(numeric_root)
    output_root = Path(output_root) if output_root is not None else numeric_root
    numeric, scored = {}, {}
    for split in ('train', 'validation', 'test'):
        with (numeric_root / f'{split}_trajectories.pickle').open('rb') as stream:
            numeric[split] = pkl.load(stream)
        with (numeric_root / 'safety' / f'{split}_trajs_with_safety.pickle').open('rb') as stream:
            scored[split] = pkl.load(stream)
        if len(numeric[split]) != len(scored[split]):
            raise ValueError(f"Numeric/Safety trajectory count mismatch: {split}")
        for index, (plain, annotated) in enumerate(zip(numeric[split], scored[split])):
            if not set(plain.columns).issubset(annotated.columns) or not plain.equals(annotated[plain.columns]):
                raise ValueError(f"Numeric/Safety row mismatch: {split} trajectory {index}")

    poi_id_range = max((int(poi) for traj in numeric['train'] for poi in traj.poi_id), default=-1) + 1
    outputs = {}
    for split in numeric:
        for include_safety in (False, True):
            trajectories = scored[split] if include_safety else numeric[split]
            name = (f'safety/safety_textual_{split}_trajs.json' if include_safety
                    else f'textual_{split}_trajs.json')
            outputs[name] = [
                build_prompt(traj, traj.user_id.iloc[0], poi_id_range, include_safety=include_safety)
                for traj in trajectories
            ]
    for name, texts in outputs.items():
        path = output_root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary = tempfile.mkstemp(prefix=f'.{path.name}.', dir=path.parent)
        try:
            with os.fdopen(descriptor, 'w', encoding='utf-8') as stream:
                json.dump(texts, stream, ensure_ascii=False)
            os.replace(temporary, path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
    return {str(output_root / name): len(texts) for name, texts in outputs.items()}
