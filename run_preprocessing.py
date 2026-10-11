import os
import time
import json
import pickle as pkl
import pandas as pd
from configs import preprocessing_config as pc
import text_utils as tu


def main(dataset: str = "NYC", traj_len: int = 10, crime_radius: int = 500, crime_time_weeks: int = 4,
    base_dir: str = "/absolute/path/to/Safety-POIs-Reproducibility", *,
    prompts_only: bool = False, prompt_output_dir=None):
    pc.update_config(dataset, traj_len, crime_radius, crime_time_weeks, base_dir=base_dir)
    if prompts_only:
        outputs = tu.write_textual_trajectories(pc.CURRENT_DATA_DIR, prompt_output_dir)
        print("Created textual prompts:", outputs)
        return outputs

    import preprocessing as pp
    import safety as sf

    start = time.time()
    print(f"Run with: dataset={pc.DATASET}, traj_len={pc.TRAJ_LENGTH}, radius={pc.CRIME_RADIUS}m, time_window={pc.CRIME_TIME_WINDOW}w")
    # Load crimes + checkins
    crime_df = pd.read_csv(pc.CRIME_CSV)
    crime_gdf = pp.build_crime_geodf(crime_df)
    print(f"Loaded crime data, shape={crime_gdf.shape}")
    checkins_df = pp.load_checkins_dataset(pc.CHECKINS_CSV)
    print(f"Loaded {pc.DATASET} check-ins, shape={checkins_df.shape}")
    # Time-based split
    train_df, validation_df, test_df = pp.time_based_split(checkins_df, pc.TIME_SPLIT_RATIOS)
    print("Split sizes:", len(train_df), len(validation_df), len(test_df))
    # Remove unseen POIs/Users in val/test
    train_df, validation_df, test_df = pp.remove_unseen_pois_users(train_df, validation_df, test_df)
    print("Removed unseen POIs/users:", len(train_df), len(validation_df), len(test_df))
    # Reindex POIs to ensure contiguous [0..M-1]
    train_df, validation_df, test_df = pp.reindex_poi_ids(train_df, validation_df, test_df)
    print("Reindexed POI IDs to contiguous range")
    # Build trajectories
    if os.path.exists(pc.TRAIN_TRAJECTORIES_PKL_PATH) and os.path.exists(pc.VALIDATION_TRAJECTORIES_PKL_PATH) and os.path.exists(pc.TEST_TRAJECTORIES_PKL_PATH):
        print("Loading pre-saved trajectories (train/val/test)")
        with open(pc.TRAIN_TRAJECTORIES_PKL_PATH, 'rb') as f:
            train_trajectories = pkl.load(f)
        with open(pc.VALIDATION_TRAJECTORIES_PKL_PATH, 'rb') as f:
            validation_trajectories = pkl.load(f)
        with open(pc.TEST_TRAJECTORIES_PKL_PATH, 'rb') as f:
            test_trajectories = pkl.load(f)
    else:
        traj_dict = pp.build_all_phase_trajectories(
            train_df, validation_df, test_df,
            traj_len=pc.TRAJ_LENGTH,
            time_threshold=pd.Timedelta(hours=pc.TRAJ_TIME_THRESHOLD_HOURS)
        )
        train_trajectories, validation_trajectories, test_trajectories = (
            traj_dict['train'], traj_dict['validation'], traj_dict['test']
        )
        pp.save_pickle(train_trajectories, pc.TRAIN_TRAJECTORIES_PKL_PATH)
        pp.save_pickle(validation_trajectories, pc.VALIDATION_TRAJECTORIES_PKL_PATH)
        pp.save_pickle(test_trajectories, pc.TEST_TRAJECTORIES_PKL_PATH)
        print("Saved trajectories as pickle.")

    # Load caches
    if os.path.exists(pc.SEGMENTS_COORDINATES_HASHMAP_PKL_PATH):
        with open(pc.SEGMENTS_COORDINATES_HASHMAP_PKL_PATH, "rb") as f:
            print("Loaded segments coordinates hashmap...")
            segments_coordinates_hashmap = pkl.load(f)
    else:
        segments_coordinates_hashmap = {}

    if os.path.exists(pc.SEGMENTS_CRIMES_HASHMAP_JSON_PATH):
        with open(pc.SEGMENTS_CRIMES_HASHMAP_JSON_PATH, 'r') as f:
            print("Loaded segments crimes hashmap...")
            segments_crimes_hashmap = json.load(f)
    else:
        segments_crimes_hashmap = {}

    # Compute & store train crime counts
    if os.path.exists(pc.TRAIN_CRIME_SCORES_JSON_PATH):
        with open(pc.TRAIN_CRIME_SCORES_JSON_PATH, 'r') as f:
            train_crime_scores = json.load(f)
            print("Loaded raw train crime scores.")
    else:
        train_crime_scores = sf.compute_crime_scores_for_trajectories(
            train_trajectories, crime_gdf, segments_crimes_hashmap, segments_coordinates_hashmap,
            pc.CRIME_RADIUS, pc.CRIME_TIME_WINDOW
        )
        with open(pc.TRAIN_CRIME_SCORES_JSON_PATH, 'w') as f:
            json.dump(train_crime_scores, f)
        print("Stored raw train crime scores.")

    # Train distribution stats
    if os.path.exists(pc.TRAIN_CRIME_DIST_JSON_PATH):
        with open(pc.TRAIN_CRIME_DIST_JSON_PATH, 'r') as f:
            dist_stats = json.load(f)
            print("Loaded train crime distribution stats:", dist_stats)
    else:
        dist_stats = sf.derive_distribution_stats(train_crime_scores)
        with open(pc.TRAIN_CRIME_DIST_JSON_PATH, 'w') as f:
            json.dump(dist_stats, f)
        print("Computed train crime distribution stats:", dist_stats)

    # Inject safety into train trajectories
    if os.path.exists(pc.TRAIN_TRAJS_WITH_SAFETY_PKL_PATH):
        with open(pc.TRAIN_TRAJS_WITH_SAFETY_PKL_PATH, 'rb') as f:
            train_trajs_with_safety = pkl.load(f)
            print("Loaded train trajectories with safety scores")
    else:
        train_trajs_with_safety = sf.apply_safety_scores_to_train_trajectories(
            train_trajectories, train_crime_scores, dist_stats
        )
        pp.save_pickle(train_trajs_with_safety, pc.TRAIN_TRAJS_WITH_SAFETY_PKL_PATH)
        print("Injected safety into train trajectories")

    # Validation / Test safety (on the fly)
    if os.path.exists(pc.VALIDATION_TRAJS_WITH_SAFETY_PKL_PATH):
        with open(pc.VALIDATION_TRAJS_WITH_SAFETY_PKL_PATH, 'rb') as f:
            validation_trajs_with_safety = pkl.load(f)
            print("Loaded validation trajectories with safety scores")
    else:
        validation_trajs_with_safety = sf.apply_safety_scores_to_non_train_trajectories(
            validation_trajectories, crime_gdf, dist_stats,
            segments_crimes_hashmap, segments_coordinates_hashmap,
            pc.CRIME_RADIUS, pc.CRIME_TIME_WINDOW
        )
        pp.save_pickle(validation_trajs_with_safety, pc.VALIDATION_TRAJS_WITH_SAFETY_PKL_PATH)
        print("Injected safety into validation trajectories")

    if os.path.exists(pc.TEST_TRAJS_WITH_SAFETY_PKL_PATH):
        with open(pc.TEST_TRAJS_WITH_SAFETY_PKL_PATH, 'rb') as f:
            test_trajs_with_safety = pkl.load(f)
            print("Loaded test trajectories with safety scores")
    else:
        test_trajs_with_safety = sf.apply_safety_scores_to_non_train_trajectories(
            test_trajectories, crime_gdf, dist_stats,
            segments_crimes_hashmap, segments_coordinates_hashmap,
            pc.CRIME_RADIUS, pc.CRIME_TIME_WINDOW
        )
        pp.save_pickle(test_trajs_with_safety, pc.TEST_TRAJS_WITH_SAFETY_PKL_PATH)
        print("Injected safety into test trajectories")

    # Always regenerate derived text; existing JSONs may use an older format.
    outputs = tu.write_textual_trajectories(pc.CURRENT_DATA_DIR, prompt_output_dir)
    print("Created textual prompts:", outputs)

    end = time.time()
    print(f"ALL DONE in {(end - start) / 60:.2f} minutes.")
    return outputs


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="NYC", choices=["NYC", "CHICAGO"])
    parser.add_argument("--traj_len", type=int, default=20)
    parser.add_argument("--crime_radius", type=int, default=500)
    parser.add_argument("--crime_time_weeks", type=int, default=4)
    parser.add_argument("--base_dir", default="/absolute/path/to/Safety-POIs-Reproducibility")
    parser.add_argument("--prompts-only", "--prompts_only", action="store_true",
                        help="Regenerate text from existing numeric/Safety pickles; no CSVs or routing calls")
    parser.add_argument("--prompt-output-dir", default=None,
                        help="Optional prompt output directory; defaults to the selected trajectory directory")
    args = parser.parse_args()
    main(
        dataset=args.dataset,
        traj_len=args.traj_len,
        crime_radius=args.crime_radius,
        crime_time_weeks=args.crime_time_weeks,
        base_dir=args.base_dir,
        prompts_only=args.prompts_only,
        prompt_output_dir=args.prompt_output_dir,
    )
