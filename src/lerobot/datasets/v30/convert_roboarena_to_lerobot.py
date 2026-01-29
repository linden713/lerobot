#!/usr/bin/env python

"""
Script to convert RoboArena dataset to LeRobot v3.0 format.
"""

import argparse
import glob
import logging
import math
import os
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import tqdm
import yaml
from datasets import Dataset, Features, Image, Value, Sequence

from lerobot.datasets.compute_stats import aggregate_stats, compute_episode_stats
from lerobot.datasets.lerobot_dataset import CODEBASE_VERSION
from lerobot.datasets.utils import (
    DEFAULT_CHUNK_SIZE,
    DEFAULT_DATA_PATH,
    DEFAULT_VIDEO_PATH,
    DEFAULT_FEATURES,
    create_empty_dataset_info,
    update_chunk_file_indices,
    write_episodes,
    write_info,
    write_stats,
    write_tasks,
    get_parquet_file_size_in_mb,
)
from lerobot.utils.utils import init_logging
from lerobot.datasets.video_utils import get_video_duration_in_s, concatenate_video_files

# Constants
FPS = 30  # Assumed FPS, check if this is correct or available in metadata
DEFAULT_DATA_FILE_SIZE_IN_MB = 100
DEFAULT_VIDEO_FILE_SIZE_IN_MB = 500

def load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def get_sessions(root, session_id=None):
    sessions_dir = root / "evaluation_sessions"
    if session_id:
         session_paths = [sessions_dir / session_id]
         if not session_paths[0].exists():
             raise ValueError(f"Session {session_id} not found in {sessions_dir}")
    else:
        session_paths = sorted(list(sessions_dir.glob("*")))
    
    # Filter only directories
    session_paths = [p for p in session_paths if p.is_dir()]
    return session_paths

def parse_session(session_path: Path):
    metadata_path = session_path / "metadata.yaml"
    if not metadata_path.exists():
        logging.warning(f"No metadata.yaml found in {session_path}, skipping.")
        return []
    
    metadata = load_yaml(metadata_path)
    task_description = metadata.get("language_instruction", "default task")
    policies_meta = metadata.get("policies", {})
    
    parsed_episodes = []
    
    # Iterate over subdirectories which represent policy rollouts
    # Expecting folders like "A_pi0_droid", "B_paligemma_vq_droid", etc.
    policy_dirs = sorted([d for d in session_path.glob("*") if d.is_dir()])
    
    for pol_dir in policy_dirs:
        # Check if it has an NPZ format we expect: *npz_file.npz or proprio_and_actions.npz
        # Also could be named with timestamp
        npz_files = list(pol_dir.glob("*.npz"))
        if not npz_files:
            continue
        
        # Heuristic: Pick the largest NPZ file or specifically proprio_and_actions if exists
        # Based on previous inspection the files look like: pi0_droid_2025_04_29_10_46_23_npz_file.npz
        npz_file = npz_files[0] 
        
        # Check for videos
        # Expecting left_shoulder.mp4, right_shoulder.mp4, wrist.mp4 based on plan
        # Actually based on ls earlier: pi0_droid_..._video_left.mp4
        
        # We need to be flexible with video names or map them
        video_files = list(pol_dir.glob("*.mp4"))
        if not video_files:
             logging.warning(f"No videos found in {pol_dir} (Session: {session_path.name}), skipping.")
             continue

        # Map videos to keys
        videos_map = {}
        for v in video_files:
            if "left" in v.name:
                videos_map["observation.images.left"] = v
            elif "right" in v.name:
                videos_map["observation.images.right"] = v
            elif "wrist" in v.name:
                videos_map["observation.images.wrist"] = v
        
        if not videos_map:
             logging.warning(f"Could not key map videos in {pol_dir} (Session: {session_path.name}). Found: {[v.name for v in video_files]}, skipping.")
             continue
        
        # Get policy name key from directory name?
        # Dir name: A_pi0_droid. Metadata keys: A, B, ...
        # Match "A" from "A_pi0_droid"
        pol_key = pol_dir.name.split("_")[0]
        
        success = False
        partial_success = 0.0
        if pol_key in policies_meta:
             success = bool(policies_meta[pol_key].get("binary_success", 0))
             partial_success = float(policies_meta[pol_key].get("partial_success", 0.0))
        
        parsed_episodes.append({
            "npz_path": npz_file,
            "videos": videos_map,
            "task": task_description,
            "is_episode_successful": success,
            "partial_success": partial_success,
            "session_id": session_path.name,
            "policy_id": pol_dir.name
        })
        
    return parsed_episodes


def convert_dataset(
    root: Path,
    output_dir: Path,
    repo_id: str,
    session_id: str | None = None,
    push_to_hub: bool = False,
):
    logging.info(f"Converting dataset from {root} to {output_dir}")
    
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)
    
    sessions = get_sessions(root, session_id)
    logging.info(f"Found {len(sessions)} sessions.")
    
    all_episodes_info = []
    
    for session in tqdm.tqdm(sessions, desc="Parsing sessions"):
        all_episodes_info.extend(parse_session(session))
    
    logging.info(f"Found {len(all_episodes_info)} valid episodes.")
    
    if len(all_episodes_info) == 0:
        logging.error("No episodes found. Exiting.")
        return

    # Prepare features definition
    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": (8,),
            "names": [
                "joint_position_0", "joint_position_1", "joint_position_2", "joint_position_3", "joint_position_4", "joint_position_5", "joint_position_6",
                "gripper_position" 
            ]
        },
        "observation.cartesian_position": {
            "dtype": "float32",
            "shape": (6,),
            "names": ["x", "y", "z", "roll", "pitch", "yaw"] # Assuming 6D
        },
        "action": {
            "dtype": "float32",
            "shape": (8,),
            "names": [
                "action_joint_0", "action_joint_1", "action_joint_2", "action_joint_3", "action_joint_4", "action_joint_5", "action_joint_6",
                "action_gripper"
            ]
        },
        "next.done": {
            "dtype": "bool",
            "shape": (1,),
            "names": None
        }
    }
    
    # Add DEFAULT_FEATURES
    features.update(DEFAULT_FEATURES)
    
    # Add video features
    # Check first episode to see which cameras are available
    # Assuming all episodes have same cameras for consistency, or we handle missing ones
    # For now assume consistent cameras: left, right, wrist
    if all_episodes_info:
        sample_videos = all_episodes_info[0]["videos"]
        for key in sample_videos:
            features[key] = {
                "dtype": "video",
                "shape": (224, 224, 3), # Placeholder shape, will be updated by video info?
                "names": ["height", "width", "channel"],
                "info": None
            }

    # Initialize Info
    info = create_empty_dataset_info(
        codebase_version=CODEBASE_VERSION,
        fps=FPS,
        features=features,
        use_videos=True,
    )
    info["data_path"] = DEFAULT_DATA_PATH
    info["video_path"] = DEFAULT_VIDEO_PATH

    # Processing state variables
    chunk_idx = 0
    file_idx = 0
    current_data_size_mb = 0
    paths_to_cat_data = []
    
    # Video processing state variables (per camera)
    video_states = {
        k: {
            "chunk_idx": 0,
            "file_idx": 0,
            "current_size_mb": 0,
            "paths_to_cat": [],
            "current_batch_duration": 0.0
        } for k in features if features[k]["dtype"] == "video"
    }

    tasks_set = set()
    rows = []
    
    # Main processing loop
    episode_idx = 0
    
    
    # We need to collect stats to aggregate later
    all_stats = []

    # Create temp dir outside loop
    temp_dir = output_dir / "temp_data"
    temp_dir.mkdir(exist_ok=True)
    
    current_global_frame_idx = 0

    for ep_info in tqdm.tqdm(all_episodes_info, desc="Processing episodes"):
        npz_path = ep_info["npz_path"]
        
        # Validate video keys match expected features
        expected_video_keys = [k for k in features if features[k]["dtype"] == "video"]
        current_video_keys = list(ep_info["videos"].keys())
        
        if set(expected_video_keys) != set(current_video_keys):
            logging.warning(f"Episode {episode_idx} (Session: {ep_info['session_id']}) missing/mismatch video keys. Expected {expected_video_keys}, got {current_video_keys}. Skipping.")
            continue
            
        # Load NPZ
        try:
            arr = np.load(npz_path, allow_pickle=True)
            if 'data' in arr:
                data_objs = arr['data']
                
                n_frames = len(data_objs)
                if n_frames == 0:
                     logging.warning(f"Empty data in {npz_path}, skipping.")
                     continue
                     
                state_arr = np.zeros((n_frames, 8), dtype=np.float32)
                cart_arr = np.zeros((n_frames, 6), dtype=np.float32)
                action_arr = np.zeros((n_frames, 8), dtype=np.float32)
                
                valid_frame = True
                for i, frame in enumerate(data_objs):
                    if not isinstance(frame, dict):
                         logging.warning(f"Frame {i} in {npz_path} is not a dict. Skipping episode.")
                         valid_frame = False
                         break
                    
                    try:
                        # State: joint_position (7) + gripper_position (scalar)
                        jp = np.array(frame["joint_position"]).flatten()
                        
                        # Fix gripper position extraction
                        gp_raw = np.array(frame["gripper_position"]).flatten()
                        if gp_raw.size == 1:
                            gp = float(gp_raw[0])
                        else:
                            # Fallback if empty or weird
                            gp = 0.0 # Or raise?
                            
                        
                        if jp.shape[0] != 7:
                            # Try simple fix if it's 8 (maybe included gripper?)
                            if jp.shape[0] == 8:
                                jp = jp[:7] 
                            elif jp.size == 0:
                                # Skip invalid
                                raise ValueError("Empty joint position")
                            else:
                                raise ValueError(f"Joint position shape mismatch: {jp.shape}")
                                
                        state_arr[i, :7] = jp
                        state_arr[i, 7] = gp
                        
                        # Cartesian: cartesian_position (6)
                        cp = np.array(frame["cartesian_position"]).flatten()
                        if cp.shape[0] != 6:
                             # Handle mismatch
                             if cp.shape[0] > 6:
                                 cp = cp[:6]
                             elif cp.shape[0] < 6:
                                 # Pad?
                                 cp_padded = np.zeros(6, dtype=np.float32)
                                 if cp.shape[0] > 0:
                                     cp_padded[:cp.shape[0]] = cp
                                 cp = cp_padded
                        cart_arr[i] = cp
                        
                        # Action: action (8)
                        ac = np.array(frame["action"]).flatten()
                        if ac.shape[0] != 8:
                            if ac.shape[0] > 8:
                                ac = ac[:8]
                            elif ac.shape[0] < 8:
                                ac_padded = np.zeros(8, dtype=np.float32)
                                if ac.shape[0] > 0:
                                    ac_padded[:ac.shape[0]] = ac
                                ac = ac_padded
                        action_arr[i] = ac
                        
                    except Exception as e:
                        logging.error(f"Error processing frame {i} in {npz_path}: {e}")
                        # logging.error(f"Debug: jp={jp}, gp={gp}, cp={cp}, ac={ac}")
                        valid_frame = False
                        break
                
                if not valid_frame:
                    continue

            else:
                logging.warning(f"No 'data' key in {npz_path}, skipping.")
                continue
                
        except Exception as e:
            logging.error(f"Failed to load {npz_path}: {e}")
            continue

        # Create DataFrame for this episode
        ep_df = pd.DataFrame({
            "observation.state": list(state_arr), 
            "observation.cartesian_position": list(cart_arr),
            "action": list(action_arr),
            "episode_index": episode_idx,
            "frame_index": np.arange(0, n_frames),
            "timestamp": np.arange(0, n_frames) / FPS,
            "next.done": [False] * (n_frames - 1) + [True],
            # Add other default features if needed
            "index": np.arange(current_global_frame_idx, current_global_frame_idx + n_frames), 
            "task_index": 0 # Placeholder, will be updated? or just keep 0 if 1 task
        })
        
        current_global_frame_idx += n_frames
        
        ep_parquet_path = temp_dir / f"ep_{episode_idx}.parquet"
        
        
        ep_df.to_parquet(ep_parquet_path, index=False)
        
        ep_size_mb = get_parquet_file_size_in_mb(ep_parquet_path)
        
        # Chunking Logic for Data
        current_data_size_mb += ep_size_mb
        paths_to_cat_data.append(ep_parquet_path)
        
        data_chunk_idx = chunk_idx
        data_file_idx = file_idx
        
        if current_data_size_mb >= DEFAULT_DATA_FILE_SIZE_IN_MB:
             # Flush data
             final_path = output_dir / DEFAULT_DATA_PATH.format(chunk_index=chunk_idx, file_index=file_idx)
             final_path.parent.mkdir(parents=True, exist_ok=True)
             
             # Concat
             dfs = [pd.read_parquet(p) for p in paths_to_cat_data]
             combined_df = pd.concat(dfs, ignore_index=True)
             combined_df.to_parquet(final_path, index=False)
             
             # Reset
             chunk_idx, file_idx = update_chunk_file_indices(chunk_idx, file_idx, DEFAULT_CHUNK_SIZE)
             current_data_size_mb = 0
             paths_to_cat_data = []

        
        # Handle Videos
        # For each camera
        video_meta = {}
        for cam_key, vid_file in ep_info["videos"].items():
            # Get Duration
            # We trust fps=30?
            dur = get_video_duration_in_s(vid_file)
            
            # Use video_utils concatenate
            # Check size limit
            v_state = video_states[cam_key]
            
            # Simple file size check
            fs_mb = vid_file.stat().st_size / (1024 * 1024)
            
            v_state["current_size_mb"] += fs_mb
            v_state["paths_to_cat"].append(vid_file)
            v_state["current_batch_duration"] += dur # track if needed
            
            vid_chunk = v_state["chunk_idx"]
            vid_file_idx = v_state["file_idx"]
            
            # Metadata for this episode's video portion
            start_timestamp = v_state["current_batch_duration"] - dur # Duration includes current
            end_timestamp = v_state["current_batch_duration"]
            
            video_meta[f"videos/{cam_key}/chunk_index"] = vid_chunk
            video_meta[f"videos/{cam_key}/file_index"] = vid_file_idx
            video_meta[f"videos/{cam_key}/from_timestamp"] = start_timestamp
            video_meta[f"videos/{cam_key}/to_timestamp"] = end_timestamp
            
            if v_state["current_size_mb"] >= DEFAULT_VIDEO_FILE_SIZE_IN_MB:
                # Flush
                dest_path = output_dir / DEFAULT_VIDEO_PATH.format(video_key=cam_key, chunk_index=vid_chunk, file_index=vid_file_idx)
                dest_path.parent.mkdir(parents=True, exist_ok=True)
                concatenate_video_files(v_state["paths_to_cat"], dest_path)
                
                # Reset
                ni, nf = update_chunk_file_indices(vid_chunk, vid_file_idx, DEFAULT_CHUNK_SIZE)
                v_state["chunk_idx"] = ni
                v_state["file_idx"] = nf
                v_state["current_size_mb"] = 0
                v_state["paths_to_cat"] = []
                v_state["current_batch_duration"] = 0.0

        
        # Stats
        # Compute stats for this episode
        # Convert to dict of numpy arrays for compute_episode_stats
        ep_dict_for_stats = {}
        for k in features:
            if k in ep_df.columns:
                try:
                     stack = np.stack(ep_df[k].values)
                     ep_dict_for_stats[k] = stack
                except Exception:
                     # Skip if not stackable?
                     pass
                
        ep_stats = compute_episode_stats(ep_dict_for_stats, features)
        all_stats.append(ep_stats)

        # Meta row
        task_str = ep_info["task"]
        tasks_set.add(task_str)
        
        row = {
            "episode_index": episode_idx,
            "data/chunk_index": data_chunk_idx,
            "data/file_index": data_file_idx,
            "length": n_frames,
            "task": task_str, 
            "is_episode_successful": ep_info["is_episode_successful"],
            "partial_success": ep_info["partial_success"],
            **video_meta
        }
        rows.append(row)
        
        episode_idx += 1

    # Flush remaining data
    if paths_to_cat_data:
         final_path = output_dir / DEFAULT_DATA_PATH.format(chunk_index=chunk_idx, file_index=file_idx)
         final_path.parent.mkdir(parents=True, exist_ok=True)
         dfs = [pd.read_parquet(p) for p in paths_to_cat_data]
         combined_df = pd.concat(dfs, ignore_index=True)
         combined_df.to_parquet(final_path, index=False)
    
    # Flush remaining videos
    for cam_key, v_state in video_states.items():
        if v_state["paths_to_cat"]:
            dest_path = output_dir / DEFAULT_VIDEO_PATH.format(video_key=cam_key, chunk_index=v_state["chunk_idx"], file_index=v_state["file_idx"])
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            concatenate_video_files(v_state["paths_to_cat"], dest_path)

    shutil.rmtree(temp_dir)

    # Calculate global frame indices for metadata
    current_global_idx = 0
    for r in rows:
        l = r["length"]
        r["dataset_from_index"] = current_global_idx
        r["dataset_to_index"] = current_global_idx + l
        current_global_idx += l
        
    # Tasks
    tasks_list = sorted(list(tasks_set))
    tasks_df = pd.DataFrame({"task_index": range(len(tasks_list)), "task": tasks_list})
    tasks_df = tasks_df.set_index("task")
    write_tasks(tasks_df, output_dir)
    
    # Identify task index for each episode
    task_to_idx = {t: i for i, t in enumerate(tasks_list)}
    
    # Metadata / Episodes Parquet
    episodes_dict = {
        "episode_index": [],
        "tasks": [],
        "length": [],
        "dataset_from_index": [],
        "dataset_to_index": [],
        "data/chunk_index": [],
        "data/file_index": [],
        "is_episode_successful": [],
        "partial_success": [],
        "meta/episodes/chunk_index": [],
        "meta/episodes/file_index": []
    }
    
    # Add video cols
    for k in video_states:
        episodes_dict[f"videos/{k}/chunk_index"] = []
        episodes_dict[f"videos/{k}/file_index"] = []
        episodes_dict[f"videos/{k}/from_timestamp"] = []
        episodes_dict[f"videos/{k}/to_timestamp"] = []
    
    # Add stats cols
    if all_stats:
        final_stats = aggregate_stats(all_stats)
        write_stats(final_stats, output_dir)
        
        first_flat_stats =  pd.json_normalize(all_stats[0], sep="/").to_dict(orient='records')[0]
        for k in first_flat_stats:
            episodes_dict[f"stats/{k}"] = []
    
    for i, r in enumerate(rows):
        episodes_dict["episode_index"].append(r["episode_index"])
        episodes_dict["tasks"].append([r["task"]]) 
        episodes_dict["length"].append(r["length"])
        episodes_dict["dataset_from_index"].append(r["dataset_from_index"])
        episodes_dict["dataset_to_index"].append(r["dataset_to_index"])
        episodes_dict["data/chunk_index"].append(r["data/chunk_index"])
        episodes_dict["data/file_index"].append(r["data/file_index"])
        episodes_dict["is_episode_successful"].append(r["is_episode_successful"])
        episodes_dict["partial_success"].append(r["partial_success"])
        episodes_dict["meta/episodes/chunk_index"].append(0)
        episodes_dict["meta/episodes/file_index"].append(0)
        
        for k in video_states:
             episodes_dict[f"videos/{k}/chunk_index"].append(r[f"videos/{k}/chunk_index"])
             episodes_dict[f"videos/{k}/file_index"].append(r[f"videos/{k}/file_index"])
             episodes_dict[f"videos/{k}/from_timestamp"].append(r[f"videos/{k}/from_timestamp"])
             episodes_dict[f"videos/{k}/to_timestamp"].append(r[f"videos/{k}/to_timestamp"])
             
        if all_stats:
            flat_st = pd.json_normalize(all_stats[i], sep="/").to_dict(orient='records')[0]
            for k, v in flat_st.items():
                episodes_dict[f"stats/{k}"].append(v)
            
    # Write Episodes Parquet
    episodes_table = pd.DataFrame(episodes_dict)
    
    write_episodes(Dataset.from_pandas(episodes_table), output_dir)
    
    # Finalize Info
    info["total_episodes"] = len(rows)
    info["total_frames"] = current_global_idx
    info["total_tasks"] = len(tasks_list)
    write_info(info, output_dir)
    
    logging.info("Conversion complete!")

if __name__ == "__main__":
    init_logging()
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, required=True, help="Path to RoboArena dataset dump")
    parser.add_argument("--output", type=str, default=None, help="Output directory")
    parser.add_argument("--repo-id", type=str, default="lerobot/roboarena_v3", help="Repo ID")
    parser.add_argument("--session", type=str, default=None, help="Specific session ID to convert")
    
    args = parser.parse_args()
    
    if args.output is None:
        args.output = str(Path(args.root).parent / "roboarena_lerobot_v30")
        
    convert_dataset(Path(args.root), Path(args.output), args.repo_id, args.session)
