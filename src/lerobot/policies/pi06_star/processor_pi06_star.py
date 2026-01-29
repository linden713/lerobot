#!/usr/bin/env python

# Copyright 2025 Physical Intelligence and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from copy import deepcopy
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from lerobot.configs.types import PipelineFeatureType, PolicyFeature, FeatureType
from lerobot.policies.pi06_star.configuration_pi06_star import PI06StarConfig
from lerobot.policies.pi06_star.modeling_pi06_star import pad_vector
from lerobot.processor import (
    AddBatchDimensionProcessorStep,
    DeviceProcessorStep,
    NormalizerProcessorStep,
    PolicyAction,
    PolicyProcessorPipeline,
    ProcessorStep,
    ProcessorStepRegistry,
    RenameObservationsProcessorStep,
    TokenizerProcessorStep,
    UnnormalizerProcessorStep,
)
from lerobot.processor.converters import policy_action_to_transition, transition_to_policy_action
from lerobot.processor.core import EnvTransition, TransitionKey
from lerobot.utils.constants import (
    OBS_STATE,
    POLICY_POSTPROCESSOR_DEFAULT_NAME,
    POLICY_PREPROCESSOR_DEFAULT_NAME,
)


@ProcessorStepRegistry.register(name="pi06_star_prepare_state_tokenizer_processor_step")
@dataclass
class Pi06StarPrepareStateTokenizerProcessorStep(ProcessorStep):
    """
    Processor step to prepare the state and tokenize the language input.
    """

    max_state_dim: int = 32
    task_key: str = "task"

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        transition = transition.copy()

        state = transition.get(TransitionKey.OBSERVATION, {}).get(OBS_STATE)
        if state is None:
            raise ValueError("State is required for PI06Star")
        tasks = transition.get(TransitionKey.COMPLEMENTARY_DATA, {}).get(self.task_key)
        if tasks is None:
            raise ValueError("No task found in complementary data")

        # TODO: check if this necessary
        state = deepcopy(state)

        # Prepare state (pad to max_state_dim)
        state = pad_vector(state, self.max_state_dim)

        # State should already be normalized to [-1, 1] by the NormalizerProcessorStep that runs before this step
        # Discretize into 256 bins (see openpi `PaligemmaTokenizer.tokenize()`)
        state_np = state.cpu().numpy()
        discretized_states = np.digitize(state_np, bins=np.linspace(-1, 1, 256 + 1)[:-1]) - 1

        # Helper to get advantage conditioning from complementary data if available
        advantage_statuses = transition.get(TransitionKey.COMPLEMENTARY_DATA, {}).get("advantage_status")

        full_prompts = []
        for i, task in enumerate(tasks):
            cleaned_text = task.strip().replace("_", " ").replace("\n", " ")
            state_str = " ".join(map(str, discretized_states[i]))
            
            # ReCAP Advantage Conditioning
            if advantage_statuses is None:
                # Inference time default: assume positive advantage (success)
                advantage_str = "Advantage: positive "
            elif i < len(advantage_statuses):
                status = advantage_statuses[i]
                if status: # True/Positive
                     advantage_str = "Advantage: positive "
                else:
                     advantage_str = "Advantage: negative "
            else:
                # Fallback if status list is shorter than task list (shouldn't happen)
                advantage_str = "Advantage: positive "
            
            # Matches format Task: ..., State: ...; Advantage: ...\nAction: 
            full_prompt = f"Task: {cleaned_text}, State: {state_str}; {advantage_str}\nAction: "
            full_prompts.append(full_prompt)

        transition[TransitionKey.COMPLEMENTARY_DATA][self.task_key] = full_prompts
        # Normalize state to [-1, 1] range if needed (assuming it's already normalized by normalizer processor step!!)
        # Discretize into 256 bins (see openpi `PaligemmaTokenizer.tokenize()`)
        return transition

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        """
        This step does not alter the feature definitions.
        """
        return features


@ProcessorStepRegistry.register(name="pi06_star_value_target_processor_step")
@dataclass
class Pi06StarValueTargetProcessorStep(ProcessorStep):
    """
    Computes value target (time to completion) and adds it to the batch.
    """
    max_task_length: int = 500
    dataset_meta: Any = None # LeRobotDatasetMetadata

    episode_lengths: torch.Tensor | None = None
    episode_success: torch.Tensor | None = None
    episode_partial_success: torch.Tensor | None = None

    def __post_init__(self):
        if self.dataset_meta is None:
             return

        # Helper to access columns from either dict or HF Dataset
        episodes = self.dataset_meta.episodes
        # HF Dataset has column_names, dict has keys()
        available_keys = getattr(episodes, "column_names", None)
        if available_keys is None:
            available_keys = episodes.keys()
        
        available_keys = set(available_keys) # fast lookup

        if "length" not in available_keys:
            return  # Fallback to runtime error if lengths needed but missing

        # Load standard columns
        # Note: HF Dataset access by string returns a list/column, same as dict
        self.episode_lengths = torch.tensor(episodes["length"], dtype=torch.long)
        
        if "is_episode_successful" in available_keys:
            self.episode_success = torch.tensor(episodes["is_episode_successful"], dtype=torch.bool)
        else:
            self.episode_success = torch.zeros(len(self.episode_lengths), dtype=torch.bool)
        
        if "partial_success" in available_keys:
            self.episode_partial_success = torch.tensor(episodes["partial_success"], dtype=torch.float32)
        else:
            self.episode_partial_success = torch.zeros(len(self.episode_lengths), dtype=torch.float32)

        # Determine Task IDs for aggregation (unify task_index and tasks logic)
        task_ids = None
        if "task_index" in available_keys and episodes["task_index"] is not None:
            # Assuming task_index is a list or tensor of ints
            task_ids = [int(x) for x in episodes["task_index"]]
        elif "tasks" in available_keys:
            # tasks is usually list of lists of strings (e.g. [['task_desc'], ...])
            # We use the first task string as identifier
            tasks_col = episodes["tasks"]
            task_ids = []
            for t in tasks_col:
                if isinstance(t, list) and len(t) > 0:
                    task_ids.append(t[0])
                else:
                    task_ids.append(str(t))

        if task_ids is None:
             raise ValueError("Neither 'task_index' nor 'tasks' found in dataset metadata, required for Pi06Star.")

        # Compute Max Lengths per Task
        task_max_map = {}
        input_lengths = self.episode_lengths.tolist()
        
        for t_id, length in zip(task_ids, input_lengths):
            current_max = task_max_map.get(t_id, 0)
            if length > current_max:
                task_max_map[t_id] = length
        
        # Map back to per-episode max length
        self.episode_task_max_lengths = torch.tensor(
            [task_max_map[t_id] for t_id in task_ids], 
            dtype=torch.float32
        )


    def __call__(self, transition: EnvTransition) -> EnvTransition:
        """
        Compute value targets (return-to-go) and add to batch.
        """
        if self.episode_lengths is None:
             raise ValueError("Dataset metadata (episode_lengths) is required for Pi06StarValueTargetProcessorStep.")

        # The batch (transition) contains 'info' and 'complementary_data' dictionaries
        # We need to extract episode_index and frame_index from there.
        
        ep_indices = None
        frame_indices = None
        
        # Helper to look in nested dict
        def get_tensor(src, key):
            if key in src:
                val = src[key]
                if not isinstance(val, torch.Tensor):
                    return torch.tensor(val)
                return val
            return None

        # Check in 'info'
        info = transition.get("info", {})
        if "episode_index" in info:
            ep_indices = get_tensor(info, "episode_index")
        if "index" in info:
            frame_indices = get_tensor(info, "index")
        elif "frame_index" in info: # sometimes called frame_index
            frame_indices = get_tensor(info, "frame_index")

        # Check in 'complementary_data' if not found
        if ep_indices is None or frame_indices is None:
            comp = transition.get("complementary_data", {})
            if ep_indices is None and "episode_index" in comp:
                ep_indices = get_tensor(comp, "episode_index")
            if frame_indices is None:
                if "index" in comp:
                    frame_indices = get_tensor(comp, "index")
                elif "frame_index" in comp:
                    frame_indices = get_tensor(comp, "frame_index")

        if ep_indices is None or frame_indices is None:
             raise ValueError(f"Missing 'episode_index' or 'index' in batch info/complementary_data.")

        # Ensure they are tensors (handle batching)
        if not isinstance(ep_indices, torch.Tensor):
            ep_indices = torch.tensor(ep_indices)
        if not isinstance(frame_indices, torch.Tensor):
            frame_indices = torch.tensor(frame_indices)
            ep_indices = torch.tensor(ep_indices)
        if not isinstance(frame_indices, torch.Tensor):
            frame_indices = torch.tensor(frame_indices)
            
        # Handle singleton batch or scalar
        if ep_indices.ndim == 0:
            ep_indices = ep_indices.unsqueeze(0)
        if frame_indices.ndim == 0:
            frame_indices = frame_indices.unsqueeze(0)

        batch_size = ep_indices.shape[0]
        # print(f"DEBUG: Found indices. Batch size: {batch_size}", flush=True)
        target_bins = []

        # Vectorized lookup if metadata is available
        if self.episode_lengths is not None:
            # ep_indices might be on GPU or CPU, move to CPU for indexing cached CPU tensors
            # Ensure indices are long for indexing
            cpu_ep_indices = ep_indices.cpu().long()
            cpu_frame_indices = frame_indices.cpu().long()
            
            # Clamp in case indexes are out of bounds (shouldn't happen)
            cpu_ep_indices = torch.clamp(cpu_ep_indices, 0, len(self.episode_lengths) - 1)

            total_steps = self.episode_lengths[cpu_ep_indices]
            is_success_batch = self.episode_success[cpu_ep_indices]
            partial_success_batch = self.episode_partial_success[cpu_ep_indices]
            
            # Proceed with vectorized calculation (much faster than loop)
            current_steps = cpu_frame_indices
            
            # Calculate c_fails:
            # If Success: 0.0
            # If Failure: (1.0 - partial_success) * max_task_length
            # Note: We use 1.0 * max_len (instead of 2.0) to map partial=0 -> c_fail=max_len -> Value=-1
            # If partial=0.5 -> c_fail=0.5*max_len -> Value=-0.5
            
            # Use pre-computed task max lengths for this batch
            batch_max_lens = self.episode_task_max_lengths[cpu_ep_indices]
            
            c_fails = torch.where(
                is_success_batch, 
                torch.tensor(0.0), 
                (1.0 - partial_success_batch) * batch_max_lens
            )
            
            steps_remaining = total_steps - current_steps
            raw_values = -steps_remaining.float() - c_fails
            norm_values = raw_values / batch_max_lens
            target_values = torch.maximum(norm_values, torch.tensor(-1.0))
            
            # Bins
            target_bins_tensor = ((target_values + 1.0) * 200).long()
            target_bins_tensor = torch.clamp(target_bins_tensor, 0, 200)
            
            # To ensure compatibility with pipeline, we can keep it as tensor on correct device
            target_bins_tensor = target_bins_tensor.to(device=ep_indices.device)
            
            # Assign directly
            if TransitionKey.OBSERVATION not in transition:
                 transition[TransitionKey.OBSERVATION] = {}
            transition[TransitionKey.OBSERVATION]["value_target"] = target_bins_tensor
            
            return transition


    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        # Register value_target as an observation feature so it passes through the pipeline
        if PipelineFeatureType.OBSERVATION not in features:
            features[PipelineFeatureType.OBSERVATION] = {}
            
        features[PipelineFeatureType.OBSERVATION]["value_target"] = PolicyFeature(
            type=FeatureType.STATE,
            shape=(1,), # Scalar
        )
        return features


def make_pi06_star_pre_post_processors(
    config: PI06StarConfig,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
    dataset_meta: Any = None,
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    """
    Constructs pre-processor and post-processor pipelines for the PI06Star policy.


    The pre-processing pipeline prepares input data for the model by:
    1. Renaming features to match pretrained configurations.
    2. Normalizing input and output features based on dataset statistics.
    3. Adding a batch dimension.
    4. Appending a newline character to the task description for tokenizer compatibility.
    5. Tokenizing the text prompt using the PaliGemma tokenizer.
    6. Moving all data to the specified device.

    The post-processing pipeline handles the model's output by:
    1. Moving data to the CPU.
    2. Unnormalizing the output features to their original scale.

    Args:
        config: The configuration object for the PI06Star policy.
        dataset_stats: A dictionary of statistics for normalization.
        preprocessor_kwargs: Additional arguments for the pre-processor pipeline.
        postprocessor_kwargs: Additional arguments for the post-processor pipeline.

    Returns:
        A tuple containing the configured pre-processor and post-processor pipelines.
    """

    # Add remaining processors
    input_steps: list[ProcessorStep] = [
        RenameObservationsProcessorStep(rename_map={}),  # To mimic the same processor as pretrained one
        AddBatchDimensionProcessorStep(),
        # NOTE: NormalizerProcessorStep MUST come before Pi05PrepareStateTokenizerProcessorStep
        # because the tokenizer step expects normalized state in [-1, 1] range for discretization
        NormalizerProcessorStep(
            features={**config.input_features, **config.output_features},
            norm_map=config.normalization_mapping,
            stats=dataset_stats,
        ),
        Pi06StarPrepareStateTokenizerProcessorStep(max_state_dim=config.max_state_dim),
        TokenizerProcessorStep(
            tokenizer_name="google/paligemma-3b-pt-224",
            max_length=config.tokenizer_max_length,
            padding_side="right",
            padding="max_length",
        ),
    ]

    if config.use_value_function:
        input_steps.append(
            Pi06StarValueTargetProcessorStep(
                max_task_length=config.max_task_length,
                dataset_meta=dataset_meta
            )
        )

    input_steps.append(DeviceProcessorStep(device=config.device))

    output_steps: list[ProcessorStep] = [
        UnnormalizerProcessorStep(
            features=config.output_features, norm_map=config.normalization_mapping, stats=dataset_stats
        ),
        DeviceProcessorStep(device="cpu"),
    ]

    return (
        PolicyProcessorPipeline[dict[str, Any], dict[str, Any]](
            steps=input_steps,
            name=POLICY_PREPROCESSOR_DEFAULT_NAME,
        ),
        PolicyProcessorPipeline[PolicyAction, PolicyAction](
            steps=output_steps,
            name=POLICY_POSTPROCESSOR_DEFAULT_NAME,
            to_transition=policy_action_to_transition,
            to_output=transition_to_policy_action,
        ),
    )
