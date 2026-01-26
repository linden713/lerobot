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

from lerobot.configs.types import PipelineFeatureType, PolicyFeature
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
            advantage_str = ""
            if advantage_statuses is not None and i < len(advantage_statuses):
                status = advantage_statuses[i]
                if status: # True/Positive
                     advantage_str = "Advantage: positive "
                else:
                     advantage_str = "Advantage: negative "
            
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

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        transition = transition.copy()
        
        comp_data = transition.get(TransitionKey.COMPLEMENTARY_DATA, {})
        
        if "episode_index" in comp_data and "frame_index" in comp_data:
            ep_idx = comp_data["episode_index"]
            frame_idx = comp_data["frame_index"]
            
            if self.dataset_meta is not None:
                if "is_episode_successful" not in self.dataset_meta.episodes:
                    raise ValueError(
                        f"Dataset '{self.dataset_meta.repo_id}' metadata is missing 'is_episode_successful' field. "
                        "This field is required for PI06Star Value Function training to determine success/failure."
                    )

                try:
                    # In LeRobotDatasetMetadata, episodes is a dict containing 'length' key which is a list/array
                    total_steps = self.dataset_meta.episodes["length"][int(ep_idx)]
                    is_success = self.dataset_meta.episodes["is_episode_successful"][int(ep_idx)]
                except (KeyError, IndexError):
                    total_steps = self.max_task_length # Fallback
                    is_success = False

                current_step = int(frame_idx)
                
                c_fail = 0.0 if bool(is_success) else 2.0 * self.max_task_length
                
                steps_remaining = total_steps - current_step
                raw_value = -steps_remaining - c_fail
                norm_value = raw_value / self.max_task_length
                target_value = max(norm_value, -1.0)
                
                # Bins
                target_bin = int((target_value + 1.0) * 200)
                target_bin = max(0, min(200, target_bin))
                
                transition[TransitionKey.COMPLEMENTARY_DATA]["value_target"] = torch.tensor(target_bin, dtype=torch.long)

        return transition

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
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
