# Issues Identified in `src/lerobot/policies/pi06_star/processor_pi06_star.py`

## 1. Missing Default Advantage Status During Inference
- **Issue**: The code currently checks for `advantage_status` in `complementary_data`. If it's missing (which is the case during standard inference/evaluation), the `advantage_str` defaults to an empty string `""`.
- **Reference**: The Pi0.6 Star paper (Section VI-B) states: *"By default we evaluate with $\beta=1$,"* which implies $I_t = \text{True}$ (Positive Advantage) should be the default condition.
- **Consequence**: The generated prompt becomes `...; \nAction:`, missing the advantage indicator entirely. This results in a prompt format mismatch compared to training and fails to condition the model on "success", likely degrading performance significantly.
# **Recommendation**: Default to `Advantage: positive` when `advantage_status` is not provided.

## 2. Incorrect Advantage Prompt Ordering
- **Issue**: The code constructs the prompt as:
  `Task: {task}, State: {state}; {advantage}\nAction: `
  This places the Advantage indicator **before** the Action but also potentially **before** any model-generated thought/sub-task tokens if they were intended to be part of the "Action" generation phase (though the current code only handles inputs).
  More critically, the paper (Section V-B) states: *"The advantage indicator appears in the training sequence **after** $\hat{\ell}$ but **before** the (discretized and continuous) actions"*, where $\hat{\ell}$ is the sub-task prediction.
- **Consequence**: The current implementation puts Advantage conditioning inside the fixed prompt, which is correct *if* the sub-task is also part of the prompt inputs. However, if the model is expected to generate the sub-task $\hat{\ell}$ *after* receiving the prompt but *before* the Advantage token is effectively consumed/attended to for action generation, valid flow would be: `Prompt -> Sub-task -> Advantage -> Action`.
- **Note**: If $\hat{\ell}$ is just part of the input "Task" description in this implementation, the order might be fine. But if $\hat{\ell}$ is an output *token*, blocking the advantage string before it might be incorrect if the paper implies `Task -> State -> Subtask -> Advantage -> Action`.
