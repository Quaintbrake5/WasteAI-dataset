# TODO: Refactor Waste-Dataset/waste_data.py for Performance and Bug Fixes

## Pending Tasks
- [ ] Fix the function name bug: Correct `load_data()` to `load_data_stratified()` in main().
- [ ] Refactor fine_tune_model to take datasets as parameters instead of reloading data.
- [ ] Optimize tf.data pipelines: Improve prefetching, caching, and parallel calls in data loading.
- [ ] Update main() to pass datasets to fine_tune_model and streamline test dataset creation.
- [ ] Add type hints to all functions.
- [ ] Add docstrings and better error handling.
- [ ] Use a dataclass for configuration to make it more flexible and readable.
- [ ] Improve weight transfer in fine-tuning to handle mismatches gracefully.
- [ ] Add efficient checkpoint handling to avoid overwriting issues.

## Completed Tasks
- [x] Analyze code and create refactoring plan.
- [x] Get user approval for the plan.
