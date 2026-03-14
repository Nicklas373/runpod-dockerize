# LLM Compressor Patch for Qwen 3.5

## Step

- Update llmcompressor, transformers to latest version
  ```shell
  pip uninstall llmcompressor transformers
  pip install llmcompressor
  pip install "transformers[torch]"
  ```
- Patch /usr/local/lib/python3.12/dist-packages/llmcompressor/utils/pytorch/module.py
  - Line 139
  ```Shell
      :return: list of class names that shouldn't be split
      """
  --- no_split_modules = model._get_no_split_modules("auto")
  +++ no_split_modules = model._no_split_modules
      if len(no_split_modules) <= 0:
          return ALL_TARGET
  ```
- Patch /usr/local/lib/python3.12/dist-packages/llmcompressor/modeling/\_\_\_init\_\_\.py
  - Line 18
  ```Shell
        from .qwen3_next_moe import CalibrationQwen3NextSparseMoeBlock  # noqa: F401
    +++ from .qwen3_5_vl_moe import CalibrateQwen3_5MoeTextSparseMoeBlock  # noqa: F401
        # TODO: add granite4, Qwen3Next
  ```
- Patch /usr/local/lib/python3.12/dist-packages/llmcompressor/utils/dev.py
  - Line 16

  ```Shell
        from transformers import AutoModelForCausalLM, PreTrainedModel
    --- from transformers.modeling_utils import TORCH_INIT_FUNCTIONS

    +++ # from transformers.modeling_utils import TORCH_INIT_FUNCTIONS
        from transformers.utils import SAFE_WEIGHTS_INDEX_NAME, WEIGHTS_INDEX_NAME
  ```

  - Line 26

  ```Shell
        +++ from torch import nn
        +++
        +++ TORCH_INIT_FUNCTIONS = {
        +++     "uniform_": nn.init.uniform_,
        +++     "normal_": nn.init.normal_,
        +++     "trunc_normal_": nn.init.trunc_normal_,
        +++     "constant_": nn.init.constant_,
        +++     "xavier_uniform_": nn.init.xavier_uniform_,
        +++     "xavier_normal_": nn.init.xavier_normal_,
        +++     "kaiming_uniform_": nn.init.kaiming_uniform_,
        +++     "kaiming_normal_": nn.init.kaiming_normal_,
        +++     "uniform": nn.init.uniform,
        +++     "normal": nn.init.normal,
        +++     "xavier_uniform": nn.init.xavier_uniform,
        +++     "xavier_normal": nn.init.xavier_normal,
        +++     "kaiming_uniform": nn.init.kaiming_uniform,
        +++     "kaiming_normal": nn.init.kaiming_normal,
        +++ }
  ```

- Copy qwen3_5_vl_moe.py to /usr/local/lib/python3.12/dist-packages/llmcompressor/modeling/qwen3_5_vl_moe.py

## References

All of files was taken from [llmcompressor](https://github.com/vllm-project/llm-compressor/commits/qwen3_5_support) at branch qwen3_5_support on revision [40c6211d1a1c22e6331f9018b687c559827b74db](https://github.com/vllm-project/llm-compressor/commit/40c6211d1a1c22e6331f9018b687c559827b74db)
