# LLM Compressor Patch for Qwen 3.5

## Step

- Copy qwen3_5_vl_moe.py to /usr/local/lib/python3.12/dist-packages/llmcompressor/src/llmcompressor/modeling/
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
- Update llmcompressor, transformers to latest version
  ```shell
  pip uninstall llmcompressor transformers
  pip install llmcompressor
  pip install "transformers[torch]"
  ```

## References

All of files was taken from [llmcompressor](https://github.com/vllm-project/llm-compressor/commits/qwen3_5_support) at branch qwen3_5_support on revision [40c6211d1a1c22e6331f9018b687c559827b74db](https://github.com/vllm-project/llm-compressor/commit/40c6211d1a1c22e6331f9018b687c559827b74db)
