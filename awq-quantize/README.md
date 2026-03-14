# LLM Compressor - AWQ Quantization (AWQModifier)

Quantize any LLM Models into AWQ 4 Bit format by using LLM Compressor package from [llm-compressor](https://github.com/vllm-project/llm-compressor/tree/main).

## Current Recipe

```shell
SmoothQuantModifier(
    smoothing_strength=0.6,
),
AWQModifier(
    targets=["Linear"],
    ignore=ignore_modules,
    config_groups={
        "main": {
            "targets": ["Linear"],
            "weights": {
                "num_bits": 4,
                "type": "int",
                "symmetric": True,
                "strategy": "group",
                "group_size": 32,
                "observer": "mse",
                "dynamic": False,
            },
        }
    },
)
```

## Recommendation Content

- [Perplexity Score](https://github.com/Nicklas373/runpod-dockerize/blob/main/awq-quantize/Note/perplexity.md)
- [Quantization Note](https://github.com/Nicklas373/runpod-dockerize/blob/main/awq-quantize/Note/quantize_note.md)
- [Qwen3_5 Support](https://github.com/Nicklas373/runpod-dockerize/blob/main/awq-quantize/llmcompressor-patch/README.md)
- [SmoothQuant](https://github.com/Nicklas373/runpod-dockerize/blob/main/awq-quantize/SmoothQuant/README.md)

## How to use

- Deploy this Docker then Access SSH or using VSCode on Runpod with port 8080
- Exec **model_quantize.py** with this parameters (Examples)

```shell
python model_quantize.py --model_id "HUGGINGFACE/HUGGINGFACE_MODEL" --dataset_id DATASET1/YOUR_DATASET_1,DATASET2/YOUR_DATASET_2 --dataset_mix 0.5,0.5 --dataset_split train --text_column messages --num_samples 512 --max_seq_length 2048 --hf_cache False --branch main --trust_remote_code False --trust_remote_code_model False
```

- After quantization complete, before upload. You may **test models to make sure if it works at first**, run **model_eval.py** to test.

```shell
python model_eval.py --model_id "QUANTIZED_LOCAL_DIR" --trust_remote_code True
```

- To upload on HF as repo model, run **model_upload.py**

```shell
python3 model_upload.py --hf_token XXXX --repo_id YOUR_REPO_NAME --local_dir YOUR_REPO_LOCAL_DIR --repo_type YOUR_REPO_TYPE
```

## How to use on runpod

- Go to this URL template [llm-quantize-awq](https://console.runpod.io/deploy?template=5ik1p956nd&ref=xv2vjyqp)
- Deploy this template into runpod then Access SSH or using VSCode with port 8080
- Exec **model_quantize.py** with this parameters (Examples)

```shell
python model_quantize.py --model_id "HUGGINGFACE/HUGGINGFACE_MODEL" --dataset_id DATASET1/YOUR_DATASET_1,DATASET2/YOUR_DATASET_2 --dataset_mix 0.5,0.5 --dataset_split train --text_column messages --num_samples 512 --max_seq_length 2048 --hf_cache False --branch main --trust_remote_code False --trust_remote_code_model False
```

- After quantization complete, before upload. You may **test models to make sure if it works at first**, run **model_eval.py** to test.

```shell
python model_eval.py --model_id "QUANTIZED_LOCAL_DIR" --trust_remote_code True
```

- To upload on HF as repo model, run **model_upload.py**

```shell
python3 model_upload.py --hf_token XXXX --repo_id YOUR_REPO_NAME --local_dir YOUR_REPO_LOCAL_DIR --repo_type YOUR_REPO_TYPE
```

## Access

- 8080: VS Code Server

## Directory Structure

- /workspace/model_consolidated.py: Python based Llama model consolidation script
- /workspace/model_eval.py: Python based evaluate quantized model script
- /workspace/model_perplexity.py: Python based calculate perplexity score
- /workspace/model_quantize.py: Python based quantization script
- /workspace/model_upload.py: Python based upload to HF script
- /workspace/requirements.txt: Python requirements required library for mistral model

## Python package requirements

- accelerate
- causal-conv1d
- datasets
- huggingface-hub
- hf-transfer
- llmcompressor
- mamba-ssm
- transformers
