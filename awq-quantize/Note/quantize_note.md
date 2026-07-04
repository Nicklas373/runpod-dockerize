### Model Recipe Instructions

- group_size
  - For 12 > use 32
  - For 12 < use 64
  - For 4 < use 128
- smooth_quant
  - For 12 > use 0.6
  - For 12 < use 0.55 or disable it
  - For 4 < disable it

### Model Quantization Instructions

- num_samples (Use 512 samples if possible, if not then use 256 at least)
- dataset_mix (Use 0.7,0.3 (70% wikitext, 30% c4))

### Model Evaluation Command

```bash
python model_eval.py --model_id "YOUR_HUGGINGFACE_ACCOUNT/MODEL_NAME" \
                     --trust_remote_code True
```

### Model Perplexity Command

```bash
python model_perplexity.py --model_id "YOUR_HUGGINGFACE_ACCOUNT/MODEL_NAME" --max_blocks 20
```

### Model Quantization Command

```bash
python model_quantize.py --model_id "YOUR_HUGGINGFACE_ACCOUNT/MODEL_NAME" \
                         --dataset_id HuggingFaceH4/ultrachat_200k,Salesforce/wikitext \
                         --dataset_config ,wikitext-103-raw-v1 \
                         --dataset_split train_sft,train \
                         --dataset_mix 0.7,0.3 \
                         --max_seq_length 2048 \
                         --num_samples 512 \
                         --trust_remote_code True \
                         --trust_remote_code_model True
```

### Model Upload Command

```bash
python model_upload.py --hf_token xxx \
                       --repo_id YOUR_HUGGINGFACE_ACCOUNT/MODEL_NAME \
                       --local_dir MODEL_DIR \
                       --repo_type model
```

### Model Generate Safetensors Index

```bash
python model_generate_safetensors.py --model_dir MODEL_DIR
```
