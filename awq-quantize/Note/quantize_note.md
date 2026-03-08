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
python model_eval.py --model_id "xxx" \
                     --trust_remote_code True
```

### Model Perplexity Command

```bash
python model_perplexity.py --model_id "xxx"
```

### Model Quantization Command

```bash
python model_quantize.py --model_id "xxx/xxx" \
                         --dataset_id Salesforce/wikitext,allenai/c4 \
                         --dataset_config wikitext-2-raw-v1,en \
                         --dataset_mix 0.7,0.3 \
                         --dataset_split train \
                         --num_samples 256 \
                         --max_seq_length 2048 \
                         --trust_remote_code True \
                         --trust_remote_code_model True
```

### Model Upload Command

```bash
python model_upload.py --hf_token xxx \
                       --repo_id xxx/xxx \
                       --local_dir xxx \
                       --repo_type model
```
