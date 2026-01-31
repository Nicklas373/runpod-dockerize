# Smooth Quantization Modifier

## Qwen 3 Specific

```shell
SmoothQuantModifier(
    smoothing_strength=0.8,
    mappings=[
        (
            ["re:.*q_proj$", "re:.*k_proj$", "re:.*v_proj$"],
            "re:.*input_layernorm$",
        ),
        (
            ["re:.*gate_proj$", "re:.*up_proj$"],
            "re:.*post_attention_layernorm$",
        ),
    ],
),
```
