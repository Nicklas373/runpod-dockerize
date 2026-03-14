# Smooth Quantization Modifier

## Qwen 3.5

```shell
 awq_mappings = []

    for i in range(32):
    # Linear Attn: qkv -> out
    awq_mappings.append(
        AWQMapping(
            smooth_layer=f"model.language_model.layers.{i}.linear_attn.in_proj_qkv.weight",
            balance_layers=[f"model.language_model.layers.{i}.linear_attn.out_proj.weight"],
        )
    )

    # MLP: down -> up
    awq_mappings.append(
        AWQMapping(
            smooth_layer=f"model.language_model.layers.{i}.mlp.down_proj",
            balance_layers=[f"model.language_model.layers.{i}.mlp.up_proj"],
        )
    )

    # MLP gate -> up
    awq_mappings.append(
        AWQMapping(
            smooth_layer=f"model.language_model.layers.{i}.mlp.gate_proj",
            balance_layers=[f"model.language_model.layers.{i}.mlp.up_proj"],
        )
    )
```

## Apriel 1.6

```shell
    awq_mappings = [
        AWQMapping(
            smooth_layer="re:model.language_model.layers.*.input_layernorm",
            balance_layers=[
                "re:model.language_model.layers.*.q_proj",
                "re:model.language_model.layers.*.k_proj",
                "re:model.language_model.layers.*.v_proj",
            ],
        ),
        AWQMapping(
            smooth_layer="re:model.language_model.layers.*.post_attention_layernorm",
            balance_layers=[
                "re:model.language_model.layers.*.gate_proj",
                "re:model.language_model.layers.*.up_proj",
            ],
        ),
    ]
```

## Nemotron Nano v2

```shell
## Should be disabled for parameter below than 14B
SmoothQuantModifier(
    smoothing_strength=0.6,
    mappings=[
        [
            "re:model\\.backbone\\.layers\\.\\d+\\.norm$",
            [
                "re:model\\.backbone\\.layers\\.\\d+\\.mixer\\.q_proj$",
                "re:model\\.backbone\\.layers\\.\\d+\\.mixer\\.k_proj$",
                "re:model\\.backbone\\.layers\\.\\d+\\.mixer\\.v_proj$",
            ],
        ],
        [
            "re:model\\.backbone\\.layers\\.\\d+\\.mixer\\.v_proj$",
            [
                "re:model\\.backbone\\.layers\\.\\d+\\.mixer\\.o_proj$",
            ],
        ],
        [
            "re:model\\.backbone\\.layers\\.\\d+\\.norm$",
            [
                "re:model\\.backbone\\.layers\\.\\d+\\.mixer\\.up_proj$",
            ],
        ],
        [
            "re:model\\.backbone\\.layers\\.\\d+\\.mixer\\.up_proj$",
            [
                "re:model\\.backbone\\.layers\\.\\d+\\.mixer\\.down_proj$",
            ],
        ],
    ],
),
```

## Additional on AWQModifier for Nemotron Nano v2

```shell
mappings=[
    {
        "smooth_layer": r"re:model\.backbone\.layers\.\d+\.norm$",
        "balance_layers": [
            r"re:model\.backbone\.layers\.\d+\.mixer\.q_proj$",
            r"re:model\.backbone\.layers\.\d+\.mixer\.k_proj$",
            r"re:model\.backbone\.layers\.\d+\.mixer\.v_proj$",
        ],
    },
    {
        "smooth_layer": r"re:model\.backbone\.layers\.\d+\.mixer\.v_proj$",
        "balance_layers": [
            r"re:model\.backbone\.layers\.\d+\.mixer\.o_proj$",
        ],
    },
    {
        "smooth_layer": r"re:model\.backbone\.layers\.\d+\.norm$",
        "balance_layers": [
            r"re:model\.backbone\.layers\.\d+\.mixer\.up_proj$",
        ],
    },
    {
        "smooth_layer": r"re:model\.backbone\.layers\.\d+\.mixer\.up_proj$",
        "balance_layers": [
            r"re:model\.backbone\.layers\.\d+\.mixer\.down_proj$",
        ],
    },
]
```
