# Smooth Quantization Modifier

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
