# Smooth Quantization Modifier

## Qwen 3.5

```shell
 mappings=[
    {
        "smooth_layer": r"re:model.*layers[.](3|7|11|15|19|23|27|31)[.]input_layernorm",
        "balance_layers": [
            r"re:model.*self_attn.q_proj",
            r"re:model.*self_attn.k_proj",
            r"re:model.*self_attn.v_proj",
        ],
    },
    {
        "smooth_layer": r"re:model.*post_attention_layernorm",
        "balance_layers": [
            r"re:model.*mlp.gate_proj",
            r"re:model.*mlp.up_proj",
        ],
    },
],
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

# Qwen 3

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

# Gemma 4

```shell
    awq_mappings = []
    mods = dict(model.named_modules())
    num_layers = len(model.model.language_model.layers)
    for i in range(num_layers):
        prefix = f"model.language_model.layers.{i}"
        q = f"{prefix}.self_attn.q_proj"
        k = f"{prefix}.self_attn.k_proj"
        v = f"{prefix}.self_attn.v_proj"
        o = f"{prefix}.self_attn.o_proj"
        if all(x in mods for x in [q, k, v]):
            awq_mappings.append(
                AWQMapping(
                    smooth_layer=f"{prefix}.input_layernorm",
                    balance_layers=[q, k, v],
                )
            )

            if o in mods:
                awq_mappings.append(
                    AWQMapping(
                        smooth_layer=v,
                        balance_layers=[o],
                    )
                )

        gate = f"{prefix}.mlp.gate_proj"
        up = f"{prefix}.mlp.up_proj"
        down = f"{prefix}.mlp.down_proj"
        if all(x in mods for x in [gate, up]):
            awq_mappings.append(
                AWQMapping(
                    smooth_layer=f"{prefix}.pre_feedforward_layernorm",
                    balance_layers=[gate, up],
                )
            )

        if up in mods and down in mods:
            awq_mappings.append(
                AWQMapping(
                    smooth_layer=up,
                    balance_layers=[down],
                )
            )
```

# Qwen3-Coder-REAP-25B-A3B Family

```shell
    mods = dict(model.named_modules())
    awq_mappings = []
    num_layers = len(model.model.layers)
    for i in range(num_layers):
        prefix = f"model.layers.{i}"
        # ---------------- Attention ----------------
        q = f"{prefix}.self_attn.q_proj"
        k = f"{prefix}.self_attn.k_proj"
        v = f"{prefix}.self_attn.v_proj"
        o = f"{prefix}.self_attn.o_proj"
        awq_mappings.append(
            AWQMapping(
                smooth_layer=f"{prefix}.input_layernorm",
                balance_layers=[q, k, v],
            )
        )
        awq_mappings.append(
            AWQMapping(
                smooth_layer=v,
                balance_layers=[o],
            )
        )
        # ---------------- Experts ----------------
        expert = 0
        while True:
            gate = f"{prefix}.mlp.experts.{expert}.gate_proj"
            up = f"{prefix}.mlp.experts.{expert}.up_proj"
            down = f"{prefix}.mlp.experts.{expert}.down_proj"
            if gate not in mods:
                break
            awq_mappings.append(
                AWQMapping(
                    smooth_layer=f"{prefix}.post_attention_layernorm",
                    balance_layers=[gate, up],
                )
            )
            awq_mappings.append(
                AWQMapping(
                    smooth_layer=up,
                    balance_layers=[down],
                )
            )
            expert += 1
```

# Ornith 1.0-9B-AWQ

```shell
    awq_mappings = [
        {
            "smooth_layer": r"re:model\.language_model\.layers\.(3|7|11|15|19|23|27|31)\.input_layernorm",
            "balance_layers": [
                r"re:model\.language_model\.layers\.(3|7|11|15|19|23|27|31)\.self_attn\.q_proj",
                r"re:model\.language_model\.layers\.(3|7|11|15|19|23|27|31)\.self_attn\.k_proj",
                r"re:model\.language_model\.layers\.(3|7|11|15|19|23|27|31)\.self_attn\.v_proj",
            ],
        },
        {
            "smooth_layer": r"re:model\.language_model\.layers\.(0|1|2|4|5|6|8|9|10|12|13|14|16|17|18|20|21|22|24|25|26|28|29|30)\.input_layernorm",
            "balance_layers": [
                r"re:model\.language_model\.layers\.(0|1|2|4|5|6|8|9|10|12|13|14|16|17|18|20|21|22|24|25|26|28|29|30)\.linear_attn\.in_proj_qkv",
                r"re:model\.language_model\.layers\.(0|1|2|4|5|6|8|9|10|12|13|14|16|17|18|20|21|22|24|25|26|28|29|30)\.linear_attn\.in_proj_a",
                r"re:model\.language_model\.layers\.(0|1|2|4|5|6|8|9|10|12|13|14|16|17|18|20|21|22|24|25|26|28|29|30)\.linear_attn\.in_proj_b",
                r"re:model\.language_model\.layers\.(0|1|2|4|5|6|8|9|10|12|13|14|16|17|18|20|21|22|24|25|26|28|29|30)\.linear_attn\.in_proj_z",
            ],
        },
        {
            "smooth_layer": r"re:model\.language_model\.layers\.\d+\.post_attention_layernorm",
            "balance_layers": [
                r"re:model\.language_model\.layers\.\d+\.mlp\.gate_proj",
                r"re:model\.language_model\.layers\.\d+\.mlp\.up_proj",
            ],
        },
    ]
```
