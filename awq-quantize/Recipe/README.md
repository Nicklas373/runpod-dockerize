# AWQ Modifier Recipe

## >= 7B

```shell
recipe = [
        AWQModifier(
            targets=["Linear"],
            ignore=ignore_modules,
            config_groups={
                "channel_sensitive": {
                    "targets": [
                        're:.*q_proj$',
                        're:.*k_proj$',
                        're:.*v_proj$'
                    ],
                    "weights": {
                        "num_bits": 4,
                        "type": "int",
                        "symmetric": True,
                        "strategy": "channel",
                        "observer": "mse",
                        "dynamic": False,
                    },
                },
                "group_0": {
                    "targets": [
                        "re:.*down_proj$",
                        "re:.*o_proj$",
                        "re:.*gate_proj$",
                        "re:.*up_proj$"
                    ],
                    "weights": {
                        "num_bits": 4,
                        "type": "int",
                        "symmetric": True,
                        "strategy": "group",
                        "group_size": 64, # Should change to 32 for >= 12B
                        "observer": "mse",
                        "dynamic": False,
                    },
                }
            },
        )
    ]
```

## <= 4B

```shell
recipe = [
    AWQModifier(
        targets=["Linear"],
        ignore=ignore_modules,
        config_groups={
            "group_0": {
                "targets": ["Linear"],
                "weights": {
                    "num_bits": 4,
                    "type": "int",
                    "symmetric": True,
                    "strategy": "group",
                    "group_size": 128,
                    "observer": "mse",
                    "dynamic": False,
                },
            },
        },
    )
]
```
