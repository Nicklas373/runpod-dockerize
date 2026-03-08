# AWQ Modifier Recipe

```shell
recipe = [
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
                        "group_size": 32, # Increase it to 64 for <= 7B and 128 for <= 4B
                        "observer": "mse",
                        "dynamic": False,
                    },
                }
            },
        )
    ]
```
