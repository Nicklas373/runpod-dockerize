# Text Only Models

## General Models

```shell
ignore_modules = (
    "re:.*model.embed_tokens",
    "re:.*model.norm",
    "re:.*lm_head",
    #"re:.*embed_tokens.*",
    #"re:.*lm_head.*",
    #"re:.*norm.*",
)
```

## Qwen 3 Coder

```shell
ignore_modules = (
    "re:.*mlp\\.gate$",
    "re:.*shared_expert.*",
    "re:.*lm_head",
)
```

## Qwen 3.5

```shell
ignore_modules = (
    "re:.*embed_tokens",
    "re:.*lm_head",
    "re:mtp.*",
    "re:.*lm_head",
)
```

## Gemma 4

```shell
ignore_modules = (
    "re:.*lm_head",
)
```

# Vision Models

## General Models

```shell
ignore_modules = (
    "re:.*vision_encoder.*",
    "re:.*multi_modal_projector.*",
    "re:.*vision_tower.*",
     # "re:.*patch_conv.*", # apriel 1.6
)
```

## Gemma 4

```shell
ignore_modules = (
    "re:.*vision_embedder.*",
    "re:.*embed_vision.*",
    "re:.*audio_tower.*",
    "re:.*embed_audio.*",
)
```

## Qwen 3.5

```shell
ignore_modules = (
    "re:.*visual.*",
    "re:model[.]visual.*",
    "re:.*linear_attn.*",
)
```
