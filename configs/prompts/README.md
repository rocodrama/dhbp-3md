# Active Prompt Set

This is the active English prompt root used by `PromptBuilder`.

```text
02_CODE/configs/prompts/
+-- prompts/
|   +-- system.md
|   +-- medical.md
|   +-- output_format.md
|   +-- probe_format.md
+-- personas/
    +-- common.md
    +-- placebo.md
    +-- <16 type files>.md
```

## Control Logic

`BASE` receives only the shared medical/task instructions. `PLACEBO` receives the same common wrapper and the same six-section structure as the 16 conditioned arms, but without directional cognitive content. This lets the experiment separate:

```text
PLACEBO - BASE    = effect of adding similarly sized profile text
MBTI    - PLACEBO = effect of directional conditioning
```

The important control is context length. The placebo and the 16 conditioned files are deliberately similar in size, and tests enforce a 15 percent word-count band around `placebo.md`.

## Prompt Hygiene

Individual files must not name their own labels, mention the conditioning scheme, or include medical hints. They should describe general habits of attention, uncertainty, sequencing, and communication only. Medical accuracy and answer format are controlled by the shared prompt files.
