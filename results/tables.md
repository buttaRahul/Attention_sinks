### Experiment 1: perplexity by position range (exp of mean NLL)

| config | 0K–1K | 1K–2K | 2K–8K | 8K–32K | 32K–40K | 40K–64K |
|---|---:|---:|---:|---:|---:|---:|
| full cache | 17.61 | 20.56 | 22.21 | 20.72 | 19.95 | 26.65 |
| window only (1,024) | 17.61 | 288.33 | 294.27 | 294.79 | 306.03 | 551.68 |
| 4 sinks + 1,020 window | 17.61 | 21.06 | 23.53 | 21.83 | 21.95 | 34.06 |

### Headline: perplexity relative to the full cache, over the range the full cache survived

| range | full | sinks | sinks vs full | window only | window vs full |
|---|---:|---:|---:|---:|---:|
| 1,024–50,176 | 21.77 | 22.92 | +5.3% | 313.0 | 14.4× |
| 8,192–32,768 | 20.72 | 21.83 | +5.4% | 294.8 | 14.2× |
| 32,768–50,176 | 23.25 | 24.44 | +5.1% | 349.8 | 15.0× |

### Ablation: sink count

| sinks (window 1,020) | ppl 2K–16K | bf16 keys |
|---:|---:|---:|
| 0 | 312.96 | 654.14 |
| 1 | 23.34 | 31.69 |
| 2 | 23.35 | 31.69 |
| 4 | 23.29 | 31.81 |
| 8 | 23.40 | 32.21 |
| full cache | 22.44 | |

### Ablation: cache size

| cache size | 4 sinks + window | window only | 4 sinks, bf16 keys | window only, bf16 keys |
|---:|---:|---:|---:|---:|
| 256 | 25.55 | 613.16 | 25.46 | 597.77 |
| 512 | 24.17 | 430.43 | 23.66 | 369.25 |
| 1,024 | 23.29 | 312.69 | 31.81 | 676.94 |
| 2,048 | 23.24 | 259.08 | 62.02 | 8299.24 |

### Ablation: position handling and key precision

| variant (4 + 1,020) | ppl 2K–16K |
|---|---:|
| 4 sinks + 1,020 window | 23.292 |
| 4 + 1,020, bf16 keys | 31.805 |
| 4 + 1,020, no re-rotation (bf16 keys) | 47641.953 |

### Experiment 2: allocated memory (MiB) after N tokens

| tokens | full cache | window only (1,024) | 4 sinks + 1,020 window |
|---:|---:|---:|---:|
| 1,024 | 1,258 | 1,370 | 1,370 |
| 4,096 | 1,594 | 1,378 | 1,378 |
| 16,384 | 2,938 | 1,378 | 1,378 |
| 32,768 | 4,730 | 1,378 | 1,378 |
| 40,960 | 5,626 | 1,378 | 1,378 |
| 49,152 | 6,522 | 1,378 | 1,378 |
| 65,536 | – | – | – |

Measured full-cache growth: **112.0 KiB/token** (theory 112.0 KiB/token).

### Experiment 3: decode latency (3 rounds, medians)

| config | context | GPU ms/token | range | linear | attention | cache copies | other | eager ms | graph ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| full cache | 1,024 | 20.75 | 20.11–23.93 | 10.41 | 1.73 | 3.10 | 5.51 | 86.7 | – |
| full cache | 2,048 | 24.60 | 22.35–26.52 | 11.65 | 2.96 | 4.19 | 5.80 | 91.7 | – |
| full cache | 4,096 | 23.85 | 22.39–26.82 | 9.45 | 3.51 | 6.13 | 4.76 | 80.4 | – |
| full cache | 8,192 | 27.50 | 27.48–28.87 | 7.98 | 5.02 | 10.60 | 3.90 | 82.8 | – |
| full cache | 16,384 | 36.82 | 36.79–40.59 | 6.98 | 10.15 | 15.83 | 3.89 | 96.6 | – |
| full cache | 32,768 | 62.23 | 61.63–62.46 | 6.96 | 18.30 | 33.09 | 3.88 | 78.1 | – |
| 4 sinks + 1,020 window | 1,024 | 22.48 | 19.43–28.18 | 9.86 | 1.61 | 5.85 | 5.15 | 78.9 | 15.3 |
| 4 sinks + 1,020 window | 2,048 | 23.15 | 20.85–29.69 | 10.16 | 1.69 | 6.00 | 5.29 | 86.2 | 15.5 |
| 4 sinks + 1,020 window | 4,096 | 27.55 | 20.74–30.39 | 12.57 | 2.06 | 6.66 | 6.26 | 105.9 | 15.5 |
| 4 sinks + 1,020 window | 8,192 | 23.22 | 21.13–34.39 | 10.37 | 1.67 | 5.95 | 5.23 | 91.1 | 15.7 |
| 4 sinks + 1,020 window | 16,384 | 27.76 | 21.51–30.21 | 12.53 | 2.09 | 6.83 | 6.32 | 100.2 | 15.5 |
| 4 sinks + 1,020 window | 32,768 | 25.06 | 23.55–27.54 | 11.16 | 1.86 | 6.31 | 5.73 | 89.4 | 15.3 |

### Experiment 4: long chat and planted-fact recall

| config | tokens seen | probe turn 3: answer | log p(4417) | probe turn 50: answer | log p(4417) | replies ended cleanly | mean distinct-2 |
|---|---:|---|---:|---|---:|---:|---:|
| full cache | 4,654 | `4417` | -0.00 | `4417` | -0.01 | 100% | 0.937 |
| window only (1,024) | 2,035 | `4417` | -0.00 | `` | -29.92 | 98% | 0.195 |
| 4 sinks + 1,020 window | 5,668 | `4417` | -0.00 | `123456789` | -13.71 | 88% | 0.900 |

### Second model: Llama-3.2-1B perplexity (16K tokens)

| config | 0–1K | 1K–2K | 2K–16K |
|---|---:|---:|---:|
| full cache | 9.23 | 9.48 | 9.73 |
| window only (1,024) | 9.23 | 132.60 | 177.28 |
| 4 sinks + 1,020 window | 9.23 | 9.52 | 10.45 |
