# GuardrailGym leaderboard

| System | GG-Score | core TPR | ASR | benign over-block | hard-neg over-block | p95 ms |
|---|---|---|---|---|---|---|
| tfidf-lr-v1 @OB<=5% | 0.772 | 1.000 | 0.000 | 0.746 | 0.785 | 0.9 |
| keyword-v1 @OB<=5% | 0.566 | 0.345 | 0.701 | 0.071 | 0.000 | 0.3 |
| CONTROL-length-stump @OB<=5% | 0.537 | 0.233 | 0.703 | 0.050 | 0.000 | 0.0 |
| incumbent-replay (PARTIAL: core only) | 0.497 | 1.000 | — | — | — | 6881.2 |

## Attack success rate by family

### tfidf-lr-v1 @OB<=5%

| family | ASR | n |
|---|---|---|
| `fictional_framing` | 0.000 | 81 |
| `romanized_l2` | 0.000 | 86 |
| `authority_pretext` | 0.000 | 85 |
| `incremental_split` | 0.000 | 86 |
| `encoding_obfuscation` | 0.000 | 82 |
| `indirect_injection` | 0.000 | 85 |
| `output_only` | 0.000 | 81 |

### keyword-v1 @OB<=5%

| family | ASR | n |
|---|---|---|
| `encoding_obfuscation` | 0.817 | 82 |
| `romanized_l2` | 0.814 | 86 |
| `output_only` | 0.765 | 81 |
| `indirect_injection` | 0.659 | 85 |
| `authority_pretext` | 0.635 | 85 |
| `fictional_framing` | 0.617 | 81 |
| `incremental_split` | 0.605 | 86 |

### CONTROL-length-stump @OB<=5%

| family | ASR | n |
|---|---|---|
| `fictional_framing` | 1.000 | 81 |
| `romanized_l2` | 1.000 | 86 |
| `authority_pretext` | 1.000 | 85 |
| `incremental_split` | 1.000 | 86 |
| `encoding_obfuscation` | 0.902 | 82 |
| `indirect_injection` | 0.000 | 85 |
| `output_only` | 0.000 | 81 |
