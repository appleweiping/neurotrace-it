# Compute Budget

## First Gate Budget

| Item | Estimate | Buffer | Status |
| --- | ---: | ---: | --- |
| Activation extraction | 12 GPU hours | 30 percent | planned |
| Endpoint selection | 2 GPU/CPU hours | 30 percent | planned |
| Trajectory selection | 4 GPU/CPU hours | 30 percent | planned |
| Matched LoRA training | 24 GPU hours | 30 percent | planned |
| Storage | 500 GB raw activations | 30 percent | planned |

## Feasibility Rule

Raw activations stay server-side. If storage plus buffer is unavailable, the
run must reduce trajectory resolution before starting.

