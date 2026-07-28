# Recorded intervention command/actual audit

The action stream is a position command, not the original VR target. All lag values minimize command-to-observation joint RMSE on moving samples.

## Valid arms

| episode | side | lag ms | q gap rms rad | command dq p99 | actual dq p99 | actual ddq p99 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 187 | right | 24.0 | 0.0386 | 2.89 | 2.55 | 22.5 |
| 187 | left | 24.0 | 0.0323 | 1.96 | 1.79 | 15.5 |
| 188 | right | 36.0 | 0.0193 | 0.88 | 0.75 | 6.1 |
| 188 | left | 28.0 | 0.0199 | 0.74 | 0.65 | 4.9 |
| 189 | right | 24.0 | 0.0327 | 1.84 | 1.66 | 13.5 |
| 189 | left | 24.0 | 0.0273 | 2.17 | 1.89 | 15.0 |
| 194 | left | 20.0 | 0.0406 | 3.31 | 2.84 | 41.3 |
| 196 | left | 48.0 | 0.5640 | 6.21 | 3.52 | 49.7 |
| 198 | left | 28.0 | 0.1402 | 5.72 | 3.50 | 56.7 |

## Frozen/reconnect arms

| episode | side | frozen joints | q gap rms rad |
| --- | --- | --- | ---: |
| 194 | right | 3,4,5,6,7 | 0.2624 |
| 196 | right | 3,4,5,6,7 | 0.5000 |
| 198 | right | 3,4,5,6,7 | 0.1493 |

## Aggregate

- Median valid arm lag: `24.0 ms`.
- Valid lag p90: `38.4 ms`.
- Episodes with frozen joints are excluded from controller tuning.
