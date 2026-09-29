# WINDOWS HARDWARE REPORT

_Generated 2026-09-29 10:03:12 on **Linux (NOT the target laptop — sandbox/dev numbers; re-run on Windows before Gate 5)._

## Hardware inventory

| item | value |
|---|---|
| cpu | Intel(R) Xeon(R) Processor |
| physical_cores | 1 |
| logical_cores | 2 |
| ram_gb | 4.1 |
| gpu | None |
| gpu_vram_gb | None |
| os_version | #1 SMP Thu Jul 30 12:54:59 UTC 2026 |

## Pipeline benchmark (synthetic drive)

- runtime: `mock`, chunk 50.0 ms
- **capture_to_detect**: P50 23.28 ms · P95 39.51 ms · max 40.74 ms (n=96)
- **detector_only**: P50 4.05 ms · P95 5.36 ms · max 7.5 ms (n=96)
- **brain_chunk_wall**: P50 0.0 ms · P95 0.0 ms · max 0.0 ms (n=52)
- **brain_to_action**: P50 0.06 ms · P95 0.23 ms · max 0.41 ms (n=52)
- **full_sensory_to_action**: P50 49.52 ms · P95 92.03 ms · max 99.76 ms (n=96)

- achieved rates: `{'fast_vision': 24.5, 'brain_worker': 13.0, 'motor_executor': 51.5}`
- brain throughput: 0.0 bio-s per wall-s

## Canonical brain benchmark

- init (once): **2.36 s**
- chunk 50ms: P50 501.69 ms · P95 508.81 ms (n=20)
- chunk 100ms: P50 779.16 ms · P95 784.37 ms (n=20)

## Auto-chosen conservative live settings

```json
{
 "derived_from": "measured",
 "margin": 0.7,
 "capture_fps": 30,
 "fast_vision_hz": 24,
 "heavy_vision_hz": 8,
 "brain_hz": 7.0,
 "brain_chunk_ms": 100,
 "motor_executor_hz": 40,
 "planner_hz": 1.0,
 "dashboard_hz": 15,
 "measured_s2a_p95_ms": 92.0,
 "s2a_target_ms": 250.0,
 "s2a_target_met": true,
 "canonical_50ms_p95_ms": 508.81,
 "canonical_100ms_p95_ms": 784.37
}
```

Saved to `config/live_settings.json`.
