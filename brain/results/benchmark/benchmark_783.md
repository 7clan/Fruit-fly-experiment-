# D3 benchmark — FlyWire v783 (public release, 138,639 neurons) - CANONICAL

- Machine: Intel(R) Xeon(R) Processor — 2 cores, 4041.6 MB RAM (Linux-5.10.134-013.15.kangaroo.al8.x86_64-x86_64-with-glibc2.41)
- Stack: Python 3.12.14, Brian2 2.10.1, NumPy 2.5.3, pandas 3.0.6
- Neurons: **138,639**
- Connections (synapse rows): **15,091,983**
- Data load (pandas): 0.72 s
- Network build per process: [1.425, 1.381, 1.431] s
- Full init per fresh process (mean): **10.26 s**
- Peak RSS per process: **2915.8 MB** (max 2921.5 MB)
- Simulation speed: **0.14845 bio-s per wall-s** (stimulated network (sugar set, 150 Hz), codegen cache warm)
- Spikes/trial: [13309, 13011, 12741]
- Active neurons/trial: [368, 370, 370]
