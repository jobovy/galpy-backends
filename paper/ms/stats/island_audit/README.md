# Island audit (paper section `sec:process-islands`)

Where does galpy hand backend data to numpy when the existing suite runs under a
forced backend? See `plan/ISLAND_AUDIT.md` for the categories and the results.

    GALPY=<galpy checkout> bash run_audit.sh   # per-file / per-CI-group pytest runs with -p island_audit
    python merge.py torch                      # merged_torch.json + sites by test count
    python classify.py torch                   # tiers (the paper's numbers) + island functions
    python groups.py torch                     # island families

Paths inside the scripts point at /var/tmp/claude-probes/audit (the run directory).
Use torch: numpy reads CPU jax arrays through the buffer protocol, which the
`__array__` hook cannot see.
