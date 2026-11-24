# Install EXP 6.1 integration bundle

Upload the ZIP to WTC9 and run:

```bash
cd /data/Kunal/FORMAL_EXP6_ECAPA_MULTITASK
unzip -o /home/wtc9/EXP61_INTEGRATION_BUNDLE.zip

chmod +x \
  scripts/validation/08_validate_exp61_integration.sh \
  scripts/launch/10_launch_exp61_gpu1_queue.sh \
  scripts/launch/11_status_exp61_gpu1.sh \
  scripts/launch/12_attach_exp61_gpu1.sh \
  scripts/launch/13_stop_exp61_gpu1.sh

bash scripts/validation/08_validate_exp61_integration.sh
```

Expected validation ending:

```text
EXP61_SYNTHETIC_INTEGRATION_PASS
EXP61_REAL_CACHE_PREFLIGHT_PASS
EXP61_INTEGRATION_VALIDATION_PASS
```

Do not launch training until the validation passes and physical GPU 1 is free.
