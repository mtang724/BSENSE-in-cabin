# Range Doppler Baseline

Run the range-doppler BPM detection by running from the base BSense directory:

```bash 
python /baseline/doppler/main.py
```

It will calculate:

* **Ground truth BPM**
* **Estimated BPM** by finding the peak in range-doppler

It will produce two plots under current working directory:

* `bsense_baseline_doppler_result.png`: visualize the peak in range doppler
* `bsense_baseline_gt_peaks.png`: visualize peaks in the ground truth respiration wave