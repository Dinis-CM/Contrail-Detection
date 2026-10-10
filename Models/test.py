import os
import h5py
import numpy as np

# Global trackers across all files
global_stats = {
    "t11": [float("inf"), float("-inf")],
    "t14": [float("inf"), float("-inf")],
    "t15": [float("inf"), float("-inf")],
    "btd_15_14": [float("inf"), float("-inf")],
    "btd_14_11": [float("inf"), float("-inf")],
}

chunk_size = 200  # 200 records at a time (~78 MB RAM)

for train_file in train_files:
    file_name = os.path.basename(train_file)
    print(f"\nProcessing File: {file_name}")

    with h5py.File(train_file, "r") as f:
        bands_dset = f["bands"]
        scale = float(bands_dset.attrs["scale"])
        offset = float(bands_dset.attrs["offset"])
        num_records = bands_dset.shape[0]

        file_t11_min, file_t11_max = float("inf"), float("-inf")
        file_t14_min, file_t14_max = float("inf"), float("-inf")
        file_t15_min, file_t15_max = float("inf"), float("-inf")
        file_btd1514_min, file_btd1514_max = float("inf"), float("-inf")
        file_btd1411_min, file_btd1411_max = float("inf"), float("-inf")

        for start_idx in range(0, num_records, chunk_size):
            end_idx = min(start_idx + chunk_size, num_records)
            raw = bands_dset[start_idx:end_idx]
            kelvin = raw.astype(np.float32) * scale + offset

            t11 = kelvin[:, 0]
            t14 = kelvin[:, 1]
            t15 = kelvin[:, 2]
            btd_15_14 = t15 - t14
            btd_14_11 = t14 - t11

            # Update per-file bounds
            file_t11_min = min(file_t11_min, t11.min())
            file_t11_max = max(file_t11_max, t11.max())
            file_t14_min = min(file_t14_min, t14.min())
            file_t14_max = max(file_t14_max, t14.max())
            file_t15_min = min(file_t15_min, t15.min())
            file_t15_max = max(file_t15_max, t15.max())

            file_btd1514_min = min(file_btd1514_min, btd_15_14.min())
            file_btd1514_max = max(file_btd1514_max, btd_15_14.max())
            file_btd1411_min = min(file_btd1411_min, btd_14_11.min())
            file_btd1411_max = max(file_btd1411_max, btd_14_11.max())

        print(f"  Band 11 (8.4 µm)  min: {file_t11_min:.1f} K, max: {file_t11_max:.1f} K")
        print(f"  Band 14 (11.2 µm) min: {file_t14_min:.1f} K, max: {file_t14_max:.1f} K")
        print(f"  Band 15 (12.3 µm) min: {file_t15_min:.1f} K, max: {file_t15_max:.1f} K")
        print(f"  BTD 15-14         min: {file_btd1514_min:.1f} K, max: {file_btd1514_max:.1f} K")
        print(f"  BTD 14-11         min: {file_btd1411_min:.1f} K, max: {file_btd1411_max:.1f} K")

        # Update global bounds
        global_stats["t11"][0] = min(global_stats["t11"][0], file_t11_min)
        global_stats["t11"][1] = max(global_stats["t11"][1], file_t11_max)
        global_stats["t14"][0] = min(global_stats["t14"][0], file_t14_min)
        global_stats["t14"][1] = max(global_stats["t14"][1], file_t14_max)
        global_stats["t15"][0] = min(global_stats["t15"][0], file_t15_min)
        global_stats["t15"][1] = max(global_stats["t15"][1], file_t15_max)
        global_stats["btd_15_14"][0] = min(global_stats["btd_15_14"][0], file_btd1514_min)
        global_stats["btd_15_14"][1] = max(global_stats["btd_15_14"][1], file_btd1514_max)
        global_stats["btd_14_11"][0] = min(global_stats["btd_14_11"][0], file_btd1411_min)
        global_stats["btd_14_11"][1] = max(global_stats["btd_14_11"][1], file_btd1411_max)

print("\n" + "=" * 50)
print("DATASET-WIDE SUMMARY ACROSS ALL SHARDS")
print("=" * 50)
print(f"Band 11 (8.4 µm)  : [{global_stats['t11'][0]:.1f} K, {global_stats['t11'][1]:.1f} K]")
print(f"Band 14 (11.2 µm) : [{global_stats['t14'][0]:.1f} K, {global_stats['t14'][1]:.1f} K]")
print(f"Band 15 (12.3 µm) : [{global_stats['t15'][0]:.1f} K, {global_stats['t15'][1]:.1f} K]")
print(f"BTD 15-14         : [{global_stats['btd_15_14'][0]:.1f} K, {global_stats['btd_15_14'][1]:.1f} K]")
print(f"BTD 14-11         : [{global_stats['btd_14_11'][0]:.1f} K, {global_stats['btd_14_11'][1]:.1f} K]")