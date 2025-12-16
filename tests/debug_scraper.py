import streetview
import numpy as np
import os
from PIL import Image


def get_panoid(pano_obj):
    """Safe way to get panoid regardless of object type (Dict or Object)"""
    # Option 1: It's an object (Your case)
    if hasattr(pano_obj, "panoid"):
        return pano_obj.panoid
    # Option 2: It's a dictionary (Older versions)
    try:
        return pano_obj["panoid"]
    except (TypeError, KeyError):
        return None


# 1. Target: University Dr NW (Known good location)
lat, lon = 51.0782, -114.1360
print(f"[DEBUG] Searching for panoramas at {lat}, {lon}...")

# 2. Test Search
panos = streetview.search_panoramas(lat, lon)

if not panos:
    print(
        "[FAIL] search_panoramas returned empty list. Google API might be blocking or changed."
    )
else:
    print(f"[SUCCESS] Found {len(panos)} panoramas.")

    # 3. Test Download of Top 3
    for i, meta in enumerate(panos[:3]):
        panoid = get_panoid(meta)

        if not panoid:
            print(f"   [SKIP] Could not extract panoid from object: {type(meta)}")
            continue

        print(f"\n--- Checking Pano #{i+1} (ID: {panoid[:10]}...) ---")

        try:
            img = streetview.get_panorama(panoid)
            if img:
                # Calculate Stats
                arr = np.array(img.convert("L"))
                mean_brightness = np.mean(arr)
                std_contrast = np.std(arr)

                print(f"   [STATS] Brightness (Mean): {mean_brightness:.2f}")
                print(f"   [STATS] Contrast (StdDev): {std_contrast:.2f}")

                # Save it so you can see if it's actually black
                os.makedirs("tests/output", exist_ok=True)
                out_path = f"tests/output/debug_pano_{i}.jpg"
                img.save(out_path)
                print(f"   [SAVED] Saved to {out_path}")

                # Check against RELAXED thresholds
                if mean_brightness < 5 or std_contrast < 5:
                    print("   [RESULT] REJECTED (Too Dark/Flat)")
                else:
                    print("   [RESULT] ACCEPTED")
            else:
                print("   [FAIL] Download returned None.")

        except Exception as e:
            print(f"   [ERROR] {e}")
