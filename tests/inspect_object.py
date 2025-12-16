import streetview

# Target: University Dr NW
lat, lon = 51.0782, -114.1360
print(f"Searching at {lat}, {lon}...")

panos = streetview.search_panoramas(lat, lon)

if panos:
    p = panos[0]
    print(f"\n[INFO] Object Type: {type(p)}")
    print(f"[INFO] Available Attributes:\n{dir(p)}")
else:
    print("[FAIL] No panos found.")
