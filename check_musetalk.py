#!/usr/bin/env python3
"""
Script de diagnostic pour vérifier l'installation de MuseTalk
"""
import os
import sys

print("=" * 60)
print("MuseTalk Installation Check")
print("=" * 60)

# 1. Vérifier le dossier MuseTalk
musetalk_path = '/app/MuseTalk'
print(f"\n1. Checking MuseTalk path: {musetalk_path}")
print(f"   Exists: {os.path.exists(musetalk_path)}")

if os.path.exists(musetalk_path):
    contents = os.listdir(musetalk_path)
    print(f"   Contents ({len(contents)} items):")
    for item in contents[:15]:  # First 15 items
        item_path = os.path.join(musetalk_path, item)
        item_type = "DIR" if os.path.isdir(item_path) else "FILE"
        print(f"     - [{item_type}] {item}")
    if len(contents) > 15:
        print(f"     ... and {len(contents) - 15} more items")

# 2. Vérifier sys.path
print(f"\n2. Python sys.path (first 5):")
sys.path.insert(0, musetalk_path)
for i, p in enumerate(sys.path[:5]):
    print(f"   {i}. {p}")

# 3. Tenter l'import
print(f"\n3. Attempting to import MuseTalk modules...")
try:
    from musetalk.utils.preprocessing import get_landmark_and_bbox
    print("   ✅ musetalk.utils.preprocessing imported successfully")
except ImportError as e:
    print(f"   ❌ Failed to import musetalk.utils.preprocessing")
    print(f"   Error: {e}")
    import traceback
    print(f"   Traceback:\n{traceback.format_exc()}")

try:
    from musetalk.utils.blending import get_image
    print("   ✅ musetalk.utils.blending imported successfully")
except ImportError as e:
    print(f"   ❌ Failed to import musetalk.utils.blending")
    print(f"   Error: {e}")

try:
    from musetalk.utils.utils import load_all_model
    print("   ✅ musetalk.utils.utils imported successfully")
except ImportError as e:
    print(f"   ❌ Failed to import musetalk.utils.utils")
    print(f"   Error: {e}")

print("\n" + "=" * 60)
print("Check complete")
print("=" * 60)
