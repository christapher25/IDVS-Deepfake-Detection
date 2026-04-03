import tensorflow as tf
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # Hide mess, show errors

print("\n" + "="*40)
print("   HARDWARE INSPECTION")
print("="*40)

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"✅ SUCCESS! Found {len(gpus)} GPU(s)")
    for gpu in gpus:
        print(f"   🚀 Device: {gpu.name}")
        details = tf.config.experimental.get_device_details(gpu)
        print(f"   💾 Name: {details.get('device_name', 'Unknown')}")
else:
    print("❌ WARNING: No GPU detected. You are running on CPU.")

print("="*40 + "\n")