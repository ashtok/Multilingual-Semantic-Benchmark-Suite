# Minimal test script to isolate the issue
import sys
import os

print("🧪 Minimal BabelNet Test")
print(f"Python version: {sys.version}")
print(f"Working directory: {os.getcwd()}")

# Test 1: Import
try:
    import babelnet as bn
    from babelnet import BabelSynsetID, Language

    print("✅ BabelNet modules imported successfully")
except ImportError as e:
    print(f"❌ Import failed: {e}")
    sys.exit(1)

# Test 2: API Key
api_key = os.environ.get('BABELNET_KEY')
if api_key:
    print(f"✅ API Key found (length: {len(api_key)})")
else:
    print("❌ No BABELNET_KEY environment variable found")
    print("Set it with: export BABELNET_KEY='your_key_here'")

# Test 3: Simple BabelNet call
try:
    print("\n🔍 Testing BabelNet API call...")
    synset = bn.get_synset(BabelSynsetID('bn:00015267n'))
    print(f"✅ Successfully retrieved synset: {synset}")

    # Test getting English lemma
    if synset.main_sense(Language.EN):
        lemma = synset.main_sense(Language.EN).full_lemma
        print(f"✅ English lemma: {lemma}")
    else:
        print("❌ No English lemma found")

except Exception as e:
    print(f"❌ BabelNet API call failed: {e}")
    import traceback

    traceback.print_exc()

print("\n✅ Minimal test completed!")