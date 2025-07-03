import babelnet as bn
import json

synset_id = 'bn:00005054n'

synset = bn.get_synset(synset_id)

# Select only these key attributes (fast to fetch)
fields_to_fetch = [
    'id', 'pos', 'is_sense_lemma', 'lemma', 'lemmas', 'glosses', 'domains', 'examples'
]

data = {}
for attr in fields_to_fetch:
    try:
        val = getattr(synset, attr)
        # call if method without args
        if callable(val):
            val = val()
        # Convert complex lists or objects to strings or dicts
        if isinstance(val, (list, tuple)):
            val = [str(v) for v in val]
        else:
            val = str(val)
        data[attr] = val
    except Exception:
        data[attr] = 'Error fetching this attribute'

print(json.dumps(data, indent=2, ensure_ascii=False))