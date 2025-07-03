import babelnet as bn
from babelnet import Language

for synset in bn.get_synsets('class', from_langs=[Language.EN]):
    print('Synset ID:', synset.id)