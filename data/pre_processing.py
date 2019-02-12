import string
from copy import deepcopy

from nltk.corpus import stopwords
from nltk.probability import FreqDist
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.util import ngrams


def pre_process(ds):
    ds_pp = deepcopy(ds)
    wl = WordNetLemmatizer()
    for d in ds_pp:
        d['text_pp'] = d['text']
        d['text_pp'] = d['text_pp'].lower()
        d['text_pp'] = d['text_pp'].strip()
        d['text_pp'] = d['text_pp'].replace('<br /><br />', '')
        d['text_pp'] = d['text_pp'].translate(d['text_pp'].maketrans('', '', string.punctuation))
        d['text_pp'] = d['text_pp'].translate(d['text_pp'].maketrans('', '', string.digits))
        d['text_pp'] = word_tokenize(d['text_pp'], 'english')
        d['text_pp'] = list(filter(lambda x: x not in stopwords.words('english'), d['text_pp']))
        d['text_pp'] = list(map(wl.lemmatize, d['text_pp']))
        d['tf'] = FreqDist(d['text_pp'])
        d['bf'] = FreqDist(map(' '.join, ngrams(d['text_pp'], 2)))
    return ds_pp
