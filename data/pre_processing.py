import string

from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize


def preprocessor(s):
    s = s.lower()
    s = s.replace('<br /><br />', ' ')
    s = s.replace('-', ' ')
    s = s.replace('/', ' ')
    for ws in string.whitespace:
        s.replace(ws, ' ')
    s = s.translate(s.maketrans('', '', string.punctuation))
    s = s.translate(s.maketrans('', '', string.digits))
    s = ''.join(filter(lambda x: x in string.printable, s))
    return s


def tokenizer(s):
    wl = WordNetLemmatizer()
    st = SnowballStemmer('english', ignore_stopwords=True)
    ts = word_tokenize(s, 'english')
    ts = list(filter(lambda x: x not in stopwords.words('english'), ts))
    ts = list(map(lambda x: wl.lemmatize(x), ts))
    ts = list(map(lambda x: st.stem(x), ts))
    return ts
