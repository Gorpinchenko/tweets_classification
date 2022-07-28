import nltk
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from autocorrect import Speller

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

tknzr = TweetTokenizer(strip_handles=True, reduce_len=True)
stop_words = stopwords.words('english')
stammer = PorterStemmer()
lemmatizer = WordNetLemmatizer()
spell = Speller(lang='en')


def tokenize_tweet(tweet):
    tokens = tknzr.tokenize(tweet)
    return tokens


def remove_stop_words(tokens):
    return [token for token in tokens if token not in stop_words]


def stem_tokens(tokens):
    return [stammer.stem(token) for token in tokens]


def lemmatize_tokens(tokens):
    return [lemmatizer.lemmatize(token) for token in tokens]


def spell_tokens(tokens):
    return [spell(token) for token in tokens]


def preprocess(tweet, add_methods):
    data = tokenize_tweet(tweet)
    for method in add_methods:
        data = method(data)
    return data


def get_tokenizer(methods):
    def wrapper(tweet):
        return preprocess(tweet, methods)

    return wrapper
