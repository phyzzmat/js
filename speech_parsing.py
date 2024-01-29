import speech_recognition as sr
from alsa_hack import noalsaerr
from utils import Quote


def parse_speech(init_rec):

    def is_float(x):
        return x.replace('.', '').isnumeric()

    def is_partial(words):
        return lambda x: x.replace('.', '') in words

    def check_pattern(text, pattern):
        if len(text) != len(pattern):
            return False
        for text, subpattern in zip(text, pattern):
            if not subpattern(text):
                return False
        return True

    ALLOWED_WORDS = [
        "bid", "ask", "for", "at", "up"
    ]

    with noalsaerr() as n, sr.Microphone() as source:
        print('Ready!')
        audio_data = init_rec.record(source, duration=5)
        print('Stop!')
        text = init_rec.recognize_whisper(
            audio_data,
            model='small.en',
            load_options={
                'device': 'cuda:0'
            },
            initial_prompt='Lexicon: bid, ask, at, for, numbers, up. Example: 10 for 6, 11 at 7. Another example:')
        tokenized_text = list(map(lambda x: x[:-1] if x.endswith('.') else x, filter(lambda word: is_float(word) or word.replace('.', '') in ALLOWED_WORDS, ''.join([
            i.lower() for i in text if i.isalnum() or i in ' .'
        ]).split())))
        print(text)
        if check_pattern(tokenized_text, [is_float, is_partial('at'), is_float, is_float, is_partial('up')]):
            return Quote(
                bid_amount=float(tokenized_text[3]),
                ask_amount=float(tokenized_text[3]),
                ask_price=float(tokenized_text[2]),
                bid_price=float(tokenized_text[0])
            )
        elif check_pattern(tokenized_text, [is_float, is_partial('bid'), is_partial('for,4'), is_float, is_float, is_partial('at'), is_float]):
            return Quote(
                bid_amount=float(tokenized_text[3]),
                ask_amount=float(tokenized_text[4]),
                ask_price=float(tokenized_text[6]),
                bid_price=float(tokenized_text[0])
            )
        else:
            print('Heard the following:', tokenized_text)
            print(
                'Failed to parse your speech; please repeat your quote more clearly. Tap Enter when ready.')
            return None
