import re

def split_to_paragraphs(raw_text, lang='english'):
    paragraphs = raw_text.split('\n')
    if lang == 'english':
        rcomp = r'[A-Za-z]+'
    elif lang == 'russian':
        rcomp = r'[А-яЁё]+'
    res_paragraphs = []
    for chunk in paragraphs:
        preprocessed_chunk = ' '.join(re.findall(rcomp, chunk))
        if len(preprocessed_chunk.split()) > 1:    # more than one word
            res_paragraphs.append(preprocessed_chunk)
    return res_paragraphs