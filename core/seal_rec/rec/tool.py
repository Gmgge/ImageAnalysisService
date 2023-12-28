import json


def read_vocab(path):
    """
    加载词典
    """
    with open(path, encoding='utf-8') as f:
        vocab = json.load(f)
    return vocab


def decode_text(tokens, vocab, vocab_inp):
    """
    decode trocr
    """
    s_start = vocab.get('<s>')
    s_end = vocab.get('</s>')
    unk = vocab.get('<unk>')
    pad = vocab.get('<pad>')
    text = ''
    for tk in tokens:

        if tk == s_end:
            break
        if tk not in [s_end, s_start, pad, unk]:
            text += vocab_inp[tk]
    return text
