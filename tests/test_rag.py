from app.rag import _token_chunks


class CharacterTokenizer:
    def encode(self, text, add_special_tokens=False):
        return [ord(char) for char in text]

    def decode(self, ids, skip_special_tokens=True):
        return "".join(chr(token) for token in ids)


def test_chunking_uses_size_and_overlap():
    chunks = _token_chunks("abcdefghij", CharacterTokenizer(), size=5, overlap=2)
    assert chunks == ["abcde", "defgh", "ghij"]
