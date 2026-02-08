import math

def cosine(a, b):
    dot = sum(x*y for x, y in zip(a, b))
    na = math.sqrt(sum(x*x for x in a))
    nb = math.sqrt(sum(x*x for x in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)

def test_cosine_high_for_similar_vectors():
    a = [1, 0, 1, 0]
    b = [1, 0, 1, 0]
    assert cosine(a, b) > 0.99

def test_cosine_low_for_different_vectors():
    a = [1, 0, 0, 0]
    b = [0, 1, 0, 0]
    assert cosine(a, b) < 0.1
