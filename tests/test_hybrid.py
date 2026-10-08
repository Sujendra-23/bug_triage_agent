from agent.hybrid import bm25_rank, rrf_fuse, tokenize


def test_tokenize_keeps_identifiers_and_their_parts():
    toks = tokenize("KeyError: PAYMENT_GATEWAY_URL in v2.3.1")
    assert "payment_gateway_url" in toks
    assert "gateway" in toks
    assert "v2.3.1" in toks


def test_rrf_rewards_agreement_between_rankers():
    fused = rrf_fuse([["a", "b", "c"], ["c", "a", "d"]])
    assert fused[0] == "a"  # ranks 1 and 2 beats c's 3 and 1
    assert set(fused) == {"a", "b", "c", "d"}


def test_rrf_is_deterministic_on_ties():
    assert rrf_fuse([["x"], ["y"]]) == ["x", "y"]


def test_rrf_with_one_empty_ranker_keeps_the_other():
    assert rrf_fuse([[], ["a", "b"]]) == ["a", "b"]


def test_bm25_finds_exact_identifier_and_drops_zero_scores():
    ids = ["1", "2", "3"]
    docs = ["redis pool exhausted", "KeyError PAYMENT_GATEWAY_URL missing", "kafka lag"]
    assert bm25_rank("PAYMENT_GATEWAY_URL", ids, docs, limit=5) == ["2"]


def test_bm25_empty_corpus():
    assert bm25_rank("anything", [], [], limit=3) == []
