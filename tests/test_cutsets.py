from itertools import combinations
import pytest
from faultree import minimal_cut_sets


def test_repeated_events_absorb_nonminimal_supersets():
    tree = {"id": "t", "gate": "OR", "children": [
        {"id": "a"}, {"id": "both", "gate": "AND", "children": [{"id": "a"}, {"id": "b"}]}]}
    result = minimal_cut_sets(tree)
    assert result["cut_sets"] == [["a"]]
    assert result["complete"]


def test_k_out_of_n_equals_independent_combination_oracle():
    tree = {"id": "t", "gate": "K_OF_N", "k": 2, "children": [{"id": x} for x in 'abcd']}
    result = minimal_cut_sets(tree)
    assert result["cut_sets"] == [list(c) for c in combinations('abcd', 2)]
    assert result["complete"] and not result["truncated"]


def test_partial_results_never_claim_completeness():
    tree = {"id": "t", "gate": "OR", "children": [{"id": x} for x in 'abcd']}
    for kwargs, reason in [({'max_sets': 2}, 'max_sets'), ({'max_candidates': 2}, 'max_candidates'), ({'max_order': 0}, 'max_order')]:
        result = minimal_cut_sets(tree, **kwargs)
        assert result["truncated"] and not result["complete"]
        assert result["reason"] == reason
        assert all(len(cut) == 1 for cut in result["cut_sets"])


def test_nonmonotone_and_invalid_limits_are_rejected():
    tree = {"id": "t", "gate": "XOR", "children": [{"id": 'a'}, {"id": 'b'}]}
    with pytest.raises(ValueError, match='monotone'):
        minimal_cut_sets(tree)
    with pytest.raises(ValueError, match='max_candidates'):
        minimal_cut_sets(tree, max_candidates=True)
    with pytest.raises(ValueError, match='max_basic_events'):
        minimal_cut_sets({'id': 't', 'gate': 'OR', 'children': [{'id':'a'},{'id':'b'}]}, max_basic_events=1)
