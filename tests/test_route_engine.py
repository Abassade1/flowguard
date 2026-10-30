"""Tests for the multi-criteria route comparison."""

from modules.config import SOCIAL_NOT_ASSESSED
from modules.route_engine import CRITERIA, explain_tradeoffs, score_routes, synthetic_route_alternatives


def test_scores_are_bounded_and_hard_constraints_unranked():
    scored = score_routes(synthetic_route_alternatives())
    assert scored["score"].between(0, 100).all()
    hard = scored[scored["has_hard_constraint"]]
    assert not hard.empty
    assert hard["rank"].isna().all()
    ranked = scored[~scored["has_hard_constraint"]]["rank"]
    assert sorted(ranked) == list(range(1, len(ranked) + 1))


def test_weights_change_the_ranking():
    routes = synthetic_route_alternatives()
    only_capex = {c.key: 0.0 for c in CRITERIA} | {"capex_cad": 1.0}
    only_forest = {c.key: 0.0 for c in CRITERIA} | {"forest_clearing_km": 1.0}
    top = lambda s: s.loc[s["rank"] == 1, "route_id"].iloc[0]  # noqa: E731
    assert top(score_routes(routes, only_capex)) != top(score_routes(routes, only_forest))


def test_social_considerations_are_not_scored():
    routes = synthetic_route_alternatives()
    assert (routes["social_indigenous"] == SOCIAL_NOT_ASSESSED).all()
    assert all("social" not in c.key and "indigenous" not in c.key for c in CRITERIA)


def test_explanations_mention_unknowns_and_constraints():
    scored = score_routes(synthetic_route_alternatives())
    notes = explain_tradeoffs(scored)
    assert set(notes) == set(scored["route_id"])
    assert any("Hard constraint" in line for line in notes["A"])
    assert all(any("Unknown" in line for line in lines) for lines in notes.values())
