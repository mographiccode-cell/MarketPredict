from assessment import apply_auto_assessment, budget_adequacy_score, evidence_score, rubric_score


def test_budget_adequacy_matches_v5_reference_midpoint():
    # Sales need = SAR 22 per 1000; exactly that density maps to 50 in V5 formula.
    score, meta = budget_adequacy_score(22000, 1_000_000, "Sales")
    assert score == 50.0
    assert meta["objective_reference_need_per_1000_sar"] == 22


def test_legacy_rubric_is_still_deterministic_for_api_compatibility():
    a, _ = rubric_score("Creative_Quality_Score", "ready")
    b, _ = rubric_score("Creative_Quality_Score", "ready")
    assert a == b == 75.0


def test_evidence_score_is_deterministic():
    payload = {
        "evidence__Creative_Quality_Score__0": "1",
        "evidence__Creative_Quality_Score__1": "1",
        "evidence__Creative_Quality_Score__2": "0",
    }
    score, meta = evidence_score("Creative_Quality_Score", payload)
    assert score == 70.0
    assert meta["method"] == "evidence-based pre-launch checklist"
    assert meta["verified_checks"] == 2
    assert meta["total_checks"] == 3


def test_server_recomputes_evidence_and_overrides_tampered_numeric_score():
    payload = {
        "Campaign_Objective": "Sales",
        "Budget_SAR": "22000",
        "Estimated_Audience_Size": "1000000",
        "Creative_Quality_Score": "100",
        "evidence__Creative_Quality_Score__0": "1",
        "evidence__Creative_Quality_Score__1": "0",
        "evidence__Creative_Quality_Score__2": "0",
    }
    out, provenance, _ = apply_auto_assessment(payload)
    assert out["Creative_Quality_Score"] == 50.0
    assert out["Budget_Adequacy_Score"] == 50.0
    assert provenance["Creative_Quality_Score"]["method"] == "evidence-based pre-launch checklist"
    assert not any(key.startswith("evidence__") for key in out)


def test_incomplete_evidence_is_rejected():
    payload = {
        "evidence__Offer_Strength_Score__0": "1",
        "evidence__Offer_Strength_Score__1": "0",
    }
    try:
        evidence_score("Offer_Strength_Score", payload)
    except ValueError as exc:
        assert "أكمل جميع أدلة" in str(exc)
    else:
        raise AssertionError("Incomplete evidence must be rejected")
