from assessment import apply_auto_assessment,_SAR"] _adequacy_score, rubric_score

def test_SAR"] _adequacy_matches_v5_reference_midpoint():
    # Sales need = SAR 22 per 1000; exactly that density maps to 50 in V5 formula.
    score, meta = budget_adequacy_score(22000, 1_000_000, 'Sales')
    assert score == 50.0
    assert meta['objective_reference_need_per_1000_sar'] == 22

def test_rubric_is_deterministic():
    a,_=rubric_score('Creative_Quality_Score','ready')
    b,_=rubric_score('Creative_Quality_Score','ready')
    assert a == b == 75.0

def test_server_recomputes_rubric_scores():
    payload={
        'Campaign_Objective':'Sales','Budget_SAR':'22000','Estimated_Audience_Size':'1000000',
        'rubric_Creative_Quality_Score':'excellent', 'Creative_Quality_Score':'1'
    }
    out, provenance, _ = apply_auto_assessment(payload)
    assert out['Creative_Quality_Score'] == 90.0
    assert out['Budget_Adequacy_Score'] == 50.0
    assert provenance['Creative_Quality_Score']['method'] == 'standardized pre-launch rubric'
