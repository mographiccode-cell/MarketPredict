def test_fastapi_app_imports():
    from app import app
    assert app is not None
    assert any(getattr(route, 'path', None) == '/health' for route in app.routes)


def test_predictor_metadata_is_v5():
    from predictor import CampaignPredictor
    predictor = CampaignPredictor()
    assert predictor.suite.get('dataset_version') == 'V5'
    assert set(predictor.models) == {
        'Sales', 'Traffic', 'Awareness', 'Lead Generation', 'Engagement', 'App Installs'
    }
