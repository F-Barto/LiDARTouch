from .invdepth import MultiScaleInvDepthPredictor

def create_multiscale_predictor(predictor_name, scales, **kwargs):
    assert scales > 1

    if predictor_name == 'inv_depth':
        return MultiScaleInvDepthPredictor(scales, **kwargs)
    else:
        raise NotImplementedError(f'Predictor {predictor_name} is not a valid predictor.')