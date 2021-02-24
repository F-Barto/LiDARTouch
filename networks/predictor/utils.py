from .invdepth import MultiScaleInvDepthPredictor, ScaledMultiScaleInvDepthPredictor

def create_multiscale_predictor(predictor_name, scales, **kwargs):
    assert scales > 1

    if predictor_name == 'inv_depth':
        return MultiScaleInvDepthPredictor(scales, **kwargs)
    elif predictor_name == 'scaled_inv_depth':
        return ScaledMultiScaleInvDepthPredictor(scales, **kwargs)
    else:
        raise NotImplementedError(f'Predictor {predictor_name} is not a valid predictor.')