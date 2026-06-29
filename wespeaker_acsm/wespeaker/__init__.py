def load_model(*args, **kwargs):
    from wespeaker.cli.speaker import load_model as _load_model
    return _load_model(*args, **kwargs)


def load_model_pt(*args, **kwargs):
    from wespeaker.cli.speaker import load_model_pt as _load_model_pt
    return _load_model_pt(*args, **kwargs)
