import numpy as np
import pytest
from acpsa_flids.privacy.gaussian_noise import private_rng
from acpsa_flids.aggregation.bayesian_shrinkage import PriorVariance
from acpsa_flids.aggregation.server import update_model
from acpsa_flids.aggregation.contribution import ContributionLayout
from acpsa_flids.models.cnn import SevenLayerCNN
from acpsa_flids.models.model_utils import blocks
from acpsa_flids.utils.config import load_config, ROOT


def test_os_gaussian_moments_and_independence():
    a, b = private_rng().normal(0, 2, 100000), private_rng().normal(0, 2, 100000)
    assert abs(a.mean()) < .05
    assert a.var() == pytest.approx(4, rel=.03)
    assert abs(np.corrcoef(a,b)[0,1]) < .02


def test_decoupled_global_update_and_history():
    config = load_config(ROOT/"configs/default.yaml")["aggregation"]
    model = SevenLayerCNN(10, 3, channels=[2]*6)
    feature, head = blocks(model)
    layout = ContributionLayout(len(feature), 3, head.shape[1], 1, .1)
    prior = PriorVariance(3, config)
    old_prior = prior.values.copy()
    count = np.array([0., 2., 4.])
    aggregate = layout.pack(np.ones(len(feature)), count, np.ones_like(head))
    stats = update_model(model, layout, aggregate, 5, 2., prior, config)
    after_feature, after_head = blocks(model)
    np.testing.assert_allclose(after_feature-feature, .2, atol=1e-7)
    np.testing.assert_array_equal(after_head[0], head[0])
    expected = old_prior[1]*4/(old_prior[1]*4+2)/2
    np.testing.assert_allclose(after_head[1]-head[1], expected, atol=1e-7)
    assert not np.array_equal(prior.values, old_prior)
