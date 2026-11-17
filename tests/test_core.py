import itertools
import numpy as np
import pytest
import torch
from acpsa_flids.models.cnn import SevenLayerCNN
from acpsa_flids.models.model_utils import blocks, set_blocks
from acpsa_flids.aggregation.contribution import ContributionLayout
from acpsa_flids.aggregation.bayesian_shrinkage import shrink
from acpsa_flids.privacy.clipping import clip_block
from acpsa_flids.privacy.rdp_accountant import RDPAccountant, calibrate_sigma, epsilon_for_sigma
from acpsa_flids.privacy.gaussian_noise import client_noise, aggregate_variance, check_honest
from acpsa_flids.semantics.threshold_selection import select_threshold
from acpsa_flids.semantics.synthetic_probing import probe
from acpsa_flids.secure_aggregation.finite_field import FixedPointField, PRIME
from acpsa_flids.secure_aggregation.shamir_secret_sharing import split, reconstruct
from acpsa_flids.secure_aggregation.protocol import SecureAggregation
from acpsa_flids.secure_aggregation.aggregation_committee import AggregationCommittee


def test_model_layout_roundtrip():
    model = SevenLayerCNN(20, 4)
    assert sum(isinstance(m, torch.nn.Conv1d) for m in model.modules()) == 6
    assert model(torch.randn(3, 20)).shape == (3, 4)
    feature, head = blocks(model)
    assert head.shape == (4, 129)
    set_blocks(model, feature+0.1, head-0.2)
    f2, h2 = blocks(model)
    np.testing.assert_allclose(f2, feature+0.1, atol=1e-7)
    np.testing.assert_allclose(h2, head-0.2, atol=1e-7)


def test_probing_and_threshold():
    q, z = np.array([0.9, 0.5, 0.4]), np.array([1, 0, 1])
    # F1 ties at .3 (2TP,1FP) and .8 (1TP,0FP): choose fewer false positives.
    tau, b = select_threshold([0.9, 0.5, 0.4, 0.3], [1, 0, 0, 1], [0.2, 0.8])
    assert tau == 0.8
    model = SevenLayerCNN(20, 3, channels=[2]*6)
    seen = []
    handle = model.register_forward_pre_hook(lambda m, args: seen.append(args[0].clone()))
    b, confidence, threshold = probe(model, [1, 0, 1], 20000, [0.1, 0.5, 0.9], torch.Generator().manual_seed(17))
    handle.remove()
    assert abs(float(seen[0].mean())) < 0.01
    assert abs(float(seen[0].std())-1) < 0.01
    np.testing.assert_allclose(confidence.sum(), 1, atol=1e-6)
    assert np.isin(b, [0, 1]).all()


def test_clipping_joint_sensitivity():
    clipped, stats = clip_block(np.array([3, 4]), 2)
    assert np.linalg.norm(clipped) == pytest.approx(2)
    assert stats["ratio"] == 0.4
    layout = ContributionLayout(2, 3, 2, 1, [0.2, 0.3, 0.4])
    rng = np.random.default_rng(2)
    vectors = []
    for _ in range(100):
        b = rng.integers(0, 2, 3)
        v, stat = layout.build(rng.normal(size=2)*10, rng.normal(size=(3, 2))*10, b)
        f, indicator, head = layout.unpack(v)
        assert np.linalg.norm(f) <= 1+1e-10
        assert (np.linalg.norm(head, axis=1) <= layout.class_bounds+1e-10).all()
        assert not head[b == 0].any()
        np.testing.assert_array_equal(indicator, b)
        assert np.linalg.norm(v) <= layout.radius+1e-10
        vectors.append(v)
    assert np.linalg.norm(vectors[0]-vectors[1]) <= layout.sensitivity


@pytest.mark.parametrize("budget", [0.5, 1, 2, 4, 8])
def test_accountant(budget):
    sigma = calibrate_sigma(budget, 100)
    assert epsilon_for_sigma(sigma, 100) == pytest.approx(budget)
    accountant = RDPAccountant()
    for t in range(100):
        assert accountant.can_release(sigma, budget)
        accountant.step(sigma)
        assert accountant.epsilon == pytest.approx(epsilon_for_sigma(sigma, t+1))
    assert not accountant.can_release(sigma, budget)
    assert epsilon_for_sigma(sigma*2, 100) < budget


def test_distributed_noise_variance_independence():
    a = client_noise(100000, 2, 3, 80, np.random.default_rng(1))
    b = client_noise(100000, 2, 3, 80, np.random.default_rng(2))
    assert np.var(a) == pytest.approx(36/80, rel=0.02)
    assert abs(np.corrcoef(a, b)[0, 1]) < 0.02
    assert aggregate_variance(100, 2, 3, 80) == 45
    assert aggregate_variance(80, 2, 3, 80) == 36
    check_honest(100, 20, 80)
    with pytest.raises(RuntimeError):
        check_honest(99, 20, 80)


def test_shamir_all_subsets_and_randomness():
    secret = np.array([0, 1, PRIME-1, 9371], dtype=np.int64)
    shares = split(secret)
    for nodes in itertools.combinations(range(1, 6), 3):
        np.testing.assert_array_equal(reconstruct({n: shares[n] for n in nodes}), secret)
    np.testing.assert_array_equal(reconstruct(shares), secret)
    assert not np.array_equal(shares[1], split(secret)[1])
    with pytest.raises(RuntimeError):
        reconstruct({1: shares[1], 2: shares[2]})


def test_fixed_point_overflow_and_quantization():
    field = FixedPointField()
    x = np.array([-0.200034, 0.033346, 1.123414])
    np.testing.assert_allclose(field.decode(field.encode(x, 100)), x, atol=0.5/field.scale)
    with pytest.raises(OverflowError):
        field.encode(np.array([1e8]), 100)
    with pytest.raises(ValueError):
        field.encode(np.array([np.inf]), 1)


def test_secure_sum_dropout_and_agreement():
    rng = np.random.default_rng(42)
    vectors = rng.normal(size=(7, 101))
    protocol = SecureAggregation()
    committee = AggregationCommittee(5, 101, round_id=9)
    for identity, vector in enumerate(vectors):
        committee.deliver(identity, protocol.share(vector, 7))
    for nodes in itertools.combinations(range(1, 6), 3):
        shares = committee.finish([n for n in range(1, 6) if n not in nodes])
        recovered = protocol.recover(shares, list(range(7)), 9)
        np.testing.assert_allclose(recovered, vectors.sum(axis=0), atol=7/20000)
    with pytest.raises(RuntimeError):
        protocol.recover(committee.finish([1, 2, 3]), list(range(7)), 9)
    with pytest.raises(RuntimeError):
        protocol.recover(committee.finish(), [0, 1], 9)
    shares = committee.finish()
    shares[-1].value[0] = (shares[-1].value[0]+1) % PRIME
    with pytest.raises(RuntimeError):
        protocol.recover(shares, list(range(7)), 9)


def test_bayesian_posterior_and_monotonicity():
    n = np.array([-2.0, 1, 2, 10, 200])
    h = np.ones((5, 3))*2
    update, projected, weight, direct = shrink(n, h, 100, 4, 0.2)
    assert ((weight >= 0) & (weight <= 1)).all()
    assert (np.diff(weight) >= 0).all()
    assert not update[:2].any()
    assert projected[-1] == 100
    expected = (2**2*0.2)/(2**2*0.2+4) * 2/2
    assert update[2, 0] == pytest.approx(expected)
    _, _, higher_noise_weight, _ = shrink(n, h, 100, 8, 0.2)
    assert (higher_noise_weight <= weight).all()
