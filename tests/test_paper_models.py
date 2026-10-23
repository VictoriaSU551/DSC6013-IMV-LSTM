"""Independent arithmetic, gradient, information-flow and parameter-count checks."""
import math
import sys
from pathlib import Path
import pytest
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'code'))


def api():
    import repro.models as m
    assert hasattr(m, 'mixture_statistics'), 'paper mixture implementation is missing'
    return m


def test_gaussian_posterior_and_prediction_are_distinct():
    m = api()
    mu = torch.tensor([[0., 2.]], dtype=torch.double)
    sd = torch.ones_like(mu)
    logits = torch.zeros_like(mu)
    out = m.mixture_statistics(mu, sd, logits, torch.tensor([0.], dtype=torch.double))
    assert out['prediction'].item() == pytest.approx(1.)
    assert out['posterior'][0, 0].item() == pytest.approx(0.8807970779778823)
    assert out['nll'].item() == pytest.approx(-math.log((1+math.exp(-2))/(2*math.sqrt(2*math.pi))))
    other = m.mixture_statistics(mu, sd, logits, torch.tensor([99.], dtype=torch.double))
    torch.testing.assert_close(out['prediction'], other['prediction'])


def test_detached_em_has_exact_nll_gradient():
    m = api()
    torch.manual_seed(11)
    mu = torch.randn(4, 3, dtype=torch.double, requires_grad=True)
    raw_sd = torch.randn(4, 3, dtype=torch.double, requires_grad=True)
    logits = torch.randn(4, 3, dtype=torch.double, requires_grad=True)
    out = m.mixture_statistics(mu, raw_sd.exp(), logits, torch.tensor([0.,1.,-1.,2.], dtype=torch.double))
    a = torch.autograd.grad(out['nll'], [mu, raw_sd, logits], retain_graph=True)
    b = torch.autograd.grad(out['em_loss'], [mu, raw_sd, logits])
    for x, y in zip(a, b):
        torch.testing.assert_close(x, y, atol=1e-10, rtol=1e-10)
    assert not out['posterior'].requires_grad


@pytest.mark.parametrize('kind', ['full','tensor'])
@pytest.mark.parametrize('batch', [1,4])
def test_paper_shapes_normalization_and_finite_backward(kind, batch):
    m=api()
    model=m.IMV(3, 2, kind, dropout=0.)
    x=torch.randn(batch,10,3)
    out=model(x)
    assert out['alpha'].shape == (batch,9,3)
    assert out['hidden'].shape == (batch,10,3,2)
    assert out['mu'].shape == (batch,3)
    torch.testing.assert_close(out['alpha'].sum(1), torch.ones(batch,3))
    torch.testing.assert_close(out['prior'].sum(1), torch.ones(batch))
    assert (out['sigma'] >= .001).all()
    s=m.mixture_statistics(out['mu'],out['sigma'],out['logits'],torch.randn(batch))
    s['em_loss'].backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


def test_tensor_hidden_does_not_blend_variables():
    model=api().IMV(3,4,'tensor').eval()
    x=torch.randn(2,6,3)
    changed=x.clone(); changed[:,:,1]+=5
    a,b=model(x),model(changed)
    torch.testing.assert_close(a['hidden'][:,:,[0,2]],b['hidden'][:,:,[0,2]])
    assert not torch.allclose(a['hidden'][:,:,1],b['hidden'][:,:,1])


def test_recurrence_is_causal_and_parameter_savings_match_lemma():
    m=api()
    for kind, want in [('full',204),('tensor',96)]:
        # N=3,d=2,D=6: Full = 3*6*(3+6)+18 + 3*4+6+6 = 204;
        # Tensor = 4*N*(d*d + d + d) = 96. Counts exclude output heads.
        model=m.IMV(3,2,kind).eval()
        assert model.recurrent_parameter_count() == want
        x=torch.randn(2,6,3); changed=x.clone(); changed[:,4:]+=10
        torch.testing.assert_close(model(x)['hidden'][:,:4],model(changed)['hidden'][:,:4])


@pytest.mark.parametrize('kind',['lstm','dual','retain'])
def test_deep_baseline_batch_one_backward_and_normalized_rank(kind):
    m=api()
    net=m.build_model(kind,3,4,10,2,0.)
    out=net(torch.randn(1,10,3))
    assert out['prediction'].shape == (1,)
    assert torch.isfinite(out['prediction']).all()
    assert out['importance'].shape == (1,3)
    torch.testing.assert_close(out['importance'].sum(1),torch.ones(1))
    out['prediction'].square().mean().backward()
    assert all(torch.isfinite(p.grad).all() for p in net.parameters() if p.grad is not None)


def test_dual_encoder_excludes_target_but_decoder_uses_history():
    net=api().DUAL(3,4,10,2,0.).eval()
    x=torch.randn(2,10,3)
    out=net(x)
    assert torch.equal(out['importance'][:,2],torch.zeros(2))
    torch.testing.assert_close(out['importance'].sum(1),torch.ones(2))
