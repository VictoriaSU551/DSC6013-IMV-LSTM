import numpy as np
from repro.stats_solver import encode,decode,chain_score


def test_log_variance_coordinates_preserve_values_and_likelihood_gradient():
    physical=np.array([.2,.003,.5,.01]);scale=np.array([.02,1.,.02,1.]);variance=[1,3]
    u=encode(physical,scale,variance)
    np.testing.assert_allclose(decode(u,scale,variance),physical,rtol=1e-14)
    gradient=chain_score(2*physical,physical,scale,variance)
    for i in range(len(u)):
        delta=np.zeros(len(u));delta[i]=1e-5
        numeric=(np.sum(decode(u+delta,scale,variance)**2)-np.sum(decode(u-delta,scale,variance)**2))/(2e-5)
        np.testing.assert_allclose(gradient[i],numeric,rtol=1e-6,atol=1e-10)


def test_new_solver_attempt_preserves_previous_candidate_evidence(tmp_path):
    from repro.stats_solver import fresh_attempt
    a=fresh_attempt(tmp_path,'pm25');(a/'result.json').write_text('preserved')
    b=fresh_attempt(tmp_path,'pm25')
    assert a!=b and (a/'result.json').read_text()=='preserved'
