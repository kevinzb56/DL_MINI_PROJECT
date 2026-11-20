"""Run with pytest after installing PyTorch; this is recovery evidence only."""
import random
import pytest

torch = pytest.importorskip("torch")
np = pytest.importorskip("numpy")
from src.checkpoints import load_checkpoint, save_checkpoint


def test_resume_matches_uninterrupted_next_step(tmp_path):
    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)
    model = torch.nn.Linear(3, 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=.01)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=.9)
    def step():
        x = torch.randn(4, 3) * float(np.random.random()) + random.random()
        optimizer.zero_grad()
        loss = model(x).square().mean()
        loss.backward()
        optimizer.step()
        scheduler.step()
        return loss.detach().clone()
    step()
    save_checkpoint(tmp_path, model=model, optimizer=optimizer, scheduler=scheduler,
                    scaler=None, epoch=0, global_step=1, sampler_state={"cursor": 4},
                    config={"test": True}, manifest_hash="fixture", metrics={},
                    best_metric=.5, is_best=True)
    expected_loss = step()
    expected = {k: v.clone() for k, v in model.state_dict().items()}
    restored = load_checkpoint(tmp_path, model=model, optimizer=optimizer,
                               scheduler=scheduler, scaler=None,
                               expected_manifest_hash="fixture", expected_config={"test": True})
    assert restored["sampler_state"]["cursor"] == 4
    assert torch.equal(step(), expected_loss)
    for key, value in model.state_dict().items():
        assert torch.equal(value, expected[key])
    with pytest.raises(ValueError, match="manifest mismatch"):
        load_checkpoint(tmp_path, model=model, optimizer=optimizer, scheduler=scheduler,
                        scaler=None, expected_manifest_hash="changed", expected_config={"test": True})
