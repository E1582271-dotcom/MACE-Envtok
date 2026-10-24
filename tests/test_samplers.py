import numpy as np

from envtok_mace.data.samplers import TrainPlan, eval_shards, snake_slots, stratified_subset


def test_plan_is_deterministic_and_partitions_each_global_batch():
    cost = np.random.default_rng(0).integers(10, 500, size=1000)
    plan = TrainPlan(cost, per_rank_batch=16, world_size=4, accumulation=2, seed=3)
    assert plan.global_batch == 128 and plan.steps_per_epoch == 7
    batches = plan.global_batches(epoch=5)
    np.testing.assert_array_equal(batches, plan.global_batches(epoch=5))
    assert not np.array_equal(batches, plan.global_batches(epoch=6))
    per_rank = [plan.rank_steps(5, r) for r in range(4)]
    for s in range(plan.steps_per_epoch):
        got = np.concatenate([m for r in range(4) for m in per_rank[r][s]])
        assert sorted(got.tolist()) == sorted(batches[s].tolist())  # same members, no duplicates
        assert all(len(m) == 16 for r in range(4) for m in per_rank[r][s])


def test_balancing_reduces_the_heaviest_slot():
    cost = np.random.default_rng(1).lognormal(4.0, 1.0, size=4096)
    kw = dict(per_rank_batch=32, world_size=8, accumulation=1, seed=0)
    balanced, plain = TrainPlan(cost, balance=True, **kw), TrainPlan(cost, balance=False, **kw)

    def worst(plan):
        loads = [
            [cost[plan.rank_steps(0, r)[s][0]].sum() for r in range(8)] for s in range(plan.steps_per_epoch)
        ]
        loads = np.array(loads)
        return (loads.max(1) / loads.mean(1)).mean()

    assert worst(balanced) < 1.05 < worst(plain)


def test_eval_shards_cover_everything_once():
    cost = np.arange(103)
    shards = [np.concatenate(eval_shards(np.arange(103), cost, 4, r, batch_size=7)) for r in range(4)]
    allidx = np.concatenate(shards)
    assert sorted(allidx.tolist()) == list(range(103))


def test_snake_slots_pattern():
    np.testing.assert_array_equal(snake_slots(7, 3), [0, 1, 2, 2, 1, 0, 0])


def test_stratified_subset_takes_a_fraction_of_every_group():
    groups = np.array(["a"] * 100 + ["b"] * 10)
    idx = stratified_subset(groups, 0.1, seed=0)
    assert (groups[idx] == "a").sum() == 10 and (groups[idx] == "b").sum() == 1


def test_tracker_appends_with_the_existing_columns(tmp_path):
    import csv

    from envtok_mace.train.tracking import Tracker

    t = Tracker(tmp_path, True, {})
    t.log("epochs", {"epoch": -1, "lr": 1.0, "epoch_seconds": 0.0, "loss": 3.0})
    t.close()
    t2 = Tracker(tmp_path, True, {})  # a resumed run logs keys in another order
    t2.log("epochs", {"loss": 2.0, "epoch": 0, "epoch_seconds": 9.5, "lr": 0.5, "extra": 1})
    t2.close()
    rows = list(csv.DictReader(open(tmp_path / "epochs.csv")))
    assert rows[1] == {"epoch": "0", "lr": "0.5", "epoch_seconds": "9.5", "loss": "2.0"}


def test_eval_shards_respect_the_cost_budget():
    cost = np.array([50, 40, 30, 5, 5, 5, 5, 5])
    batches = eval_shards(np.arange(8), cost, 1, 0, batch_size=4, max_cost=60)
    assert sorted(np.concatenate(batches).tolist()) == list(range(8))
    assert all(len(b) <= 4 for b in batches)
    assert all(cost[b].sum() <= 60 or len(b) == 1 for b in batches)
