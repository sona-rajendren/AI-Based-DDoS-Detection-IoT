import warnings
import numpy as np

warnings.simplefilter("ignore")


def test_dataset_loads_with_expected_shape():
    from src.data_pipeline import load_dataset
    d = load_dataset()
    assert d.X_train.shape[1] == len(d.features) == 10
    assert set(np.unique(d.y_train)) == {0, 1}
    assert d.X_train.min() >= 0 and d.X_train.max() <= 1
    assert {"Normal", "DDoS_TCP", "DDoS_UDP", "DDoS_ICMP", "DDoS_HTTP"} <= set(d.types_test)


def test_feature_extraction_from_dataset_columns():
    from src.features import extract_raw_from_dataframe, IDX
    from src.synthetic import raw_to_dataset_frame
    raw = np.zeros((2, 10)); raw[0, IDX["tcp_syn"]] = 1; raw[1, IDX["udp_present"]] = 1
    back = extract_raw_from_dataframe(raw_to_dataset_frame(raw, "Normal"))
    assert np.allclose(back, raw)


def test_fedavg_is_weighted_average():
    from src.fl.common import fedavg
    out = fedavg([[np.ones((1, 3)), np.ones(1)], [np.zeros((1, 3)), np.zeros(1)]], [3, 1])
    assert np.allclose(out[0], 0.75) and np.allclose(out[1], 0.75)


def test_partition_covers_all_data_without_overlap():
    from src.fl.common import partition
    X = np.arange(100).reshape(-1, 1).astype(float); y = np.zeros(100)
    parts = [partition(X, y, 3, c)[0].ravel() for c in range(3)]
    assert sorted(np.concatenate(parts)) == list(range(100))


def test_federated_model_learns():
    from src.fl.simulate import run
    h = run(clients=3, rounds=3, save=False)
    assert h[-1]["accuracy"] > 0.8 and h[-1]["roc_auc"] > 0.8
