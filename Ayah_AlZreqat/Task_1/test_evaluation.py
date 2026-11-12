from annolab.evaluation import evaluate_predictions


def test_correct_detection():
    predictions = [
        {"bbox": [0, 0, 100, 100], "class_id": 0, "confidence": 0.9}
    ]
    ground_truth = [
        {"bbox": [0, 0, 100, 100], "class_id": 0}
    ]

    result = evaluate_predictions(predictions, ground_truth)

    assert result["TP"] == 1
    assert result["FP"] == 0
    assert result["FN"] == 0


def test_wrong_class():
    predictions = [
        {"bbox": [0, 0, 100, 100], "class_id": 1, "confidence": 0.9}
    ]
    ground_truth = [
        {"bbox": [0, 0, 100, 100], "class_id": 0}
    ]

    result = evaluate_predictions(predictions, ground_truth)

    assert result["TP"] == 0
    assert result["FP"] == 1
    assert result["FN"] == 1


def test_low_confidence():
    predictions = [
        {"bbox": [0, 0, 100, 100], "class_id": 0, "confidence": 0.2}
    ]
    ground_truth = [
        {"bbox": [0, 0, 100, 100], "class_id": 0}
    ]

    result = evaluate_predictions(predictions, ground_truth)

    assert result["TP"] == 0
    assert result["FP"] == 0
    assert result["FN"] == 1