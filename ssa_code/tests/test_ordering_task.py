from ssa.tasks.ordering import OrderingResponse, OrderingTask


def test_ordering_task_scoring_and_feedback():
    task = OrderingTask(task_id="order", n_items=6, m_probe_items=4, p_feedback_pairs=1, item_len=3)
    task.generate_ground_truth(seed=123)

    q = task.generate_question(benchmark=True)

    score_correct = task.score_response(q, OrderingResponse(reasoning="", answer=q.correct_answer))
    assert score_correct == 1.0

    reversed_answer = list(reversed(q.correct_answer))
    score_reversed = task.score_response(q, OrderingResponse(reasoning="", answer=reversed_answer))
    assert 0.0 <= score_reversed <= 1.0
    assert score_reversed < 1.0

    fb = task.extract_feedback_info(q, OrderingResponse(reasoning="", answer=q.correct_answer))
    assert len(fb) == 1
    larger, smaller = fb[0]
    assert task.ground_truth.index(larger) < task.ground_truth.index(smaller)

    rand_fb = task.get_random_feedback()
    assert len(rand_fb) == 1
    larger, smaller = rand_fb[0]
    assert task.ground_truth.index(larger) < task.ground_truth.index(smaller)

