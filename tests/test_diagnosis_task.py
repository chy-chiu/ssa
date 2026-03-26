from ssa.tasks.diagnosis import DiagnosisResponse, DiagnosisTask


def test_diagnosis_task_scoring_and_feedback():
    task = DiagnosisTask(task_id="diagnosis", n_attributes=3)
    task.generate_ground_truth(seed=0)

    q = task.generate_question(benchmark=True)

    assert task.score_response(q, DiagnosisResponse(reasoning="", answer=q.correct_answer)) == 1.0

    correct_binary = task.reverse_class_name_map[q.correct_answer]
    opposite_binary = "".join("1" if b == "0" else "0" for b in correct_binary)
    assert task.score_response(q, opposite_binary) == 0.0

    fb = task.extract_feedback_info(q, opposite_binary)
    assert fb is not None
    disease, (attr, op, threshold) = fb
    assert disease == q.correct_answer
    assert attr in task.attribute_names
    assert op in (">", "<=")
    assert 0.1 <= threshold <= 0.9

