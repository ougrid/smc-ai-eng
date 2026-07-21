from app.agent.nodes.clarify import clarify_node


def test_clarify_surfaces_the_route_nodes_question():
    result = clarify_node({"clarification": "Which company do you mean?"})
    assert result["final_answer"] == "Which company do you mean?"


def test_clarify_falls_back_when_no_question_was_generated():
    result = clarify_node({})
    assert result["final_answer"]
