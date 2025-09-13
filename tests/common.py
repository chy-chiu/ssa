from pydantic import ValidationError
from ssa.common import AgentActionResponse
from langchain_core.output_parsers import JsonOutputParser

def test_agent_action_response_targets_normalization():
    # Test case 1: ["task_a", 1] - single bid as list
    data = {"reasoning": "test", "action": "bid", "targets": ["task_a", 1]}
    response = AgentActionResponse(**data)
    assert response.targets == [("task_a", 1.0)]
    
    # Test case 2: [["task_a", 1]] - already correct format
    data = {"reasoning": "test", "action": "bid", "targets": [["task_a", 1]]}
    response = AgentActionResponse(**data)
    assert response.targets == [("task_a", 1.0)]
    
    # Test case 3: ["task_a"] - single task without price for bid
    data = {"reasoning": "test", "action": "bid", "targets": ["task_a"]}
    response = AgentActionResponse(**data)
    assert response.targets == [("task_a", 0)]
    
    # Test case 4: "task_a" - string input for bid
    data = {"reasoning": "test", "action": "bid", "targets": "task_a"}
    response = AgentActionResponse(**data)
    assert response.targets == [("task_a", 0)]
    
    # Test case 5: "task_a" - string input for train
    data = {"reasoning": "test", "action": "train", "targets": "task_a"}
    response = AgentActionResponse(**data)
    assert response.targets == [("task_a", -1)]
    
    # Test case 6: ["task_a"] - single task for train
    data = {"reasoning": "test", "action": "train", "targets": ["task_a"]}
    response = AgentActionResponse(**data)
    assert response.targets == [("task_a", -1)]
    
    # Test case 7: Multiple tasks for bid
    data = {"reasoning": "test", "action": "bid", "targets": ["task_a", "task_b", "task_c"]}
    response = AgentActionResponse(**data)
    assert response.targets == [("task_a", 0), ("task_b", 0), ("task_c", 0)]
    
    # Test case 8: Invalid price handling
    data = {"reasoning": "test", "action": "bid", "targets": [["task_a", "invalid_price"]]}
    response = AgentActionResponse(**data)
    assert response.targets == [("task_a", 0)]
    
    # Test case 9: Multiple bids with prices
    data = {"reasoning": "test", "action": "bid", "targets": [("task_a", 10), ["task_b", 20]]}
    response = AgentActionResponse(**data)
    assert response.targets == [("task_a", 10), ("task_b", 20.0)]
    
    # Test case 10: Empty targets
    data = {"reasoning": "test", "action": "bid", "targets": []}
    response = AgentActionResponse(**data)
    assert response.targets == []

    data = '{"reasoning": "My SK-C reputation reached 3.4* (rounded from 3.35*) after last round’s +0.1 bump, essentially tying glm’s recent 3.6–3.8* and overtaking qwen’s 3.6*. All four SK-C jobs are open, and in past rounds glm has won every one when bidding budget. By bidding exactly budget on every SK-C job I eliminate price as a factor and let my now-comparable reputation decide. I list them in descending value so the client algorithm sees the \$10 job first and, if it skips me there, still has my identical bids on the \$8, \$6 and \$4 tasks. No other skill offers a better expected return: SK-B is strong (3.9*) but goss/gpt5 sit at 4.2–4.4*, and SK-A/SK-D lag further. Focusing all five bids on the skill where I’m finally competitive maximises the chance of breaking my nine-round income drought and starting to climb the leaderboard.", "action": "bid", "targets": [["JB-C0", 10.0], ["JB-C1", 8.0], ["JB-C2", 6.0], ["JB-C3", 4.0]]}'
    data = data.replace("\\", "" )
    # response = AgentActionResponse(**data)
    parser = JsonOutputParser(pydantic_object=AgentActionResponse)
    parser.parse(data)
    import json
    json.loads(data)
    

    
if __name__ == "__main__":
    test_agent_action_response_targets_normalization()
