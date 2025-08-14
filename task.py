# %%
from abc import ABC, abstractmethod

from typing import List, Any, Tuple
from pydantic import BaseModel
import numpy as np

class Question(BaseModel):
    question_text: str
    question_data: Any  # Task-specific data needed for scoring
    correct_answer: Any


class TaskBase(ABC):
    """Abstract base class for all task types"""
    
    base_reward: float

    def __init__(self, task_id: str):
        self.id = task_id
        self.ground_truth = None

    def generate_ground_truth(self, seed: int = None):
        """Generate deterministic ground truth for this task"""
        pass

    def generate_probe_question(self) -> Question:
        """Generate a single probe question"""
        pass

    def score_response(self, question: Question, agent_response: Any) -> bool:
        """Score agent's response to a question"""
        pass

    def extract_feedback_info(self, question: Question) -> Any:
        """Extract information needed for agent feedback"""
        pass
    

class TaskRunner:
    
    def __init__(self, agent, task):
        self.agent = agent
        self.task = task
        
    def run_task(self):
        pass 
    
    def perform_task(self) -> Tuple[float, float, str]:
        # TODO: Make task payment here dynamic / stochastic
        
        ADJUSTED_REWARD = self.task.base_reward * np.random.uniform(0, 1)
        feedback = ""
        return self.task.base_reward, ADJUSTED_REWARD, feedback
                
# TODO: Incorporate logic below into the runner class
# %%

# Training loop function
# def run_task(
#     agent: OrderingAgent
#     task: OrderingTask,
#     num_iterations: int = 10,
#     batch_size: int = 1,
# ):
#     """Training loop for the agent on a specific task"""

#     all_scores = []

#     for i in range(num_iterations):
#         iter_scores = []

#         print(f"\n--- Iteration {i+1} ---")

#         # Generate a probe question
#         for _ in range(batch_size):
#             question = task.generate_probe_question()
#             print(f"Question: {question.question_text}")

#             # Agent attempts to answer
#             agent_response = agent.probe_task(question)
#             print(f"Agent response: {agent_response}")
#             print(f"Correct answer: {question.correct_answer}")

#             # Score the response
#             score = task.score_response(question, agent_response)
#             iter_scores.append(score)
#             print(f"Score: {score:.3f}")

#         # Extract feedback and update knowledge base (only for one question)
#         feedback = task.extract_feedback_info(question)
#         if feedback:
#             print(f"Feedback: {feedback}")
#             agent.update_knowledge_base(feedback)
#             print(f"Updated KB: {agent.knowledge_base}")

#         all_scores.append(iter_scores)

#     return all_scores

# %%
