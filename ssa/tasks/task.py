# %%
from abc import ABC, abstractmethod

from typing import List, Any, Tuple, Dict, Optional
from pydantic import BaseModel
import numpy as np
from loguru import logger
from ssa.common import AgentLog, SubAgentLog


class Question(BaseModel):
    question_text: str
    question_data: Any  # Task-specific data needed for scoring
    correct_answer: Any


class TaskBase(ABC):
    """Abstract base class for all task types"""

    base_reward: float
    initial_info: Any

    def __init__(self, task_id: str):
        self.id = task_id
        self.ground_truth = None

    def generate_ground_truth(self, seed: int = None):
        """Generate deterministic ground truth for this task"""
        pass

    def generate_question(self) -> Question:
        """Generate a question to probe agent performance"""
        pass

    def score_response(self, question: Question, agent_response: Any) -> bool:
        """Score agent's response to a question"""
        pass

    def extract_feedback_info(self, question: Question, agent_response: Any) -> Any:
        """Extract information needed for agent feedback"""
        pass

    def get_random_feedback(self):

        pass


class TaskSubAgent(ABC):
    """Subagent class to handle specific tasks"""

    system_prompt: str
    token_usage: List[Dict] = []
    trace: List = []

    def __init__(self, model, task_id: str):
        self.model = model
        self.task_id = task_id
        self.knowledge_base: Dict[str, str] = {}

    @abstractmethod
    def run_task(self, question: Question) -> str:
        """Calls LLM to answer a question"""
        pass

    @abstractmethod
    def update_knowledge_base(self, feedback_info: Optional[Tuple[str, str]]):
        """Updates the knowledge base"""
        pass

    @property
    def skill_level(self):
        """Length of knowledge base as proxy of skill level for now?"""
        return len(self.knowledge_base)

    def get_token_usage(self):
        return dict(
            total_tokens=sum([t["total_tokens"] for t in self.token_usage]),
            completion_tokens=sum([t["completion_tokens"] for t in self.token_usage]),
            prompt_tokens=sum([t["prompt_tokens"] for t in self.token_usage]),
        )

    def export(self) -> SubAgentLog:
        return SubAgentLog(
            knowledge_base=self.knowledge_base,
            token_usage=self.get_token_usage(),
            trace=self.trace,
        )


class TaskRunner:

    def __init__(self, subagent: TaskSubAgent, task: TaskBase):
        self.subagent = subagent
        self.task = task

    # reflect update skill maybe
    def perform_task(self, upgrade_skill_p: float = 1.0, benchmark=False) -> float:
        """Returns a 0-1 float reflecting agent performance"""

        question = self.task.generate_question(benchmark=benchmark)
        agent_response = self.subagent.run_task(question)
        agent_performance = self.task.score_response(question, agent_response)
        feedback = self.task.extract_feedback_info(question, agent_response)

        # TODO: ? Maybe this should be randomly increasing ? decrease ? by chance ?
        # Alternatively, if it's just random snippets of information, naturally the growth curve will be convex and plateaus?
        # Either way, takes a random skill P here for now
        if np.random.uniform(0, 1) <= upgrade_skill_p:
            self.subagent.update_knowledge_base(feedback)

        return agent_performance

    def upgrade_skill(self, question=None, agent_response=None):
        """Statically upgrade a task for the agent I guess LOL"""

        if agent_response:

            feedback = self.task.extract_feedback_info(question, agent_response)
        else:
            feedback = self.task.get_random_feedback()

        self.subagent.update_knowledge_base(feedback)


class ProxyTask(TaskBase):
    """Proxy task that rewards purely by skill level"""

    def __init__(self, task_id: str, noise=0.0):
        super().__init__(task_id=task_id)
        self.debug_int = 0
        self.noise = noise

    def generate_ground_truth(self, seed=None):
        return None

    def generate_question(self, benchmark: bool):
        if benchmark:

            return Question(question_text="benchmark", question_data=None, correct_answer=None)

        else:
            return Question(question_text="random", question_data=None, correct_answer=None)
        
    def score_response(self, question, agent_response):
        
        if question.question_text == "benchmark":
            return agent_response
        else:
            return np.clip(agent_response * (1 + np.random.normal(0, self.noise)), 0, 1)
    
    def extract_feedback_info(self, question, agent_response):
        # logger.debug(f"task_id: {self.id}, id: {self.debug_int}")
        self.debug_int += 1
        return None


class ProxyAgent(TaskSubAgent):
    """Proxy agent that has a skill √alue that grows with repeated tasks"""

    def __init__(self, model, task_id: str, decay=0.9, starting=0.4):
        super().__init__(model=model, task_id=task_id)
        self._skill_level = starting
        self.decay = decay

    def run_task(self, question):
        return self._skill_level
        
    def update_knowledge_base(self, feedback_info):
        self._skill_level = 1 - (1 - self._skill_level) * self.decay
        return None

    @property
    def skill_level(self):
        return int(self._skill_level * 100)


# # %%
# import numpy as np
# import matplotlib.pyplot as plt

# d = 0.9
# s = 0.4
# S = [s]
# for _ in range(100):
#     s = 1 - (1 - s) * d
#     S.append(s)
# plt.plot(S)
# %%
# Test
# task_id = "test"
# agent = ProxyAgent(model = None, task_id=task_id)
# task = ProxyTask(task_id=task_id)

# runner = TaskRunner(agent, task)
# for _ in range(10):
#     print(runner.perform_task())
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
