from base import TaskBase, Question
from scipy.stats import kendalltau
import numpy as np
import random
import string
from typing import List, Any, Tuple

class OrderingTask(TaskBase):
    def __init__(
        self,
        task_id: int = 0,
        n_items: int = 10,
        m_probe_items: int = 5,
        p_feedback_pairs: int = 1,
    ):
        super().__init__(task_id)
        self.n_items = n_items
        self.m_probe_items = m_probe_items
        self.p_feedback_pairs = p_feedback_pairs
        self.items = []

    def generate_ground_truth(self, seed: int = None):
        if seed is not None:
            random.seed(seed)

        # Generate N unique random 3-letter strings
        self.items = []
        while len(self.items) < self.n_items:
            item = "".join(random.choices(string.ascii_uppercase, k=3))
            if item not in self.items:
                self.items.append(item)

        # Ground truth is decreasing order: items[0] > items[1] > ... > items[n-1]
        self.ground_truth = self.items.copy()

    def generate_probe_question(self) -> Question:
        # Select M random items from the N items
        probe_items = random.sample(self.items, self.m_probe_items)

        # Get correct ordering for these items
        correct_order = [item for item in self.ground_truth if item in probe_items]

        question_text = f"Order these items from largest to smallest: [{', '.join(probe_items)}]. Do not include any items that are not in these items. You should return a list of {self.m_probe_items} items."

        return Question(
            question_text=question_text,
            question_data=probe_items,
            correct_answer=correct_order,
        )

    def score_response(self, question: Question, agent_response: List[str]) -> float:
        correct_order = question.correct_answer

        if len(agent_response) != len(correct_order) or set(agent_response) != set(
            correct_order
        ):
            return 0.0  # Invalid response

        # Calculate Kendall tau distance
        return self._kendall_tau_similarity(correct_order, agent_response)

    def extract_feedback_info(self, question: Question) -> List[Tuple[str, str]]:
        # Return P random pairings with ordering info
        items = question.question_data
        if len(items) < 2:
            return []

        all_pairs = [
            (items[i], items[j])
            for i in range(len(items))
            for j in range(i + 1, len(items))
        ]
        num_pairs = min(self.p_feedback_pairs, len(all_pairs))
        selected_pairs = random.sample(all_pairs, num_pairs)

        # Format as (larger_item, smaller_item) based on ground truth
        result = []
        for item1, item2 in selected_pairs:
            idx1 = self.ground_truth.index(item1)
            idx2 = self.ground_truth.index(item2)
            if idx1 < idx2:  # item1 appears earlier, so item1 > item2
                result.append((item1, item2))
            else:
                result.append((item2, item1))

        return result

    def _kendall_tau_similarity(
        self, correct_order: List[str], agent_response: List[str]
    ) -> float:
        # Convert to rankings (positions in the list)
        n = len(correct_order)
        if n <= 1:
            return 1.0

        # Create rank mappings
        correct_ranks = {item: i for i, item in enumerate(correct_order)}
        agent_ranks = [correct_ranks[item] for item in agent_response]
        correct_ranks_list = list(range(n))

        # Calculate Kendall tau correlation
        tau, _ = kendalltau(correct_ranks_list, agent_ranks)

        # Convert correlation to similarity score (tau ranges from -1 to 1)
        # Return value between 0 and 1, where 1 is perfect agreement
        return (tau + 1) / 2


from langchain.schema import SystemMessage, HumanMessage
import ast
import re

AGENT_SYSTEM = """You are an ordering agent. Your goal is to order items from largest to smallest given a set of items. Additionally, you are given some pre-existing knowledge, which may or may not help you with your comparisons for that particular question.

This is your current knowledge base:
{knowledge_base}

Some of the items within your knowledge base might not be relevant to the question.

When given a set of items to order, respond with a Python list of the items ordered from largest to smallest.
Example response format: ['ABC', 'DEF', 'GHI']

Only respond with the ordered list with {m_probe_items} items that were actually in the question, and nothing else.
"""


class OrderingAgent:

    def __init__(self, model, task_id: str = ""):
        self.model = model
        self.knowledge_base = []
        self.task_id = task_id

    def probe_task(self, question: Question):
        # Format knowledge base for display
        kb_text = self._format_knowledge_base()

        # Create system prompt with current knowledge
        system_prompt = AGENT_SYSTEM.format(knowledge_base=kb_text, m_probe_items=6)

        question_text = question.question_text

        response = self.model.invoke(
            [SystemMessage(system_prompt), HumanMessage(question_text)]
        )

        model_response = response.content.strip()

        # Parse the response into a list
        parsed_response = self._parse_response(model_response)

        return parsed_response

    def update_knowledge_base(self, feedback_info: List[Tuple[str, str]]):
        """Update knowledge base with pairwise comparisons"""
        for larger_item, smaller_item in feedback_info:
            comparison = f"({larger_item} > {smaller_item})"
            if comparison not in self.knowledge_base:
                self.knowledge_base.append(comparison)

    def _format_knowledge_base(self) -> str:
        """Format knowledge base for display in prompt"""
        if not self.knowledge_base:
            return "No comparisons learned yet."

        # Group by task if we have task info, otherwise just list all
        if self.task_id:
            return f"{self.task_id}: {', '.join(self.knowledge_base)}"
        else:
            return ", ".join(self.knowledge_base)

    def _parse_response(self, response: str) -> List[str]:
        """Parse model response into a list of items"""
        try:
            # Try to parse as Python literal
            parsed = ast.literal_eval(response)
            if isinstance(parsed, list):
                return [str(item) for item in parsed]
        except:
            pass

        # Fallback: extract items from text using regex
        # Look for quoted strings or word-like patterns
        matches = re.findall(r"['\"]([A-Z]{3})['\"]|([A-Z]{3})", response)
        items = [match[0] or match[1] for match in matches if match[0] or match[1]]

        return items if items else []


# Training loop function
def train_agent_on_task(
    agent: OrderingAgent,
    task: OrderingTask,
    num_iterations: int = 10,
    batch_size: int = 1,
):
    """Training loop for the agent on a specific task"""

    all_scores = []

    for i in range(num_iterations):
        iter_scores = []

        print(f"\n--- Iteration {i+1} ---")

        # Generate a probe question
        for _ in range(batch_size):
            question = task.generate_probe_question()
            print(f"Question: {question.question_text}")

            # Agent attempts to answer
            agent_response = agent.probe_task(question)
            print(f"Agent response: {agent_response}")
            print(f"Correct answer: {question.correct_answer}")

            # Score the response
            score = task.score_response(question, agent_response)
            iter_scores.append(score)
            print(f"Score: {score:.3f}")

        # Extract feedback and update knowledge base (only for one question)
        feedback = task.extract_feedback_info(question)
        if feedback:
            print(f"Feedback: {feedback}")
            agent.update_knowledge_base(feedback)
            print(f"Updated KB: {agent.knowledge_base}")

        all_scores.append(iter_scores)

    return all_scores

# Example usage
def run_ordering_experiment(batch_size=5, feedback_pairs=1, n_items=10, m_probe_items=6, n_iterations=20):
    task = OrderingTask(task_id=0, n_items=n_items, m_probe_items=m_probe_items, p_feedback_pairs=feedback_pairs)
    task.generate_ground_truth(seed=42)
    print(f"Ground truth: {task.ground_truth}")
    
    agent = OrderingAgent(model, task_id=0)
    scores = train_agent_on_task(agent, task, num_iterations=n_iterations, batch_size=batch_size)
    
    return np.array(scores)
    print(f"Final scores: {scores}")
    

# %%

if __name__ == "__main__":
    task = OrderingTask(task_id=0, n_items=5, m_probe_items=3)
    # %%
    task.generate_ground_truth()
    # %%
    question = task.generate_probe_question()
    question
    # %%
    task.extract_feedback_info(question)

    task.score_response(question, ["KZL", "GAY", "KTW"])

    # %%
    from utils import init_openrouter_chat_model
    from langchain.schema import AIMessage, HumanMessage, SystemMessage

    OPENROUTER_API = (
        "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"
    )

    model = init_openrouter_chat_model(
        "openai/gpt-5-chat", api_key=OPENROUTER_API, temperature=0.5
    )

    # %%
    scores_1 = run_ordering_experiment(
        n_items=8, m_probe_items=6, n_iterations=10, feedback_pairs=1
    )
    scores_2 = run_ordering_experiment(
        n_items=8, m_probe_items=6, n_iterations=10, feedback_pairs=2
    )
    # %%
    scores_3 = run_ordering_experiment(
        n_items=8, m_probe_items=6, n_iterations=10, feedback_pairs=3
    )

    # %%
    import matplotlib.pyplot as plt

    for i, scores in enumerate([scores_1, scores_2, scores_3]):
        plt.plot(np.mean(scores, axis=1), label=f"S={i+1}")
        plt.fill_between(
            range(len(scores)),
            np.mean(scores, axis=1) - np.std(scores, axis=1),
            np.mean(scores, axis=1) + np.std(scores, axis=1),
            alpha=0.3,
        )

    plt.legend()
