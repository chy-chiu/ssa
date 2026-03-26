from __future__ import annotations

import ast
import random
import re
import string
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from langchain.schema import HumanMessage, SystemMessage
from langchain_core.output_parsers import JsonOutputParser
from loguru import logger
from pydantic import BaseModel, Field
from scipy.stats import kendalltau

from ssa.tasks.task import Question, TaskBase, TaskSubAgent


class OrderingResponse(BaseModel):
    reasoning: str = Field(description="Your reasoning for this ordering.")
    answer: List[str] = Field(description="Items ordered from largest to smallest.")


class OrderingTask(TaskBase):
    def __init__(
        self,
        task_id: str = "order",
        n_items: int = 10,
        m_probe_items: int = 5,
        p_feedback_pairs: int = 1,
        item_len: int = 3,
    ):
        super().__init__(task_id)
        self.n_items = int(n_items)
        self.m_probe_items = int(m_probe_items)
        self.p_feedback_pairs = int(p_feedback_pairs)
        self.item_len = int(item_len)

        self.items: List[str] = []
        self.ground_truth: Optional[List[str]] = None

    def generate_ground_truth(self, seed: int | None = None):
        if seed is not None:
            random.seed(seed)

        if self.n_items <= 0:
            raise ValueError("n_items must be > 0")
        if self.m_probe_items <= 0 or self.m_probe_items > self.n_items:
            raise ValueError("m_probe_items must be in [1, n_items]")
        if self.item_len <= 0:
            raise ValueError("item_len must be > 0")

        items: List[str] = []
        while len(items) < self.n_items:
            item = "".join(random.choices(string.ascii_uppercase, k=self.item_len))
            if item not in items:
                items.append(item)

        self.items = items
        # Ground truth is decreasing order: items[0] > items[1] > ... > items[n-1]
        self.ground_truth = self.items.copy()

    def generate_question(self, benchmark: bool = False) -> Question:
        if not self.ground_truth:
            raise RuntimeError("Ground truth has not been generated. Call generate_ground_truth() first.")

        if benchmark:
            probe_items = self.items[: self.m_probe_items]
        else:
            probe_items = random.sample(self.items, self.m_probe_items)

        correct_order = [item for item in self.ground_truth if item in probe_items]
        question_text = (
            f"Order these items from largest to smallest: [{', '.join(probe_items)}]. "
            f"Return only these {self.m_probe_items} items in a list."
        )

        return Question(question_text=question_text, question_data=probe_items, correct_answer=correct_order)

    def score_response(self, question: Question, agent_response: Union[OrderingResponse, List[str], str]) -> float:
        correct_order = list(question.correct_answer)
        response_items = self._coerce_items(agent_response)

        if len(response_items) != len(correct_order) or set(response_items) != set(correct_order):
            return 0.0

        return float(self._kendall_tau_similarity(correct_order, response_items))

    def extract_feedback_info(
        self, question: Question, agent_response: Union[OrderingResponse, List[str], str]
    ) -> List[Tuple[str, str]]:
        if not self.ground_truth:
            return []

        items = list(question.question_data)
        if len(items) < 2:
            return []

        all_pairs = [(items[i], items[j]) for i in range(len(items)) for j in range(i + 1, len(items))]
        num_pairs = min(self.p_feedback_pairs, len(all_pairs))
        selected_pairs = random.sample(all_pairs, num_pairs)

        result: List[Tuple[str, str]] = []
        for item1, item2 in selected_pairs:
            idx1 = self.ground_truth.index(item1)
            idx2 = self.ground_truth.index(item2)
            result.append((item1, item2) if idx1 < idx2 else (item2, item1))
        return result

    def get_random_feedback(self) -> List[Tuple[str, str]]:
        if not self.ground_truth or len(self.ground_truth) < 2:
            return []
        i = random.randrange(0, len(self.ground_truth) - 1)
        return [(self.ground_truth[i], self.ground_truth[i + 1])]

    @staticmethod
    def _kendall_tau_similarity(correct_order: List[str], agent_response: List[str]) -> float:
        n = len(correct_order)
        if n <= 1:
            return 1.0

        correct_ranks = {item: i for i, item in enumerate(correct_order)}
        agent_ranks = [correct_ranks[item] for item in agent_response]
        correct_ranks_list = list(range(n))

        tau, _ = kendalltau(correct_ranks_list, agent_ranks)
        if tau is None or np.isnan(tau):
            return 0.0
        return (float(tau) + 1.0) / 2.0

    @staticmethod
    def _coerce_items(agent_response: Union[OrderingResponse, List[str], str]) -> List[str]:
        if isinstance(agent_response, OrderingResponse):
            return [str(x).strip().upper() for x in agent_response.answer]
        if isinstance(agent_response, list):
            return [str(x).strip().upper() for x in agent_response]
        if isinstance(agent_response, str):
            text = agent_response.strip()
            # Try Python list literal first.
            try:
                parsed = ast.literal_eval(text)
                if isinstance(parsed, list):
                    return [str(x).strip().upper() for x in parsed]
            except Exception:
                pass

            matches = re.findall(r"['\"]([A-Z]+)['\"]|\\b([A-Z]+)\\b", text)
            items = [m[0] or m[1] for m in matches if (m[0] or m[1])]
            return [str(x).strip().upper() for x in items]

        return []


class OrderingSubAgent(TaskSubAgent):
    def __init__(self, model, task_id: str = "order"):
        super().__init__(model=model, task_id=task_id)
        self.parser = JsonOutputParser(pydantic_object=OrderingResponse)
        self.knowledge_base: Dict[str, str] = {}
        self.token_usage = []
        self.trace = []

        self.system_prompt = (
            "You are an ordering agent. You receive a set of items and must order them from largest to smallest.\n\n"
            "You may also be given a knowledge base of pairwise comparisons of the form (AAA > BBB).\n"
            "Use them when helpful; ignore irrelevant comparisons.\n\n"
            "Reply in JSON only.\n"
            "{format_instructions}"
        ).format(format_instructions=self.parser.get_format_instructions())

    def run_task(self, question: Question) -> OrderingResponse:
        kb_text = self._format_knowledge_base()
        prompt = f"Known comparisons:\n{kb_text}\n\n{question.question_text}"

        response = self.model.invoke([SystemMessage(self.system_prompt), HumanMessage(prompt)])
        self.trace.append((prompt, response.content))
        token_usage = response.response_metadata.get("token_usage")
        if token_usage:
            self.token_usage.append(token_usage)

        try:
            parsed = self.parser.parse(response.content)
            return OrderingResponse.model_validate(parsed)
        except Exception as e:
            logger.warning(f"Failed to parse OrderingResponse; returning empty answer. err={e}")
            return OrderingResponse(reasoning="", answer=[])

    def update_knowledge_base(self, feedback_info: Optional[List[Tuple[str, str]]]):
        if not feedback_info:
            return
        for larger_item, smaller_item in feedback_info:
            key = f"{larger_item.strip().upper()}>{smaller_item.strip().upper()}"
            self.knowledge_base[key] = "1"

    def probe_task(self, question: Question) -> List[str]:
        return self.run_task(question).answer

    def _format_knowledge_base(self) -> str:
        if not self.knowledge_base:
            return "No comparisons learned yet."
        return "\n".join([f"({k.replace('>', ' > ')})" for k in sorted(self.knowledge_base.keys())])


# Backwards-compatible alias (older notebooks/scripts used this name).
OrderingAgent = OrderingSubAgent

