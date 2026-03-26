from __future__ import annotations

import itertools
import random
import string
from typing import Dict, List, Optional, Tuple, Union

from langchain.schema import HumanMessage, SystemMessage
from langchain_core.output_parsers import JsonOutputParser
from loguru import logger
from pydantic import BaseModel, Field

from ssa.tasks.task import Question, TaskBase, TaskSubAgent


class DiagnosisResponse(BaseModel):
    reasoning: str = Field(description="Your reasoning for the diagnosis.")
    answer: str = Field(description="Disease label (e.g. disease_ABC) or binary code (e.g. 010).")


Rule = Tuple[str, str, float]  # (attribute_name, operator, threshold)
Feedback = Tuple[str, Rule]  # (disease_name, rule)


class DiagnosisTask(TaskBase):
    """
    A decision-tree diagnosis task.

    Ground truth:
    - n_attributes named readings, each with a hidden threshold in (0.1, 0.9)
    - each patient induces a binary path (value > threshold => 1 else 0)
    - each binary path maps to a unique disease label (e.g. "disease_XYZ")
    """

    def __init__(self, task_id: str = "diagnosis", n_attributes: int = 3):
        super().__init__(task_id)
        if int(n_attributes) < 1:
            raise ValueError("n_attributes must be >= 1")

        self.n_attributes = int(n_attributes)
        self.attribute_names: List[str] = []
        self.thresholds: Dict[str, float] = {}
        self.class_name_map: Dict[str, str] = {}
        self.reverse_class_name_map: Dict[str, str] = {}

    @staticmethod
    def _generate_unique_names(count: int, *, prefix: str) -> List[str]:
        names: set[str] = set()
        while len(names) < count:
            names.add(prefix + "".join(random.choices(string.ascii_uppercase, k=3)))
        return list(names)

    def generate_ground_truth(self, seed: int | None = None):
        if seed is not None:
            random.seed(seed)

        self.attribute_names = self._generate_unique_names(self.n_attributes, prefix="attr_")
        self.thresholds = {attr: random.uniform(0.1, 0.9) for attr in self.attribute_names}

        binary_classes = ["".join(p) for p in itertools.product("01", repeat=self.n_attributes)]
        disease_names = self._generate_unique_names(len(binary_classes), prefix="disease_")

        self.class_name_map = dict(zip(binary_classes, disease_names))
        self.reverse_class_name_map = dict(zip(disease_names, binary_classes))
        self.ground_truth = dict(thresholds=self.thresholds, class_name_map=self.class_name_map)

    def generate_question(self, benchmark: bool = False) -> Question:
        if not self.thresholds:
            raise RuntimeError("Ground truth has not been generated. Call generate_ground_truth() first.")

        if benchmark:
            item_data = {attr: 0.5 for attr in self.attribute_names}
        else:
            item_data = {attr: round(random.random(), 3) for attr in self.attribute_names}

        correct_disease = self._classify_item(item_data)

        attr_str = ", ".join([f"'{k}': {v:.3f}" for k, v in item_data.items()])
        question_text = (
            f"An alien patient presents with the following biological readings: {{{attr_str}}}.\n"
            "Diagnose the disease.\n"
            "- Reply with the disease label (e.g. disease_ABC) or the binary code (e.g. 010).\n"
        )

        return Question(question_text=question_text, question_data=item_data, correct_answer=correct_disease)

    def score_response(self, question: Question, agent_response: Union[DiagnosisResponse, str]) -> float:
        correct_answer = str(question.correct_answer)
        agent_answer = self._coerce_answer(agent_response)
        agent_disease = self._normalize_to_disease(agent_answer)

        if agent_disease is None:
            return 0.0

        correct_binary = self.reverse_class_name_map[correct_answer]
        agent_binary = self.reverse_class_name_map[agent_disease]
        if len(correct_binary) != len(agent_binary):
            return 0.0

        matches = sum(1 for i in range(self.n_attributes) if correct_binary[i] == agent_binary[i])
        return float(matches / self.n_attributes)

    def extract_feedback_info(self, question: Question, agent_response: Union[DiagnosisResponse, str]) -> Optional[Feedback]:
        if not self.thresholds:
            return None

        correct_disease = str(question.correct_answer)
        agent_answer = self._coerce_answer(agent_response)
        agent_disease = self._normalize_to_disease(agent_answer)

        is_correct = agent_disease == correct_disease
        if not is_correct:
            target_disease = correct_disease
        else:
            other_diseases = [d for d in self.reverse_class_name_map.keys() if d != correct_disease]
            if not other_diseases:
                return None
            target_disease = random.choice(other_diseases)

        return self._random_rule_for_disease(target_disease)

    def get_random_feedback(self) -> Optional[Feedback]:
        if not self.reverse_class_name_map:
            return None
        disease = random.choice(list(self.reverse_class_name_map.keys()))
        return self._random_rule_for_disease(disease)

    @property
    def disease_classes(self) -> List[str]:
        return list(self.reverse_class_name_map.keys())

    def _classify_item(self, item_attributes: Dict[str, float]) -> str:
        bits: List[str] = []
        for attr in self.attribute_names:
            value = float(item_attributes[attr])
            threshold = float(self.thresholds[attr])
            bits.append("1" if value > threshold else "0")
        binary_id = "".join(bits)
        return self.class_name_map[binary_id]

    def _random_rule_for_disease(self, disease: str) -> Feedback:
        binary = self.reverse_class_name_map[disease]
        rules: List[Rule] = []
        for i, attr_name in enumerate(self.attribute_names):
            threshold = float(self.thresholds[attr_name])
            rules.append((attr_name, ">" if binary[i] == "1" else "<=", threshold))
        return (disease, random.choice(rules))

    def _normalize_to_disease(self, answer: str) -> Optional[str]:
        text = str(answer).strip()
        if not text:
            return None

        # Accept direct disease label.
        if text in self.reverse_class_name_map:
            return text

        # Accept bare suffix (e.g. "ABC" -> "disease_ABC") if it matches.
        if not text.startswith("disease_"):
            candidate = f"disease_{text}"
            if candidate in self.reverse_class_name_map:
                return candidate

        # Accept binary code.
        if set(text) <= {"0", "1"} and len(text) == self.n_attributes and text in self.class_name_map:
            return self.class_name_map[text]

        return None

    @staticmethod
    def _coerce_answer(agent_response: Union[DiagnosisResponse, str]) -> str:
        if isinstance(agent_response, DiagnosisResponse):
            return agent_response.answer
        return str(agent_response)


class DiagnosisSubAgent(TaskSubAgent):
    def __init__(self, model, task_id: str = "diagnosis"):
        super().__init__(model=model, task_id=task_id)
        self.parser = JsonOutputParser(pydantic_object=DiagnosisResponse)
        self.knowledge_base: Dict[str, str] = {}
        self.token_usage = []
        self.trace = []

        self.system_prompt = (
            "You are a diagnosis agent. You see patient readings and must diagnose a disease.\n"
            "You may be given a knowledge base of diagnostic rules per disease.\n"
            "Reply in JSON only.\n"
            "{format_instructions}"
        ).format(format_instructions=self.parser.get_format_instructions())

    def run_task(self, question: Question) -> DiagnosisResponse:
        kb_text = self._format_knowledge_base()
        prompt = f"Known rules:\n{kb_text}\n\n{question.question_text}"

        response = self.model.invoke([SystemMessage(self.system_prompt), HumanMessage(prompt)])
        self.trace.append((prompt, response.content))
        token_usage = response.response_metadata.get("token_usage")
        if token_usage:
            self.token_usage.append(token_usage)

        try:
            parsed = self.parser.parse(response.content)
            return DiagnosisResponse.model_validate(parsed)
        except Exception as e:
            logger.warning(f"Failed to parse DiagnosisResponse; returning empty answer. err={e}")
            return DiagnosisResponse(reasoning="", answer="")

    def update_knowledge_base(self, feedback_info: Optional[Feedback]):
        if not feedback_info:
            return
        disease, (attr, op, threshold) = feedback_info
        key = f"{disease}:{attr}"
        self.knowledge_base[key] = f"{attr} {op} {threshold:.3f}"

    def _format_knowledge_base(self) -> str:
        if not self.knowledge_base:
            return "No rules learned yet."
        lines = []
        for key in sorted(self.knowledge_base.keys()):
            disease, attr = key.split(":", 1)
            lines.append(f"{disease}: {self.knowledge_base[key]}")
        return "\n".join(lines)


# Backwards-compatible alias (older notebooks/scripts used this name).
DiagnosisAgent = DiagnosisSubAgent

