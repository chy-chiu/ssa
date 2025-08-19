# %%
import random
import string
from typing import List, Set, Dict, Any, Tuple
from pydantic import BaseModel
from dataclasses import dataclass
from langchain.schema import AIMessage, HumanMessage, SystemMessage
import ast


from task import TaskBase, train_agent_on_task

class Question(BaseModel):
    question_text: str
    question_data: Any
    correct_answer: Any

@dataclass
class BooleanRule:
    disease: str
    expression: str  # Human readable
    evaluator: callable  # Function to evaluate

class BooleanReasoningTask(TaskBase):
    def __init__(self, task_id: str, n_diseases: int = 3, n_symptoms: int = 6, n_labs: int = 4, p_feedback_rules: int = 2):
        super().__init__(task_id)
        self.n_diseases = n_diseases
        self.n_symptoms = n_symptoms
        self.n_labs = n_labs
        self.p_feedback_rules = p_feedback_rules
        self.diseases = []
        self.symptoms = []
        self.labs = []
        self.boolean_rules = []

    def generate_ground_truth(self, seed: int = None):
        if seed is not None:
            random.seed(seed)

        # Generate unique 3-letter codes
        self.diseases = self._generate_unique_codes("DIS", self.n_diseases)
        self.symptoms = self._generate_unique_codes("SYM", self.n_symptoms)
        self.labs = self._generate_unique_codes("LAB", self.n_labs)

        # Generate Boolean rules for each disease
        self.boolean_rules = []
        for disease in self.diseases:
            rule = self._generate_boolean_rule(disease)
            self.boolean_rules.append(rule)

        self.ground_truth = self.boolean_rules

    def generate_question(self) -> Question:
        # Generate a random patient case
        patient_symptoms = set(random.sample(self.symptoms, random.randint(1, 4)))
        patient_labs = set(random.sample(self.labs, random.randint(1, 3)))

        # Determine correct diagnosis based on Boolean rules
        correct_diseases = []
        for rule in self.boolean_rules:
            if rule.evaluator(patient_symptoms, patient_labs):
                correct_diseases.append(rule.disease)

        # Format patient case
        symptoms_str = ", ".join(sorted(patient_symptoms))
        labs_str = ", ".join(sorted(patient_labs))

        question_text = f"""Patient Case:
Symptoms: {symptoms_str}
Labs: {labs_str}

Which diseases does this patient have? Respond with a list of disease codes.
Example format: ['DIS_ABC', 'DIS_DEF'] or [] for no diseases"""

        return Question(
            question_text=question_text,
            question_data={
                'symptoms': patient_symptoms,
                'labs': patient_labs
            },
            correct_answer=correct_diseases
        )

    def score_response(self, question: Question, agent_response: List[str]) -> float:
        correct_diseases = set(question.correct_answer)
        predicted_diseases = set(agent_response) if agent_response else set()

        # Binary accuracy - exact match required
        return 1.0 if correct_diseases == predicted_diseases else 0.0

    def extract_feedback_info(self, question: Question) -> List[str]:
        """Return relevant Boolean rules for this case"""
        patient_symptoms = question.question_data['symptoms']
        patient_labs = question.question_data['labs']

        # Find rules that could be relevant (involve symptoms/labs in this case)
        relevant_rules = []
        for rule in self.boolean_rules:
            # Check if rule involves any of the patient's symptoms or labs
            if self._rule_involves_features(rule, patient_symptoms, patient_labs):
                relevant_rules.append(rule.expression)

        # Return up to p_feedback_rules
        num_rules = min(self.p_feedback_rules, len(relevant_rules))
        return random.sample(relevant_rules, num_rules) if relevant_rules else []

    def _generate_unique_codes(self, prefix: str, count: int) -> List[str]:
        codes = []
        while len(codes) < count:
            suffix = ''.join(random.choices(string.ascii_uppercase, k=3))
            code = f"{prefix}_{suffix}"
            if code not in codes:
                codes.append(code)
        return codes

    def _generate_boolean_rule(self, disease: str) -> BooleanRule:
        """Generate a Boolean rule for a disease"""
        # Randomly choose rule complexity
        rule_type = random.choice(['simple_and', 'simple_or', 'complex_and_or', 'with_negation'])

        if rule_type == 'simple_and':
            s1, s2 = random.sample(self.symptoms, 2)
            expression = f"{disease} = ({s1} AND {s2})"
            evaluator = lambda symp, labs: s1 in symp and s2 in symp

        elif rule_type == 'simple_or':
            l1, l2 = random.sample(self.labs, 2)
            expression = f"{disease} = ({l1} OR {l2})"
            evaluator = lambda symp, labs: l1 in labs or l2 in labs

        elif rule_type == 'complex_and_or':
            s1 = random.choice(self.symptoms)
            l1, l2 = random.sample(self.labs, 2)
            expression = f"{disease} = ({s1} AND ({l1} OR {l2}))"
            evaluator = lambda symp, labs: s1 in symp and (l1 in labs or l2 in labs)

        elif rule_type == 'with_negation':
            s1 = random.choice(self.symptoms)
            l1 = random.choice(self.labs)
            s2 = random.choice([s for s in self.symptoms if s != s1])
            expression = f"{disease} = ({s1} AND {l1} AND NOT {s2})"
            evaluator = lambda symp, labs: s1 in symp and l1 in labs and s2 not in symp

        return BooleanRule(disease, expression, evaluator)

    def _rule_involves_features(self, rule: BooleanRule, symptoms: Set[str], labs: Set[str]) -> bool:
        """Check if rule involves any of the given symptoms or labs"""
        all_features = symptoms.union(labs)
        return any(feature in rule.expression for feature in all_features)

# Agent prompt for Boolean reasoning
BOOLEAN_AGENT_SYSTEM = """You are a diagnostic agent. Your goal is to diagnose diseases based on patient symptoms and lab results using learned Boolean rules.

This is your current knowledge base of diagnostic rules:
{knowledge_base}

When given a patient case with symptoms and labs, respond with a Python list of disease codes that apply.
If no diseases apply, respond with an empty list: []

Example response formats:
['DIS_ABC']
['DIS_ABC', 'DIS_DEF']
[]

Only respond with the disease list, nothing else.
"""

class BooleanReasoningAgent:

    def __init__(self, model, task_id: str = ""):
        self.model = model
        self.knowledge_base = []
        self.task_id = task_id

    def probe_task(self, question: Question):
        # Format knowledge base for display
        kb_text = self._format_knowledge_base()

        # Create system prompt with current knowledge
        system_prompt = BOOLEAN_AGENT_SYSTEM.format(knowledge_base=kb_text)

        question_text = question.question_text

        response = self.model.invoke([
            SystemMessage(system_prompt),
            HumanMessage(question_text)
        ])

        model_response = response.content.strip()

        # Parse the response into a list
        parsed_response = self._parse_response(model_response)

        return parsed_response

    def update_knowledge_base(self, feedback_info: List[str]):
        """Update knowledge base with Boolean rules"""
        for rule in feedback_info:
            if rule not in self.knowledge_base:
                self.knowledge_base.append(rule)

    def _format_knowledge_base(self) -> str:
        """Format knowledge base for display in prompt"""
        if not self.knowledge_base:
            return "No diagnostic rules learned yet."

        return '\n'.join(self.knowledge_base)

    def _parse_response(self, response: str) -> List[str]:
        """Parse model response into a list of disease codes"""
        try:
            # Try to parse as Python literal
            parsed = ast.literal_eval(response)
            if isinstance(parsed, list):
                return [str(item) for item in parsed]
        except:
            pass

        # Fallback: extract disease codes from text
        import re
        matches = re.findall(r"['\"]?(DIS_[A-Z]{3})['\"]?", response)
        return matches if matches else []

# Example usage
def run_boolean_experiment(model):
    # Initialize task and agent
    task = BooleanReasoningTask("boolean_task_1", n_diseases=3, n_symptoms=5, n_labs=5)
    task.generate_ground_truth(seed=42)

    print("Generated Boolean Rules:")
    for rule in task.boolean_rules:
        print(f"  {rule.expression}")

    agent = BooleanReasoningAgent(model, task_id="boolean_task_1")
    scores = train_agent_on_task(agent, task, num_iterations=10, batch_size=3)

    return scores

# %%
from utils import init_openrouter_chat_model
from langchain.schema import AIMessage, HumanMessage, SystemMessage

OPENROUTER_API = "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"

model = init_openrouter_chat_model('openai/gpt-5-chat', api_key=OPENROUTER_API, temperature=0.5)

# %%
scores = run_boolean_experiment(model)
# %%
import matplotlib.pyplot as plt
import numpy as np

plt.plot(np.mean(scores, axis=1))
plt.fill_between(
    range(len(scores)),
    np.mean(scores, axis=1) - np.std(scores, axis=1),
    np.mean(scores, axis=1) + np.std(scores, axis=1),
    alpha=0.3,
)

# %%
