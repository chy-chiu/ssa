# %%
import random
import string
import itertools
from typing import List, Any, Tuple, Dict
from ssa.task import TaskBase
from ssa.utils import init_azure_model


# --- Assuming TaskBase and Question stubs from before ---
class Question:
    def __init__(self, question_text: str, question_data: Any, correct_answer: Any):
        self.question_text = question_text
        self.question_data = question_data
        self.correct_answer = correct_answer

    def __repr__(self):
        return f"Question(text='{self.question_text}', data={self.question_data}, answer='{self.correct_answer}')"


# --- New and Improved DiagnosisTask Implementation ---
class DiagnosisTask(TaskBase):
    """
    An alien diagnosis task. The agent must classify an alien's disease based
    on N biological readings (attributes).
    The ground truth is a decision tree where each reading has a hidden threshold.
    The combination of whether each reading is above or below its threshold
    determines the final disease classification.
    """

    def __init__(
        self,
        task_id: int = 1,
        n_attributes: int = 3,
    ):
        """
        Initializes the DiagnosisTask.

        Args:
            task_id (int): The identifier for this task.
            n_attributes (int): The number of biological readings. This defines
                                the decision tree depth (2^n classes).
        """
        super().__init__(task_id)
        if n_attributes < 1:
            raise ValueError("n_attributes must be at least 1.")

        self.n_attributes = n_attributes

        # Ground Truth elements
        self.attribute_names: List[str] = []
        self.thresholds: Dict[str, float] = {}
        self.ground_truth = None  # Will store the thresholds

        # Mapping for themed names
        self.class_name_map: Dict[str, str] = {}  # "010" -> "Zorvanian Pox"
        self.reverse_class_name_map: Dict[str, str] = {}  # "Zorvanian Pox" -> "010"

    def _generate_unique_names(self, count: int, prefix="") -> List[str]:
        """Generates a list of unique 3-letter uppercase strings."""
        names = set()
        while len(names) < count:
            name = prefix + "".join(random.choices(string.ascii_uppercase, k=3))
            names.add(name)
        return list(names)

    def generate_ground_truth(self, seed: int = None):
        """
        Generates the ground truth decision tree: attribute names, their
        thresholds, and the names for each resulting disease class.
        """
        if seed is not None:
            random.seed(seed)

        # 1. Generate unique names for attributes (e.g., 'KPL', 'VRT')
        self.attribute_names = self._generate_unique_names(
            self.n_attributes, prefix="attr_"
        )

        # 2. Assign a random threshold to each named attribute
        self.thresholds = {
            attr: random.uniform(0.1, 0.9) for attr in self.attribute_names
        }
        self.ground_truth = self.thresholds

        # 3. Generate internal binary class IDs and map them to unique disease names
        binary_classes = [
            "".join(p) for p in itertools.product("01", repeat=self.n_attributes)
        ]
        disease_names = self._generate_unique_names(
            len(binary_classes), prefix="disease_"
        )

        self.class_name_map = dict(zip(binary_classes, disease_names))
        self.reverse_class_name_map = dict(zip(disease_names, binary_classes))

    def _classify_item(self, item_attributes: Dict[str, float]) -> str:
        """
        Determines the disease (themed name) of an item by applying the ground truth rules.
        """
        class_path_binary = []
        for attr in self.attribute_names:  # Use consistent order
            value = item_attributes[attr]
            threshold = self.thresholds[attr]
            class_path_binary.append("1" if value > threshold else "0")

        binary_id = "".join(class_path_binary)
        return self.class_name_map[binary_id]

    def generate_question(self) -> Question:
        """
        Generates a new "alien patient" with random readings and asks for a diagnosis.
        """
        if not self.ground_truth:
            raise RuntimeError(
                "Ground truth has not been generated. Call generate_ground_truth() first."
            )

        # Generate a random item (a set of attribute values)
        item_data = {attr: round(random.random(), 3) for attr in self.attribute_names}

        # Determine its correct disease class using the ground truth
        correct_disease = self._classify_item(item_data)

        # Format attributes for the question text
        attr_str = ", ".join([f"'{k}': {v:.3f}" for k, v in item_data.items()])
        question_text = (
            f"An alien patient presents with the following biological readings: {{{attr_str}}}. "
            f"Please diagnose the disease. Respond with only the disease code"
        )

        return Question(
            question_text=question_text,
            question_data=item_data,
            correct_answer=correct_disease,
        )

    def score_response(self, question: Question, agent_response: str) -> float:
        """
        Scores based on the proportion of correct decision paths.
        1.0 for a perfect match, 0.0 if the disease name is invalid.
        Partial credit for getting some of the diagnostic criteria correct.
        """
        correct_answer = question.correct_answer
        if "disease_" not in agent_response:
            agent_response = f"disease_{agent_response}"
        if agent_response not in self.reverse_class_name_map:
            return 0.0  # Agent returned an invalid disease name

        # Convert themed names back to internal binary IDs for comparison
        correct_binary = self.reverse_class_name_map[correct_answer]
        agent_binary = self.reverse_class_name_map[agent_response]

        if len(correct_binary) != len(agent_binary):
            return 0.0  # Should not happen with valid names, but a safeguard

        matches = sum(
            1 for i in range(self.n_attributes) if correct_binary[i] == agent_binary[i]
        )

        return matches / self.n_attributes

    def extract_feedback_info(
        self, question: Question, agent_response: str
    ) -> Tuple[str, Tuple[str, str, float]]:
        """
        Provides feedback based on the agent's performance.

        - If WRONG: Returns the correct disease and one of its defining rules.
        - If RIGHT: Returns a random DIFFERENT disease and one of its defining rules.

        Returns:
            A tuple of (disease_name, rule_tuple), where rule_tuple is
            (attribute_name, operator, threshold).
        """
        is_correct = agent_response == question.correct_answer

        if not is_correct:
            target_disease = question.correct_answer
        else:
            # Agent was right, give it info on another random disease
            other_diseases = [
                d for d in self.reverse_class_name_map if d != question.correct_answer
            ]
            if not other_diseases:  # Only one possible class
                return None
            target_disease = random.choice(other_diseases)

        # Get the binary path for the target disease
        target_binary = self.reverse_class_name_map[target_disease]

        # Build all rules that define this disease
        defining_rules = []
        for i, attr_name in enumerate(self.attribute_names):
            threshold = self.thresholds[attr_name]
            if target_binary[i] == "1":
                rule = (attr_name, ">", threshold)
            else:
                rule = (attr_name, "<=", threshold)
            defining_rules.append(rule)

        # Select one random rule to provide as feedback
        feedback_rule = random.choice(defining_rules)

        return (target_disease, feedback_rule)

    @property
    def disease_classes(self):
        return list(self.reverse_class_name_map.keys())


### Example Usage and Test
if __name__ == "__main__":
    # 1. Initialize the task with 4 attributes (2^4 = 16 diseases)
    task = DiagnosisTask(task_id=1, n_attributes=4)

    # 2. Generate the hidden rules and names
    task.generate_ground_truth(seed=42)
    print("--- Ground Truth (Secret Info) ---")
    print(f"Attribute Names: {task.attribute_names}")
    print(f"Thresholds: { {k: f'{v:.3f}' for k, v in task.ground_truth.items()} }")
    print("\nDisease Name Mappings:")
    for binary, name in task.class_name_map.items():
        print(f"  {binary} -> {name}")
    print("-" * 20, "\n")

    # 3. Generate a question
    q = task.generate_question()
    print("--- Agent Interaction ---")
    print(f"Question: {q.question_text}")
    print(f"Correct Answer (for our reference): {q.correct_answer}")

    # 4. Simulate agent responses and get scores/feedback

    # Scenario A: Agent gets it completely wrong
    print("\n--- SCENARIO A: WRONG GUESS ---")
    # Let's find an answer that is very different from the correct one
    correct_binary = task.reverse_class_name_map[q.correct_answer]
    wrong_binary = "".join(
        ["1" if b == "0" else "0" for b in correct_binary]
    )  # The exact opposite
    wrong_answer = task.class_name_map[wrong_binary]

    score_a = task.score_response(q, wrong_answer)
    print(f"Agent Response: '{wrong_answer}'")
    print(f"Score: {score_a:.2f} (Expected 0.0)")

    feedback_a = task.extract_feedback_info(q, wrong_answer)
    print(
        f"Feedback (Corrective): 'Actually, the right answer was {feedback_a[0]}. One reason is that {feedback_a[1][0]} {feedback_a[1][1]} {feedback_a[1][2]:.3f}.'"
    )

    # Scenario B: Agent gets it partially right
    print("\n--- SCENARIO B: PARTIALLY CORRECT GUESS ---")
    # Let's find an answer that differs by only one bit
    partially_correct_binary = list(correct_binary)
    partially_correct_binary[0] = "1" if partially_correct_binary[0] == "0" else "0"
    partially_correct_binary = "".join(partially_correct_binary)
    partial_answer = task.class_name_map[partially_correct_binary]

    score_b = task.score_response(q, partial_answer)
    print(f"Agent Response: '{partial_answer}'")
    print(f"Score: {score_b:.2f} (Expected {(task.n_attributes-1)/task.n_attributes})")

    feedback_b = task.extract_feedback_info(q, partial_answer)
    print(
        f"Feedback (Corrective): 'Actually, the right answer was {feedback_b[0]}. One reason is that {feedback_b[1][0]} {feedback_b[1][1]} {feedback_b[1][2]:.3f}.'"
    )

    # Scenario C: Agent gets it right
    print("\n--- SCENARIO C: CORRECT GUESS ---")
    correct_answer = q.correct_answer
    score_c = task.score_response(q, correct_answer)
    print(f"Agent Response: '{correct_answer}'")
    print(f"Score: {score_c:.2f} (Expected 1.0)")

    feedback_c = task.extract_feedback_info(q, correct_answer)
    print(
        f"Feedback (Bonus Info): 'Correct! To help you further, here is a rule for another disease, {feedback_c[0]}: {feedback_c[1][0]} {feedback_c[1][1]} {feedback_c[1][2]:.3f}.'"
    )

    print(task.disease_classes)

# %%
import random
from typing import Dict, Tuple, List, Optional

# We assume the DiagnosisTask and Question classes are defined elsewhere
# from the previous code blocks.


class DummyDiagnosisAgent:
    """
    A non-LLM agent for the DiagnosisTask that serves as a baseline.

    It maintains a structured knowledge base of rules and uses a simple
    "best match" algorithm to make a diagnosis. It does not use a language model.
    """

    def __init__(
        self,
        model: ChatOpenAI,
        classes: List[str],
        task_id: str = "AlienDiag-01",
    ):
        """
        Initializes the agent. It needs the task object to know the possible
        outcomes (diseases) for random guessing.
        """
        self.model = model
        self.knowledge_base: List[str] = []
        self.task_id = task_id
        self.classes = classes
        self.system_prompt = DIAGNOSIS_AGENT_SYSTEM.format(diagnosis_options=classes)

    def probe_task(self, question: Question) -> str:
        return random.choice(self.classes)

    def update_knowledge_base(
        self, feedback_info: Optional[Tuple[str, Tuple[str, str, float]]]
    ):
        """
        Updates the knowledge base with a new diagnostic rule.
        Feedback format: (disease_name, (attribute, operator, threshold))
        """
        if feedback_info is None:
            return

        disease, rule = feedback_info
        attribute, op, threshold = rule

        # Format the rule into a human-readable string for the prompt
        rule_string = (
            f"{disease}: {attribute} {op} {threshold:.3f}"
        )

        # Add the rule if it's new
        if rule_string not in self.knowledge_base:
            self.knowledge_base.append(rule_string)


# %%
from langchain.schema import SystemMessage, HumanMessage
from typing import List, Tuple, Optional
from langchain_openai import ChatOpenAI
import re
import numpy as np

# A more detailed system prompt to guide the LLM's reasoning process.
DIAGNOSIS_AGENT_SYSTEM = """You are an alien doctor. Your goal is to diagnose an alien's disease based on its biological readings. The disease is represented by a unique 3-letter code. The full list of possible diseases are:
{diagnosis_options}

As you gain experience in diagnosing aliens, you will accumulate knowledge within a knowledge base. Each rule tells you something about the conditions required for a specific disease:

Your thought process to diagnose a new patient should be:
1.  Review the patient's readings provided in the question.
2.  Try to diagnose the patient, by comparing these readings against your knowledge base.
3.  If no perfect match is found, make your best guess based on the available information.

When you have made your diagnosis, respond with ONLY the 'disease_' prefix with 3-letter disease code, and nothing else.
Example Response: disease_ZYX
"""

DIAGNOSIS_AGENT_PROMPT = """This is your current knowledge base of diagnostic rules:
{knowledge_base}\n"""


class DiagnosisAgent:
    def __init__(
        self,
        model: ChatOpenAI,
        classes: List[str],
        task_id: str = "AlienDiag-01",
    ):
        self.model = model
        self.knowledge_base = {}  # Dict[disease_name, List[rule_strings]]
        self.task_id = task_id
        self.classes = classes
        self.system_prompt = DIAGNOSIS_AGENT_SYSTEM.format(diagnosis_options=classes)

    def probe_task(self, question: Question) -> str:
        """
        Takes a question, formats a prompt with current knowledge,
        and returns the model's parsed response.
        """
        # Format the knowledge base for display in the prompt
        kb_text = self._format_knowledge_base()

        # Get the question text from the task
        agent_prompt = DIAGNOSIS_AGENT_PROMPT.format(knowledge_base=kb_text)

        question_text = question.question_text
        
        llm_input = agent_prompt + question_text
        
        print(llm_input)

        # Invoke the LLM
        response = self.model.invoke(
            [
                SystemMessage(self.system_prompt),
                HumanMessage(llm_input),
            ]
        )
        model_response = response.content.strip()

        # Parse the 3-letter code from the response
        parsed_response = self._parse_response(model_response)

        return parsed_response

    def update_knowledge_base(self, feedback_info):
        if feedback_info is None:
            return

        disease, (attribute, op, threshold) = feedback_info
        rule_string = f"{attribute} {op} {threshold:.3f}"
        
        if disease not in self.knowledge_base:
            self.knowledge_base[disease] = []
        
        if rule_string not in self.knowledge_base[disease]:
            self.knowledge_base[disease].append(rule_string)

    def _format_knowledge_base(self) -> str:
        if not self.knowledge_base:
            return "No diagnostic rules learned yet."
        
        formatted = []
        for disease, rules in self.knowledge_base.items():
            formatted.append(f"{disease}:")
            for rule in rules:
                formatted.append(f"  - {rule}")
            formatted.append("")
        
        return "\n".join(formatted).strip()

    def _parse_response(self, response: str) -> str:
        """
        Parses the model's response to extract a 3-letter disease code.
        """
        # Find the first occurrence of a 3-letter uppercase word
        match = re.search(r"\b[A-Z]{3}\b", response)

        if match:
            return match.group(0)

        # Fallback for cases like ['XYZ'] or "XYZ"
        match = re.search(r"[A-Z]{3}", response)
        if match:
            return match.group(0)

        return ""  # Return empty string if no valid code is found


# %%
# Assuming DiagnosisTask and Question classes are defined in the same file or imported


def train_agent_on_diagnosis_task(
    agent: DiagnosisAgent,
    task: DiagnosisTask,
    num_iterations: int = 20,
    batch_size: int = 1,
):
    """
    Training loop for the DiagnosisAgent on a specific DiagnosisTask.
    """
    all_scores = []

    for i in range(num_iterations):
        iter_scores = []
        print(f"\n--- Iteration {i+1} / {num_iterations} ---")
        print(f"Current Knowledge Base Size: {len(agent.knowledge_base)}")

        for j in range(batch_size):
            # 1. Generate a probe question from the task
            question = task.generate_question()

            # 2. Agent attempts to answer
            agent_response = agent.probe_task(question)

            # 3. Score the response
            score = task.score_response(question, agent_response)
            iter_scores.append(score)

            print(f"  - Probe {j+1}: Patient readings: {question.question_data}")
            print(
                f"    Agent Response: '{agent_response}' | Correct Answer: '{question.correct_answer}' | Score: {score:.2f}"
            )

            # 4. Extract feedback based on the agent's answer and update knowledge base
            feedback = task.extract_feedback_info(question, agent_response)
            if feedback:
                is_correct = agent_response == question.correct_answer
                feedback_type = "Bonus Info" if is_correct else "Correction"
                print(
                    f"    Feedback ({feedback_type}): For disease '{feedback[0]}', rule is {feedback[1]}"
                )
                agent.update_knowledge_base(feedback)

        all_scores.append(iter_scores)

    return np.array(all_scores)


def run_diagnosis_experiment(
    model: ChatOpenAI,
    n_attributes: int = 3,
    n_iterations: int = 25,
    batch_size: int = 1,
):
    """
    Sets up and runs a full diagnosis experiment.
    """
    print("--- Starting Diagnosis Experiment ---")
    print(
        f"Config: {n_attributes} attributes, {2**n_attributes} diseases, {n_iterations} iterations"
    )

    # Initialize the task environment
    task = DiagnosisTask(task_id=1, n_attributes=n_attributes)
    task.generate_ground_truth(seed=42)

    # Initialize the agent
    agent = DiagnosisAgent(model=model, classes=task.disease_classes)

    # Run the training loop
    scores = train_agent_on_diagnosis_task(
        agent, task, num_iterations=n_iterations, batch_size=batch_size
    )

    print("\n--- Experiment Finished ---")
    print(f"Final knowledge base size: {len(agent.knowledge_base)}")

    return scores


import matplotlib.pyplot as plt

# You need to have your model initialization utility
# from utils import init_openrouter_chat_model, init_azure_model

# Assuming the training loop function `train_agent_on_diagnosis_task`
# and the `DiagnosisTask` class are already defined.
import numpy as np
import matplotlib.pyplot as plt


def run_diagnosis_experiment_with_dummy(
    n_attributes: int = 3, n_iterations: int = 25, batch_size: int = 1
):
    """
    Sets up and runs a full diagnosis experiment with the DummyDiagnosisAgent.
    """
    print("--- Starting Diagnosis Experiment (DUMMY AGENT) ---")
    print(
        f"Config: {n_attributes} attributes, {2**n_attributes} diseases, {n_iterations} iterations"
    )

    # Initialize the task environment
    task = DiagnosisTask(task_id=1, n_attributes=n_attributes)
    task.generate_ground_truth(seed=42)

    # Initialize the DUMMY agent, passing the task to it
    agent = DummyDiagnosisAgent(model=None, classes=task.disease_classes)

    print(agent.system_prompt)

    # The same training loop works without any changes
    scores = train_agent_on_diagnosis_task(
        agent, task, num_iterations=n_iterations, batch_size=batch_size
    )

    print("\n--- Experiment Finished ---")
    print(
        f"Final knowledge base state: {len(agent.knowledge_base)} diseases learned about."
    )

    return scores


if __name__ == "__main__":
    # --- RUN THE EXPERIMENT WITH THE DUMMY AGENT ---
    model = init_azure_model()
    scores = run_diagnosis_experiment(
        model=model,
        n_attributes=3,  
        n_iterations=50,  # More iterations to see learning
        batch_size=1,
    )

    # --- PLOT THE RESULTS ---
    mean_scores = np.mean(scores, axis=1)
    # Using a rolling average to smooth the curve and see the trend more clearly
    rolling_avg = np.convolve(mean_scores, np.ones(5) / 5, mode="valid")
    iterations = range(1, len(scores) + 1)

    plt.figure(figsize=(12, 7))
    plt.plot(iterations, mean_scores, "-", alpha=0.5, label="Raw Score per Iteration")
    plt.plot(
        range(3, len(rolling_avg) + 3),
        rolling_avg,
        color="red",
        linewidth=2,
        label="5-Iteration Rolling Average",
    )

    plt.title("Dummy Agent Learning Curve on Diagnosis Task")
    plt.xlabel("Iteration")
    plt.ylabel("Score (Proportion of Correct Decisions)")
    plt.ylim(0, 1.05)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.show()
# %%
