# %%
import random
import string
from typing import Dict, List, Tuple, Optional
from task import TaskBase, Question
from langchain.schema import SystemMessage, HumanMessage


class CryptogramTask(TaskBase):
    """
    A cryptogram decryption task. The agent must decrypt messages using a fixed
    substitution cipher. Ground truth is a letter-to-letter mapping (A->X, B->Y, etc.)
    """

    def __init__(self, task_id: int = 1):
        super().__init__(task_id)
        self.cipher_mapping: Dict[str, str] = {}  # A->X, B->Y, etc.
        self.reverse_mapping: Dict[str, str] = {}  # X->A, Y->B, etc.
        
        with open('words.txt', 'r') as f:
            self.words = [s.upper().strip("\n") for s in f.readlines()]

    def generate_ground_truth(self, seed: int = None):
        """Generate a random substitution cipher mapping"""
        if seed is not None:
            random.seed(seed)

        letters = list(string.ascii_uppercase)
        shuffled = letters.copy()
        random.shuffle(shuffled)

        self.cipher_mapping = dict(zip(letters, shuffled))
        self.reverse_mapping = dict(zip(shuffled, letters))

    def _encrypt_message(self, plaintext: str) -> str:
        """Encrypt a message using the cipher mapping"""
        return "".join(self.cipher_mapping[char.upper()] for char in plaintext)

    def _decrypt_message(self, ciphertext: str) -> str:
        """Decrypt a message using the reverse mapping"""
        return "".join(self.reverse_mapping[char.upper()] for char in ciphertext)

    def generate_question(self, batch_size=5) -> Question:
        """Generate a batch of random 5 letter strings to decrypt"""
        if not self.cipher_mapping:
            raise RuntimeError("Generate ground truth first")

        length = 5
        plaintexts = []
        ciphertexts = []

        for _ in range(batch_size):
            # plaintext = "".join(random.choices(string.ascii_uppercase, k=length))
            plaintext = random.choice(self.words)
            ciphertext = self._encrypt_message(plaintext)

            plaintexts.append(plaintext)
            ciphertexts.append(ciphertext)

        question_text = (
            f"Decrypt the following messages: {', '.join(ciphertexts)}. "
            f"Respond with only a list of the decrypted text, separated by commas."
        )

        return Question(
            question_text=question_text,
            question_data=ciphertexts,
            correct_answer=plaintexts,
        )
    
    def parse_agent_response(self, agent_response: str):
        return [r.strip().replace("[", "").replace("]", "").replace("'", "").replace('"', "").upper() for r in agent_response.split(',')]
            

    def score_response(self, question: Question, agent_response: str) -> float:
        """Score based on average proportion of correct letters across all messages"""
        
        # Parse agent response - assuming comma-separated format
        try:
            if isinstance(agent_response, str):
                responses = self.parse_agent_response(agent_response)
            else:
                responses = [str(r).strip().upper() for r in agent_response]
        except:
            return 0.0
        
        correct_answers = question.correct_answer
        if isinstance(correct_answers, str):
            correct_answers = [correct_answers]
        
        # Check if we have matching number of responses
        if len(responses) != len(correct_answers):
            return 0.0
        
        total_score = 0.0
        for response, correct in zip(responses, correct_answers):
            correct = correct.upper()
            
            if len(response) != len(correct):
                # This individual response gets 0 score
                individual_score = 0.0
            else:
                matches = sum(1 for i in range(len(correct)) if correct[i] == response[i])
                individual_score = matches / len(correct)
                
            total_score += individual_score
        
        return total_score / len(correct_answers)

    def extract_feedback_info(
        self, question: Question, agent_response: str
    ) -> Optional[Tuple[str, str]]:
        """Return first incorrect letter, or random if all correct"""
        response = "".join(self.parse_agent_response(agent_response))
        correct = "".join(question.correct_answer)
        for i in range(len(correct)):
            if correct[i] != response[i]:
                return (correct[i], self.cipher_mapping[correct[i]])
        return self.get_random_feedback()

    def get_random_feedback(self):
        """Return a random cipher mapping as feedback"""

        input_letter = random.choice(string.ascii_uppercase)
        mapped_letter = self.cipher_mapping[input_letter]
        return (input_letter, mapped_letter)  # (plaintext_letter, cipher_letter)


class CryptogramAgent:
    def __init__(self, model, task_id: str = "Crypto-01"):
        self.model = model
        self.knowledge_base: Dict[str, str] = (
            {}
        )  # Known mappings: cipher_letter -> plain_letter
        self.task_id = task_id
        self.system_prompt = """You are a codebreaker. Your goal is to decrypt substitution ciphers.
You will receive encrypted messages and must return the original plaintext.

Use your knowledge of known letter mappings to help decrypt new messages.
If you don't know all the mappings, make educated guesses.
Respond with ONLY the decrypted text."""

    def probe_task(self, question: Question) -> str:
        """Attempt to decrypt the ciphertext using known mappings"""
        kb_text = self._format_knowledge_base()

        prompt = f"""Known cipher mappings:
{kb_text}

{question.question_text}"""

        response = self.model.invoke(
            [SystemMessage(self.system_prompt), HumanMessage(prompt)]
        )

        return response.content.strip().upper()

    def update_knowledge_base(self, feedback_info: Optional[Tuple[str, str]]):
        """Update with new cipher mapping: (plain_letter, cipher_letter)"""
        if feedback_info is None:
            return

        plain_letter, cipher_letter = feedback_info
        # Store as cipher -> plain for decryption
        self.knowledge_base[cipher_letter] = plain_letter

    def _format_knowledge_base(self) -> str:
        """Format known mappings for the prompt"""
        if not self.knowledge_base:
            return "No mappings known yet."

        mappings = []
        for cipher_char in sorted(self.knowledge_base.keys()):
            plain_char = self.knowledge_base[cipher_char]
            mappings.append(f"  {cipher_char} → {plain_char}")

        return "\n".join(mappings)


# Example usage
if __name__ == "__main__":
    task = CryptogramTask(task_id=1)
    task.generate_ground_truth(seed=42)

    print("Ground Truth Cipher Mapping (first 10):")
    for i, (plain, cipher) in enumerate(task.cipher_mapping.items()):
        if i < 10:
            print(f"  {cipher} → {plain}")

    # Generate a question
    q = task.generate_question()
    print(f"\nQuestion: {q.question_text}")
    print(f"Correct Answer: {q.correct_answer}")

    # Simulate agent response and feedback
    agent_response = "['PVAUS', 'IECDE', 'ABABA', 'NZJVV', 'ASDD']"  # Random guess
    score = task.score_response(q, agent_response)
    feedback = task.extract_feedback_info(q, agent_response)

    print(f"Agent Response: {agent_response}")
    print(f"Score: {score:.2f}")
    print(f"Feedback: {feedback[1]} → {feedback[0]}")


# %%
from utils import init_azure_model

model = init_azure_model(temperature=1)

agent = CryptogramAgent(model)

task = CryptogramTask(task_id=1)
task.generate_ground_truth(seed=42)

for _ in range(100):
    agent.update_knowledge_base(task.get_random_feedback())

question = task.generate_question(batch_size=1)
agent_response = agent.probe_task(question)

print(question.question_data, agent_response, question.correct_answer)
task.score_response(question, agent_response)
# %%
question
# %%
agent.knowledge_base
# %%
print(agent._format_knowledge_base())
# %%
print(question.question_text)
# %%
task.words
# %%
model.invoke("Are you GPT-4o or o3")
# %%
