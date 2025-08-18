# %%
import random
import string
from typing import Dict, List, Tuple, Optional
from task import TaskBase, Question

class CryptogramTask(TaskBase):
    """
    A cryptogram decryption task. The agent must decrypt messages using a fixed
    substitution cipher. Ground truth is a letter-to-letter mapping (A->X, B->Y, etc.)
    """
    
    def __init__(self, task_id: int = 1):
        super().__init__(task_id)
        self.cipher_mapping: Dict[str, str] = {}  # A->X, B->Y, etc.
        self.reverse_mapping: Dict[str, str] = {}  # X->A, Y->B, etc.
    
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
        return ''.join(self.cipher_mapping[char.upper()] for char in plaintext)
    
    def _decrypt_message(self, ciphertext: str) -> str:
        """Decrypt a message using the reverse mapping"""
        return ''.join(self.reverse_mapping[char.upper()] for char in ciphertext)
    
    def generate_probe_question(self) -> Question:
        """Generate a random 3-5 letter string to decrypt"""
        if not self.cipher_mapping:
            raise RuntimeError("Generate ground truth first")
        
        # Generate random 3-5 letter string
        length = random.randint(3, 5)
        plaintext = ''.join(random.choices(string.ascii_uppercase, k=length))
        ciphertext = self._encrypt_message(plaintext)
        
        question_text = (
            f"Decrypt this message: '{ciphertext}'. "
            f"Respond with only the decrypted text."
        )
        
        return Question(question_text=question_text, question_data=ciphertext, correct_answer=plaintext)
    
    def score_response(self, question: Question, agent_response: str) -> float:
        """Score based on proportion of correct letters"""
        correct = question.correct_answer.upper()
        response = agent_response.upper().strip()
        
        if len(response) != len(correct):
            return 0.0
        
        matches = sum(1 for i in range(len(correct)) if correct[i] == response[i])
        return matches / len(correct)
    
    def extract_feedback_info(self, question: Question, agent_response: str) -> Optional[Tuple[str, str]]:
        """Return a random cipher mapping as feedback"""
        # Pick a random letter from the alphabet to reveal its mapping
        letter = random.choice(string.ascii_uppercase)
        mapped_letter = self.cipher_mapping[letter]
        return (letter, mapped_letter)  # (plaintext_letter, cipher_letter)


class CryptogramAgent:
    def __init__(self, model, task_id: str = "Crypto-01"):
        self.model = model
        self.knowledge_base: Dict[str, str] = {}  # Known mappings: cipher_letter -> plain_letter
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
        
        response = self.model.invoke([
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt}
        ])
        
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
            print(f"  {plain} → {cipher}")
    
    # Generate a question
    q = task.generate_probe_question()
    print(f"\nQuestion: {q.question_text}")
    print(f"Correct Answer: {q.correct_answer}")
    
    # Simulate agent response and feedback
    agent_response = "HWEAA"  # Random guess
    score = task.score_response(q, agent_response)
    feedback = task.extract_feedback_info(q, agent_response)
    
    print(f"Agent Response: {agent_response}")
    print(f"Score: {score:.2f}")
    print(f"Feedback: {feedback[0]} → {feedback[1]}")

# %%
