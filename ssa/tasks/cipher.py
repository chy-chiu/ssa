# %%
import random
import string
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from ssa.tasks.task import TaskBase, Question
from langchain.schema import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from langchain_core.output_parsers import JsonOutputParser
from loguru import logger
from ssa.tasks.task import TaskSubAgent, TaskRunner

class CipherResponse(BaseModel):

    reasoning: str = Field(description="Your reasoning for this choice")
    answer: List[str] = Field(
        description="List of decrypted words. Reply with your best guess if information is incomplete."
    )

class CipherTask(TaskBase):
    """
    A cryptogram decryption task. The agent must decrypt messages using a fixed
    substitution cipher. Ground truth is a letter-to-letter mapping (A->X, B->Y, etc.)
    """

    def __init__(self, task_id: int = 1):
        super().__init__(task_id)
        self.cipher_mapping: Dict[str, str] = {}  # A->X, B->Y, etc.
        self.reverse_mapping: Dict[str, str] = {}  # X->A, Y->B, etc.

        words_path = Path(__file__).resolve().parents[1] / "assets" / "words.txt"
        with words_path.open("r", encoding="utf-8") as f:
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

    def generate_question(self, benchmark=False, batch_size=3) -> Question:
        """Generate a batch of random 5 letter strings to decrypt"""
        if not self.cipher_mapping:
            raise RuntimeError("Generate ground truth first")

        plaintexts = []
        ciphertexts = []

        for _ in range(batch_size):
            # plaintext = "".join(random.choices(string.ascii_uppercase, k=length))
            plaintext = random.choice(self.words)
            ciphertext = self._encrypt_message(plaintext)

            plaintexts.append(plaintext)
            ciphertexts.append(ciphertext)

        question_text = f"Decrypt the following messages: {', '.join(ciphertexts)}. "

        return Question(
            question_text=question_text,
            question_data=ciphertexts,
            correct_answer=plaintexts,
        )

    def score_response(
        self, question: Question, agent_response: CipherResponse
    ) -> float:
        """Score based on average proportion of correct letters across all messages"""

        # Parse agent response - assuming comma-separated format
        try:
            responses = [str(r).strip().upper() for r in agent_response.answer]
        except Exception as e:
            logger.warning(f"Unable to parse agent_response: {agent_response}: {e}")
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
                matches = sum(
                    1 for i in range(len(correct)) if correct[i] == response[i]
                )
                individual_score = matches / len(correct)

            total_score += individual_score

        return round(total_score / len(correct_answers), 3)

    def extract_feedback_info(
        self, question: Question, agent_response: CipherResponse
    ) -> Optional[Tuple[str, str]]:
        """Return first incorrect letter, or random if all correct"""
        response = "".join(agent_response.answer)
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

class CipherAgent(TaskSubAgent):
    def __init__(self, model: ChatOpenAI, task_id: str = "Crypto-01"):
        self.model = model
        self.parser = JsonOutputParser(pydantic_object=CipherResponse)

        self.knowledge_base: Dict[str, str] = (
            {}
        )  # Known mappings: cipher_letter -> plain_letter
        self.task_id = task_id
        self.system_prompt = """This is a mock test that simulates medical coding. Your goal is to code a set of text into a word using a substitution cipher that is unknown to you initially.
You will receive encrypted messages and must return the original plaintext.

Use your knowledge of known letter mappings to help decrypt new messages.
If you don't know all the mappings, make educated guesses.
Respond with reasoning and the decrypted text. You should provide with an answer even if you guess only.

Reply in JSON output format only and nothing else. Format as below:
{format_instructions}""".format(
            format_instructions=self.parser.get_format_instructions()
        )

        self.token_usage = []
        self.trace = []

    def run_task(self, question: Question) -> CipherResponse:
        
        # TODO: Move this one level up + add retries (???) maybe... 
        """Attempt to decrypt the ciphertext using known mappings"""
        kb_text = self._format_knowledge_base()

        prompt = f"""Known cipher mappings:
(encrypted letters → decrypted letters)
{kb_text}

{question.question_text}"""

        # print(prompt)

        response = self.model.invoke(
            [SystemMessage(self.system_prompt), HumanMessage(prompt)]
        )

        self.trace.append((prompt, response.content))
        self.token_usage.append(response.response_metadata["token_usage"])

        return CipherResponse.model_validate(self.parser.parse(response.content))

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
            mappings.append(f"({cipher_char} → {plain_char})")

        return "\n".join(mappings)


# Example usage
if __name__ == "__main__":
    task = CipherTask(task_id=1)
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
    agent_response = CipherResponse(reasoning="", answer=['LIMEN' ,'REDAL', "ALAHD"])  # Random guess
    score = task.score_response(q, agent_response)
    feedback = task.extract_feedback_info(q, agent_response)

    print(f"Agent Response: {agent_response}")
    print(f"Score: {score:.2f}")
    print(f"Feedback: {feedback[1]} → {feedback[0]}")

# # %%
# from ssa.utils import OpenAIClient
# model = OpenAIClient(model_name="gpt-5-cc", secrets_path='../assets/secrets.yaml', effort='minimal')

# subagent = CipherAgent(model)

# task = CipherTask(task_id=1)

# task.generate_ground_truth(seed=42)

# runner = TaskRunner(subagent=subagent, task=task)
# # %%

# q = task.generate_question()

# model = OpenAIClient(model_name="gpt-5-cc", secrets_path='../assets/secrets.yaml', effort='minimal')
# subagen_a = CipherAgent(model)
# r_a = subagen_a.run_task(q)
# print(r_a)

# model = OpenAIClient(model_name="gpt-5-cc", secrets_path='../assets/secrets.yaml', effort='low')
# subagen_b = CipherAgent(model)
# r_b = subagen_b.run_task(q)
# print(r_b)

# model = OpenAIClient(model_name="gpt-5-cc", secrets_path='../assets/secrets.yaml', effort='medium')
# subagen_c = CipherAgent(model)
# r_c = subagen_c.run_task(q)
# print(r_c)

# model = OpenAIClient(model_name="gpt-5-cc", secrets_path='../assets/secrets.yaml', effort='high')
# subagen_d = CipherAgent(model)
# r_d = subagen_d.run_task(q)
# print(r_d)

# # %%
# subagen_d.token_usage
# # %%
# subagent.trace

# # %%
# from tqdm import trange
# scores = [] 
# for _ in trange(50):
#     scores.append(runner.perform_task())
# # %%
# import matplotlib.pyplot as plt
# plt.plot(scores)
# %%


# # %%
# from utils import init_azure_model, init_openrouter_chat_model

# model = init_azure_model(temperature=0.1)
# # model = init_openrouter_chat_model(model_name="google/gemini-2.5-flash", temperature=0.5)

# task = CipherTask(task_id=1)
# task.generate_ground_truth(seed=42)

# agent = CipherAgent(model)

# # skill_level = []
# # for _ in range(50):
# #     agent.update_knowledge_base(task.get_random_feedback())
    

# #     skill_level.append(agent.skill_level)

# # %%
# task.generate_ground_truth(1)
# task.cipher_mapping

# # %%
# task.generate_ground_truth()


# # %%
# task = CipherTask(task_id=1)
# task.generate_ground_truth(seed=42)

# all_scores = {}

# from tqdm import trange

# # check over different ranges of i:
# for i in [5, 50]:
#     round_scores = []

#     agent = CipherAgent(model)

#     for _ in range(i):
#         agent.update_knowledge_base(task.get_random_feedback())

#     agent_knowledge_level = len(agent.knowledge_base)

#     for _ in trange(10):
        
#         question = task.generate_question(batch_size=3)
#         agent_response = agent.probe_task(question)
#         round_scores.append(task.score_response(question, agent_response))
    
#     all_scores[agent_knowledge_level] = round_scores

# print(question.question_data, agent_response.answer, question.correct_answer)
# # %%
# import matplotlib.pyplot as plt
# import numpy as np

# # all_scores = {1: [0.0, 0.2, 0.0, 0.0, 0.0, 0.2, 0.0, 0.0, 0.2, 0.0],
# #  4: [0.0, 0.0, 0.0, 0.4, 0.0, 0.0, 0.2, 0.2, 0.4, 0.0],
# #  10: [0.0, 0.0, 0.6, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.6],
# #  11: [0.2, 0.2, 0.6, 0.0, 0.6, 0.4, 0.0, 0.0, 0.2, 0.0],
# #  13: [0.0, 0.0, 0.4, 0.6, 0.2, 0.6, 0.6, 0.8, 0.4, 0.6],
# #  24: [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.8, 1.0, 1.0],
# #  26: [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]}
# # %%
# import numpy as np
# X = all_scores.keys()
# scores = np.array(list(all_scores.values()))

# plt.plot(X, np.mean(scores, axis=1), label=f"S={+1}")
# plt.fill_between(
#     X, 
#     np.mean(scores, axis=1) - np.std(scores, axis=1),
#     np.mean(scores, axis=1) + np.std(scores, axis=1),
#     alpha=0.3,

# %%
