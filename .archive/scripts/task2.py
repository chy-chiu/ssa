# %%
import random
import networkx as nx
from typing import Dict, List, Tuple, Set, Any
from dataclasses import dataclass
from abc import ABC, abstractmethod
from enum import Enum

@dataclass
class TaskKnowledge:
    task_id: str
    known_relations: Set[Tuple[str, str]]  # (A, B) means A > B
    confidence_scores: Dict[Tuple[str, str], float]

@dataclass
class Question:
    question_text: str
    question_data: Any  # Task-specific data needed for scoring
    correct_answer: Any

@dataclass
class BatchResult:
    questions: List[Question]
    agent_responses: List[Any]
    scores: List[bool]
    feedback_data: List[Any]

class Task(ABC):
    """Abstract base class for all task types"""
    
    def __init__(self, task_id: str):
        self.task_id = task_id
        self.ground_truth = None
    
    @abstractmethod
    def generate_ground_truth(self, seed: int = None):
        """Generate deterministic ground truth for this task"""
        pass
    
    @abstractmethod
    def generate_probe_question(self) -> Question:
        """Generate a single probe question"""
        pass
    
    @abstractmethod
    def score_response(self, question: Question, agent_response: Any) -> bool:
        """Score agent's response to a question"""
        pass
    
    @abstractmethod
    def extract_feedback_info(self, question: Question) -> Any:
        """Extract information needed for agent feedback"""
        pass

class OrderingTask(Task):
    """Concrete task for learning orderings between elements"""
    
    def __init__(self, task_id: str, elements: List[str]):
        super().__init__(task_id)
        self.elements = elements
        self.ground_truth: nx.DiGraph = None
    
    def generate_ground_truth(self, seed: int = None) -> nx.DiGraph:
        """Generate deterministic total ordering"""
        if seed is not None:
            random.seed(seed + hash(self.task_id))  # Make seed unique per task
        
        # Create random total ordering
        shuffled = self.elements.copy()
        random.shuffle(shuffled)
        
        # Build directed graph representing total order
        G = nx.DiGraph()
        for i in range(len(shuffled)):
            for j in range(i+1, len(shuffled)):
                G.add_edge(shuffled[i], shuffled[j])  # shuffled[i] > shuffled[j]
        
        self.ground_truth = G
        return G
    
    def generate_probe_question(self) -> Question:
        """Generate ordering comparison question"""
        if self.ground_truth is None:
            raise ValueError(f"Ground truth not generated for {self.task_id}")
        
        nodes = list(self.ground_truth.nodes())
        a, b = random.sample(nodes, 2)
        
        correct_answer = self.ground_truth.has_edge(a, b)  # Is a > b?
        question_text = f"You are currently on {self.task_id}. Does {a} > {b}?"
        
        return Question(
            question_text=question_text,
            question_data={'relation': (a, b), 'task_id': self.task_id},
            correct_answer=correct_answer
        )
    
    def score_response(self, question: Question, agent_response: Any) -> bool:
        """Score T/F response for ordering question"""
        # Convert agent response to boolean
        if isinstance(agent_response, str):
            agent_bool = agent_response.strip().upper() == 'T'
        else:
            agent_bool = bool(agent_response)
        
        return agent_bool == question.correct_answer
    
    def extract_feedback_info(self, question: Question) -> Dict:
        """Extract relation and correct answer for feedback"""
        return {
            'relation': question.question_data['relation'],
            'correct_answer': question.correct_answer,
            'task_id': question.question_data['task_id']
        }

class ExaminationSystem:
    """System for managing tasks, generating questions, and providing feedback"""
    
    def __init__(self):
        self.tasks: Dict[str, Task] = {}
        self.agent_knowledge: Dict[str, TaskKnowledge] = {}
    
    def add_task(self, task: Task):
        """Add a task to the system"""
        self.tasks[task.task_id] = task
    
    def initialize_task_knowledge(self, task_id: str, 
                                initial_relations: List[Tuple[str, str]] = None):
        """Initialize agent's knowledge for a task"""
        if initial_relations is None:
            initial_relations = []
        
        self.agent_knowledge[task_id] = TaskKnowledge(
            task_id=task_id,
            known_relations=set(initial_relations),
            confidence_scores={rel: 1.0 for rel in initial_relations}
        )
    
    def generate_question_batch(self, task_id: str, batch_size: int) -> List[Question]:
        """Generate batch of questions for a specific task"""
        if task_id not in self.tasks:
            raise ValueError(f"Task {task_id} not found")
        
        task = self.tasks[task_id]
        return [task.generate_probe_question() for _ in range(batch_size)]
    
    def score_batch(self, questions: List[Question], 
                   agent_responses: List[Any]) -> BatchResult:
        """Score a batch of responses"""
        if len(questions) != len(agent_responses):
            raise ValueError("Mismatch between questions and responses")
        
        scores = []
        feedback_data = []
        
        for question, response in zip(questions, agent_responses):
            task_id = question.question_data['task_id']
            task = self.tasks[task_id]
            
            score = task.score_response(question, response)
            feedback_info = task.extract_feedback_info(question)
            
            scores.append(score)
            feedback_data.append(feedback_info)
        
        return BatchResult(
            questions=questions,
            agent_responses=agent_responses,
            scores=scores,
            feedback_data=feedback_data
        )
    
    def provide_batch_feedback(self, batch_result: BatchResult):
        """Update agent knowledge based on batch results"""
        for question, score, feedback_info in zip(
            batch_result.questions, batch_result.scores, batch_result.feedback_data
        ):
            task_id = feedback_info['task_id']
            relation = feedback_info['relation']
            correct_answer = feedback_info['correct_answer']
            
            self._update_task_knowledge(task_id, relation, correct_answer)
    
    def _update_task_knowledge(self, task_id: str, relation: Tuple[str, str], 
                              correct_answer: bool):
        """Update knowledge for a specific task"""
        if task_id not in self.agent_knowledge:
            self.initialize_task_knowledge(task_id)
        
        knowledge = self.agent_knowledge[task_id]
        a, b = relation
        
        # Add correct relation
        if correct_answer:
            knowledge.known_relations.add((a, b))
            knowledge.confidence_scores[(a, b)] = 1.0
        else:
            knowledge.known_relations.add((b, a))  # b > a instead
            knowledge.confidence_scores[(b, a)] = 1.0
        
        # Infer transitive relations
        self._infer_transitive_relations(task_id)
    
    def _infer_transitive_relations(self, task_id: str):
        """Infer new relations through transitivity"""
        knowledge = self.agent_knowledge[task_id]
        
        # Build temporary graph from known relations
        temp_G = nx.DiGraph()
        for a, b in knowledge.known_relations:
            temp_G.add_edge(a, b)
        
        # Find transitive closure
        try:
            closure = nx.transitive_closure(temp_G)
            
            # Add inferred relations with lower confidence
            for edge in closure.edges():
                if edge not in knowledge.known_relations:
                    knowledge.known_relations.add(edge)
                    knowledge.confidence_scores[edge] = 0.7  # Lower confidence
        except:
            pass  # Handle potential cycles gracefully
    
    def get_agent_system_prompt(self, task_id: str) -> str:
        """Generate system prompt with current knowledge"""
        if task_id not in self.agent_knowledge:
            known_rels = []
        else:
            known_rels = list(self.agent_knowledge[task_id].known_relations)
        
        knowledge_str = ", ".join([f"({a} > {b})" for a, b in known_rels])
        
        return f"""You are a case triage agent, where your main goal is to solve different tasks.

This is your current knowledge base:
{task_id}: {knowledge_str}

You are to reply in T/F only."""
    
    def get_task_performance(self, task_id: str) -> Dict:
        """Get performance metrics for a task"""
        if task_id not in self.agent_knowledge:
            return {"known_relations": 0, "confidence_avg": 0.0}
        
        knowledge = self.agent_knowledge[task_id]
        return {
            "known_relations": len(knowledge.known_relations),
            "confidence_avg": sum(knowledge.confidence_scores.values()) / 
                            len(knowledge.confidence_scores) if knowledge.confidence_scores else 0.0
        }

# Example usage
def setup_example_system():
    exam_system = ExaminationSystem()
    
    # Create 10 ordering tasks
    task_configs = [
        ("Task_A", ["A", "B", "C", "D"]),
        ("Task_B", ["A", "B", "E", "F"]),
        ("Task_C", ["A", "B", "C", "D"]),
        ("Task_D", ["W", "X", "Y", "Z"]),
        ("Task_E", ["P", "Q", "R", "S"]),
        ("Task_F", ["M", "N", "O", "P"]),
        ("Task_G", ["U", "V", "W", "X"]),
        ("Task_H", ["I", "J", "K", "L"]),
        ("Task_I", ["T", "U", "V", "W"]),
        ("Task_J", ["G", "H", "I", "J"])
    ]
    
    # Create and add tasks
    for task_id, elements in task_configs:
        task = OrderingTask(task_id, elements)
        task.generate_ground_truth(seed=42)  # Deterministic
        exam_system.add_task(task)
        exam_system.initialize_task_knowledge(task_id)
    
    return exam_system

# Demo
exam_system = setup_example_system()

# Generate batch of questions for Task_A
questions = exam_system.generate_question_batch("Task_A", batch_size=3)

# %%
questions
# %% 

print("Generated questions:")
for i, q in enumerate(questions):
    print(f"{i+1}. {q.question_text} (Correct: {q.correct_answer})")

# Simulate agent responses
agent_responses = [True, False, True]  # T, F, T

# Score the batch
batch_result = exam_system.score_batch(questions, agent_responses)

print(f"\nScores: {batch_result.scores}")

# Provide feedback
exam_system.provide_batch_feedback(batch_result)

print(f"\nUpdated knowledge for Task_A: {exam_system.agent_knowledge['Task_A'].known_relations}")
print(f"Performance: {exam_system.get_task_performance('Task_A')}")

# %%
