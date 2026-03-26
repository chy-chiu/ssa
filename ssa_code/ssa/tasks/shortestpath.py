# %% 
# import random
import string
from typing import Dict, List, Tuple, Optional, Set
from ssa.tasks.task import TaskBase, Question
import heapq
import random

class GraphPathTask(TaskBase):
    """
    A shortest path estimation task. Agent must estimate shortest path lengths
    in a weighted graph by learning edge weights over time.
    """
    
    def __init__(self, task_id: int = 1, n_nodes: int = 8, n_edges: int = 20):
        super().__init__(task_id)
        self.n_nodes = n_nodes
        self.n_edges = n_edges
        self.nodes: List[str] = []
        self.edges: Dict[Tuple[str, str], float] = {}  # (node1, node2) -> weight
        self.adjacency: Dict[str, List[Tuple[str, float]]] = {}  # node -> [(neighbor, weight), ...]
    
    def generate_ground_truth(self, seed: int = None):
        """Generate a random connected graph with specified nodes and edges"""
        if seed is not None:
            random.seed(seed)
        
        # Generate node names A, B, C, ...
        self.nodes = [chr(ord('A') + i) for i in range(self.n_nodes)]
        
        # Start with spanning tree to ensure connectivity (n-1 edges)
        edges_set = set()
        remaining_nodes = self.nodes[1:]
        connected_nodes = {self.nodes[0]}
        
        # Build spanning tree
        while remaining_nodes:
            # Pick random connected node and random remaining node
            from_node = random.choice(list(connected_nodes))
            to_node = random.choice(remaining_nodes)
            
            edge = tuple(sorted([from_node, to_node]))
            edges_set.add(edge)
            connected_nodes.add(to_node)
            remaining_nodes.remove(to_node)
        
        # Add remaining random edges until we have n_edges total
        all_possible_edges = {
            tuple(sorted([self.nodes[i], self.nodes[j]]))
            for i in range(self.n_nodes) 
            for j in range(i+1, self.n_nodes)
        }
        
        remaining_possible = all_possible_edges - edges_set
        additional_edges = random.sample(
            list(remaining_possible), 
            min(self.n_edges - len(edges_set), len(remaining_possible))
        )
        edges_set.update(additional_edges)
        
        # Assign random weights to all edges
        self.edges = {
            edge: round(random.uniform(1.0, 10.0), 1) 
            for edge in edges_set
        }
        
        # Build adjacency list for shortest path calculation
        self.adjacency = {node: [] for node in self.nodes}
        for (node1, node2), weight in self.edges.items():
            self.adjacency[node1].append((node2, weight))
            self.adjacency[node2].append((node1, weight))
    
    def _dijkstra(self, start: str, end: str) -> float:
        """Calculate shortest path length using Dijkstra's algorithm"""
        distances = {node: float('inf') for node in self.nodes}
        distances[start] = 0
        pq = [(0, start)]
        visited = set()
        
        while pq:
            current_dist, current_node = heapq.heappop(pq)
            
            if current_node in visited:
                continue
            visited.add(current_node)
            
            if current_node == end:
                return current_dist
            
            for neighbor, weight in self.adjacency[current_node]:
                distance = current_dist + weight
                if distance < distances[neighbor]:
                    distances[neighbor] = distance
                    heapq.heappush(pq, (distance, neighbor))
        
        return float('inf')  # No path found
    
    def generate_question(self) -> Question:
        """Generate a random shortest path length question"""
        if not self.edges:
            raise RuntimeError("Generate ground truth first")
        
        # Pick two different random nodes
        start, end = random.sample(self.nodes, 2)
        true_length = self._dijkstra(start, end)
        
        question_text = (
            f"What is the length of the shortest path from node {start} to node {end}? "
            f"Respond with only a number (e.g., '7.3')."
        )
        
        return Question(question_text=question_text, question_data=(start, end), correct_answer=true_length)
    
    def score_response(self, question: Question, agent_response: str) -> float:
        """Score based on relative error of the estimate"""
        try:
            agent_estimate = float(agent_response.strip())
            true_length = question.correct_answer
            
            if true_length == 0:
                return 1.0 if agent_estimate == 0 else 0.0
            
            relative_error = abs(agent_estimate - true_length) / true_length
            # Score: 1.0 for perfect, decreasing as error increases, 0.0 for >100% error
            return max(0.0, 1.0 - relative_error)
        
        except ValueError:
            return 0.0  # Invalid response format
    
    def extract_feedback_info(self, question: Question, agent_response: str) -> Optional[Tuple[str, str, float]]:
        """Return a random edge weight as feedback"""
        if not self.edges:
            return None
        
        # Pick a random edge to reveal
        edge, weight = random.choice(list(self.edges.items()))
        node1, node2 = edge
        return (node1, node2, weight)  # (node1, node2, weight)


class GraphPathAgent:
    def __init__(self, model, nodes: List[str], task_id: str = "Graph-01"):
        self.model = model
        self.nodes = nodes
        self.knowledge_base: Dict[Tuple[str, str], float] = {}  # Known edge weights
        self.task_id = task_id
        self.system_prompt = f"""You are solving shortest path problems in a weighted graph.

The graph has nodes: {', '.join(nodes)}
You need to estimate the shortest path length between two given nodes.

Use your knowledge of edge weights to make educated estimates.
For unknown edges, make reasonable guesses based on typical edge weights you've seen.
Respond with ONLY a number representing the estimated shortest path length."""
    
    def probe_task(self, question: Question) -> str:
        """Estimate shortest path length using known edge weights"""
        kb_text = self._format_knowledge_base()
        
        prompt = f"""Known edge weights:
{kb_text}

{question.question_text}"""
        
        response = self.model.invoke([
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt}
        ])
        
        return response.content.strip()
    
    def update_knowledge_base(self, feedback_info: Optional[Tuple[str, str, float]]):
        """Update with new edge weight"""
        if feedback_info is None:
            return
        
        node1, node2, weight = feedback_info
        # Store edge in canonical order (sorted)
        edge = tuple(sorted([node1, node2]))
        self.knowledge_base[edge] = weight
    
    def _format_knowledge_base(self) -> str:
        """Format known edge weights for the prompt"""
        if not self.knowledge_base:
            return "No edge weights known yet."
        
        edges = []
        for (node1, node2), weight in sorted(self.knowledge_base.items()):
            edges.append(f"  {node1}-{node2}: {weight}")
        
        return "\n".join(edges)


# Example usage
if __name__ == "__main__":
    task = GraphPathTask(task_id=1, n_nodes=8, n_edges=20)
    task.generate_ground_truth(seed=42)
    
    print("Ground Truth Graph:")
    print(f"Nodes: {', '.join(task.nodes)}")
    print("Edges (first 10):")
    for i, ((node1, node2), weight) in enumerate(task.edges.items()):
        if i < 10:
            print(f"  {node1}-{node2}: {weight}")
    
    # Test shortest path calculation
    test_path_length = task._dijkstra('A', 'H')
    print(f"\nShortest path A to H: {test_path_length}")
    
    # Generate question and simulate interaction
    q = task.generate_question()
    print(f"\nQuestion: {q.question_text}")
    print(f"True answer: {q.correct_answer}")
    
    # Simulate agent response
    agent_response = "15.2"
    score = task.score_response(q, agent_response)
    feedback = task.extract_feedback_info(q, agent_response)
    
    print(f"Agent estimate: {agent_response}")
    print(f"Score: {score:.3f}")
    print(f"Feedback: Edge {feedback[0]}-{feedback[1]} has weight {feedback[2]}")

# %%
from ssa.utils import init_azure_model

model = init_azure_model()
agent = GraphPathAgent(model=model)