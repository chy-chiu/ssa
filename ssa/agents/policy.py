from ssa.agents.agent import AgentBase
from abc import ABC, abstractmethod
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import (
    PromptTemplate,
    SystemMessagePromptTemplate,
    StringPromptTemplate,
    ChatPromptTemplate,
)
import numpy as np
from ssa.utils import init_azure_model
from typing import List, Dict, Optional, Literal, Tuple, Any
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from pydantic import BaseModel, Field
import matplotlib.pyplot as plt
from loguru import logger
from ssa.tasks.task import TaskBase, TaskSubAgent, TaskRunner, ProxyAgent
from ssa.common import Job, JobHistory, AgentHistory, AgentPerformance, AgentActionResponse, MarketInfo, AgentLog, SubAgentLog


class PolicyAgent(AgentBase):
    """Static Policy Agent to test things with"""

    def __init__(self, agent_id: int, jobs: List[Job], model=None, verbose=True):
        super().__init__(agent_id=agent_id, jobs=jobs, model=model, verbose=verbose)
        self.preferences = None
        self.train_p = 0.1
        self.underbid_factor = 0.9
        self.greedy = False
        self.task_preferences = None
        self.job_preferences = None
        self.train_t = None

    def set_policy(self, task_preferences=None, job_preferences=None, train_p=None, underbid_factor=None, greedy=None, train_t=None):
        if task_preferences: 
            self.task_preferences = task_preferences
        if job_preferences:
            self.job_preferences = job_preferences
        if train_p: self.train_p = train_p
        if underbid_factor: self.underbid_factor = underbid_factor        
        if greedy: self.greedy = greedy
        if train_t: self.train_t = train_t

    def get_agent_action(self, market_info: MarketInfo):
        """Return pre-defined prefs, otherwise random preferences by default"""
        
        listings = market_info.listings

        if self.train_t:
            if market_info.round > self.train_t:
                action = "bid"
            else:
                action = "train"
        else:
            action = np.random.choice(["bid", "train"], p=(1 - self.train_p, self.train_p))

        if not self.task_preferences: 
            task_preferences = [self.task_ids[i] for i in np.random.permutation(self.n_tasks) if self.task_ids[i] in listings.keys()]
        else:
            task_preferences = [task_id for task_id in self.task_preferences if task_id in listings.keys()]

        if action == "train":
            targets = [(task_preferences[0], -1)]
            reasoning = f"Training skill {task_preferences[0]}"
        
        elif action == "bid":
            targets = []
            
            # Priority 1: Check for job preferences first
            if self.job_preferences:
                reasoning = "Bidding based on job preferences"
                for job_id in self.job_preferences:
                    # Find this job_id in listings and get its price
                    for _, jobs in listings.items():
                        if job_id in jobs:
                            price = jobs[job_id]
                            targets.append((job_id, price * self.underbid_factor))
            
            # Priority 2: Greedy - bid on highest priced jobs across all skills
            elif self.greedy:
                reasoning = "Bidding greedily on highest priced jobs"
                # Collect all jobs across all skills
                all_jobs = []
                for skill_id, jobs in listings.items():
                    for job_id, price in jobs.items():
                        all_jobs.append((job_id, price))
                
                # Sort by price descending (highest first)
                all_jobs.sort(key=lambda x: x[1], reverse=True)
                
                # Bid on all jobs in price order
                for job_id, price in all_jobs:
                    targets.append((job_id, price * self.underbid_factor))
            
            # Priority 3: Original task-based logic
            else:
                reasoning = "Bidding based on task preferences"
                for task_id in task_preferences:
                    if task_id in listings:
                        for job_id, price in listings[task_id].items():
                            targets.append((job_id, price * self.underbid_factor))

        response = AgentActionResponse(reasoning=reasoning, action=action, targets=targets)
        return response
