"""
Evaluator classes for measuring agent performance metrics.
This module contains various evaluator classes that can be used to measure
different aspects of agent performance.
"""

import difflib
import math
from typing import Dict, List, Any, Optional, Set, Tuple, Union


class TopicAdherenceCalculator:
    """
    Evaluates whether queries adhere to predefined topics.
    Uses keyword matching to determine if queries are related to specific topics.
    """
    
    def __init__(self):
        self.queries = []
        self.adhered_queries = 0
        self.non_adhered_queries = 0
        self.query_topics = {}
        
        # Topic keywords mapping
        self.topic_keywords = {
            "history": ["history", "historical", "ancient", "past", "era", "century", "civilization", 
                        "dynasty", "empire", "war", "revolution", "movement", "king", "queen", 
                        "president", "leader", "who is", "when did"],
            "biography": ["who is", "biography", "life", "born", "died", "famous", "leader", 
                          "scientist", "artist", "politician", "inventor", "philosopher", "writer",
                          "career", "contribution", "achievement", "work", "legacy", "impact", "influence"],
            "physics": ["physics", "quantum", "theory", "relativity", "energy", "force", "mass", 
                        "motion", "gravity", "particle", "wave", "mechanics", "thermodynamics",
                        "einstein", "newton", "electron", "atom", "molecule", "light", "optics"],
            "science": ["science", "scientific", "research", "experiment", "discovery", "theory", 
                        "biology", "chemistry", "physics", "astronomy", "geology", "medicine",
                        "technology", "innovation", "laboratory", "scientist", "study", "analysis"],
            "customer service": ["service", "customer", "support", "help", "assistance", "issue", 
                                 "problem", "complaint", "resolve", "satisfaction", "experience",
                                 "contact", "representative", "agent", "request", "ticket", "inquiry", 
                                 "feedback", "order", "purchase", "refund", "return"],
            "product feedback": ["product", "feedback", "review", "experience", "opinion", "rating", 
                                 "recommend", "suggestion", "improvement", "feature", "quality",
                                 "performance", "satisfaction", "dissatisfaction", "like", "dislike",
                                 "good", "bad", "excellent", "terrible", "amazing", "awful", "useful"]
        }
        
    def check_topic_adherence(self, query: str, reference_topics: List[str], response: Any) -> bool:
        """
        Check if a query adheres to any of the reference topics.
        
        Args:
            query: User query to evaluate
            reference_topics: List of reference topics to check against
            response: Agent's response (used for logging tool information)
            
        Returns:
            Boolean indicating whether the query adheres to any reference topic
        """
        self.queries.append(query)
        query_lower = query.lower()
        detected_topics = []
        
        # Check each reference topic for keyword matches
        for topic in reference_topics:
            if topic in self.topic_keywords:
                for keyword in self.topic_keywords[topic]:
                    if keyword in query_lower:
                        detected_topics.append(topic)
                        break
        
        # Log the tool that the agent returned for potential topic inference
        if hasattr(response, 'get') and response.get("tool_called"):
            print(f"[TOPIC INFERENCE] Agent returned tool: {response['tool_called']}")
        
        self.query_topics[query] = list(set(detected_topics))
        adheres = len(detected_topics) > 0
        
        if adheres:
            self.adhered_queries += 1
            print(f"[TOPIC ADHERENCE] Query adheres to topics: {detected_topics}")
        else:
            self.non_adhered_queries += 1
            print(f"[TOPIC ADHERENCE] Query does not adhere to any reference topics")
        
        return adheres    
    
    def calculate_precision(self) -> float:
        """
        Calculate the precision of topic adherence.
        
        Returns:
            Precision as a float between 0 and 1
        """
        total_queries = self.adhered_queries + self.non_adhered_queries
        if total_queries == 0:
            return 0.0
        precision = self.adhered_queries / total_queries
        return precision
    
    def print_topic_adherence_summary(self) -> None:
        """
        Print a summary of topic adherence results.
        """
        precision = self.calculate_precision()
        print("\n==== TOPIC ADHERENCE SUMMARY ====")
        print(f"Total Queries Processed: {len(self.queries)}")
        print(f"Queries Adhering to Topics: {self.adhered_queries}")
        print(f"Queries Not Adhering to Topics: {self.non_adhered_queries}")
        print(f"Topic Adherence Precision: {precision:.4f}")
        print("\nQuery-Topic Mapping:")
        for query, topics in self.query_topics.items():
            topic_str = ", ".join(topics) if topics else "None"
            print(f"  Query: '{query}'")
            print(f"  Detected Topics: {topic_str}")
            print(f"  Adheres to Topics: {len(topics) > 0}")
            print()
        print("====================================\n")


class ToolCallAccuracy:
    """
    Evaluates whether the agent uses the correct tools for specific queries.
    Compares actual tool calls against reference patterns.
    """

    def __init__(self):
        self.tool_calls = []
        self.correct_calls = 0
        self.total_calls = 0
        self.query_tool_mapping = {}

    def evaluate_tool_call(self, query: str, actual_tool: Union[str, Dict[str, str]], reference_tool_calls: Dict[str, str]) -> bool:
        self.total_calls += 1

        # Extract actual tool name
        actual_tool_name = actual_tool.get("agent") if isinstance(actual_tool, dict) else actual_tool

        expected_tool = None
        matching_pattern = None

        for pattern, tool in reference_tool_calls.items():
            if pattern.strip().lower() in query.strip().lower():
                expected_tool = tool
                matching_pattern = pattern
                break

        is_correct = actual_tool_name == expected_tool

        self.query_tool_mapping[query] = {
            "actual_tool": actual_tool_name or "None",
            "expected_tool": expected_tool,
            "matching_pattern": matching_pattern,
            "is_correct": is_correct
        }

        if expected_tool is None:
            print(f"[TOOL ACCURACY] No reference tool call found for query: {query}")
            return False

        if is_correct:
            self.correct_calls += 1
            print(f"[TOOL ACCURACY] Correct tool called: {actual_tool_name}")
        else:
            print(f"[TOOL ACCURACY] Incorrect tool called: {actual_tool_name}, expected: {expected_tool}")

        return is_correct


    def calculate_accuracy(self) -> float:
        """
        Calculate the accuracy of tool calls.

        Returns:
            Accuracy as a float between 0 and 1
        """
        if self.total_calls == 0:
            return 0.0
        accuracy = self.correct_calls / self.total_calls
        return accuracy

    def print_tool_call_accuracy_summary(self) -> None:
        """
        Print a summary of tool call accuracy results.
        """
        accuracy = self.calculate_accuracy()
        print("\n==== TOOL CALL ACCURACY SUMMARY ====")
        print(f"Total Tool Calls: {self.total_calls}")
        print(f"Correct Tool Calls: {self.correct_calls}")
        print(f"Tool Call Accuracy: {accuracy:.4f}")
        print("\nQuery-Tool Mapping:")
        for query, details in self.query_tool_mapping.items():
            print(f"  Query: '{query}'")
            print(f"  Actual Tool: {details['actual_tool']}")
            print(f"  Expected Tool: {details['expected_tool'] or 'Not specified'}")
            if details['matching_pattern']:
                print(f"  Matching Pattern: '{details['matching_pattern']}'")
            print(f"  Correct: {details['is_correct']}")
            print()
        print("====================================\n")


class AgentGoalAccuracy:
    """
    Basic agent goal accuracy evaluator.
    Evaluates whether the agent's response contains the expected reference content.
    """
    
    def __init__(self):
        self.total = 0
        self.correct = 0
        self.query_goal_mapping = {}
    
    def evaluate_agent_goal_accuracy(self, query: str, reference: str, agent_response: str) -> int:
        """
        Evaluate if the agent's response achieves the goal.
        
        Args:
            query: User query to evaluate
            reference: Reference outcome that should be in the response
            agent_response: Agent's actual response
            
        Returns:
            1 if goal achieved, 0 otherwise
        """
        self.total += 1
        achieved = 1 if reference.lower() in agent_response.lower() else 0
        self.correct += achieved
        self.query_goal_mapping[query] = {
            "reference": reference,
            "agent_response": agent_response,
            "goal_achieved": achieved
        }
        print(f"[AGENT GOAL ACCURACY] Query: {query}, Goal Achieved: {achieved}")
        return achieved
    
    def calculate_accuracy(self) -> float:
        """
        Calculate the accuracy of agent goals.
        
        Returns:
            Accuracy as a float between 0 and 1
        """
        if self.total == 0:
            return 0.0
        return self.correct / self.total
        
    def print_goal_accuracy_summary(self) -> None:
        """
        Print a summary of agent goal accuracy results.
        """
        accuracy = self.calculate_accuracy()
        print("\n==== AGENT GOAL ACCURACY SUMMARY ====")
        print(f"Total Queries Evaluated: {self.total}")
        print(f"Queries with Goal Achieved: {self.correct}")
        print(f"Agent Goal Accuracy: {accuracy:.4f}")
        print("\nQuery-Goal Mapping:")
        for query, details in self.query_goal_mapping.items():
            print(f"  Query: '{query}'")
            print(f"  Reference: {details['reference']}")
            print(f"  Agent Response: {details['agent_response']}")
            print(f"  Goal Achieved: {details['goal_achieved']}")
            print()
        print("====================================\n")


class EnhancedAgentGoalAccuracy(AgentGoalAccuracy):
    """
    Enhanced agent goal accuracy evaluator with advanced NLP metrics.
    Extends AgentGoalAccuracy with additional metrics like BLEU, F1, precision, recall, and perplexity.
    """
    
    def __init__(self):
        super().__init__()
        self.query_metrics = {}
    
    def evaluate_agent_goal_accuracy(self, query: str, reference: str, agent_response: str) -> Dict[str, float]:
        """
        Evaluate agent goal accuracy using multiple metrics.
        
        Args:
            query: User query to evaluate
            reference: Reference outcome that should be in the response
            agent_response: Agent's actual response
            
        Returns:
            Dictionary with multiple evaluation metrics
        """
        # Basic binary accuracy (from parent class)
        binary_result = super().evaluate_agent_goal_accuracy(query, reference, agent_response)
        
        # Calculate advanced metrics
        bleu_score = self._calculate_bleu(reference, agent_response)
        precision, recall, f1 = self._calculate_f1(reference, agent_response)
        perplexity = self._calculate_perplexity(reference, agent_response)
        
        # Store all metrics
        metrics = {
            "binary": binary_result,
            "bleu": bleu_score,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "perplexity": perplexity
        }
        
        self.query_metrics[query] = metrics
        print(f"[ENHANCED GOAL METRICS] Query: {query}, BLEU: {bleu_score:.4f}, F1: {f1:.4f}")
        
        return metrics
    
    def _calculate_bleu(self, reference: str, hypothesis: str) -> float:
        """
        Calculate a simplified BLEU score.
        
        Args:
            reference: Reference text
            hypothesis: Hypothesis text to evaluate
            
        Returns:
            BLEU score as a float between 0 and 1
        """
        if not reference or not hypothesis:
            return 0.0
        
        # Tokenize
        ref_tokens = reference.lower().split()
        hyp_tokens = hypothesis.lower().split()
        
        if not ref_tokens or not hyp_tokens:
            return 0.0
        
        # Count matching n-grams (simplified)
        matches = 0
        for ref_token in ref_tokens:
            if ref_token in hyp_tokens:
                matches += 1
                # Remove the token to prevent double-counting
                hyp_tokens.remove(ref_token)
        
        # Calculate precision
        precision = matches / len(reference.split()) if reference else 0
        
        # Apply brevity penalty
        bp = min(1.0, math.exp(1 - len(reference.split()) / max(1, len(hypothesis.split()))))
        
        return bp * precision
    
    def _calculate_f1(self, reference: str, hypothesis: str) -> Tuple[float, float, float]:
        """
        Calculate precision, recall, and F1 score.
        
        Args:
            reference: Reference text
            hypothesis: Hypothesis text to evaluate
            
        Returns:
            Tuple of (precision, recall, f1) as floats between 0 and 1
        """
        if not reference or not hypothesis:
            return 0.0, 0.0, 0.0
        
        # Tokenize and convert to sets
        ref_tokens = set(reference.lower().split())
        hyp_tokens = set(hypothesis.lower().split())
        
        if not ref_tokens or not hyp_tokens:
            return 0.0, 0.0, 0.0
        
        # Calculate true positives (tokens in both reference and hypothesis)
        true_positives = len(ref_tokens.intersection(hyp_tokens))
        
        # Calculate precision and recall
        precision = true_positives / len(hyp_tokens) if hyp_tokens else 0
        recall = true_positives / len(ref_tokens) if ref_tokens else 0
        
        # Calculate F1
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        return precision, recall, f1
    
    def _calculate_perplexity(self, reference: str, hypothesis: str) -> float:
        """
        Calculate a simplified perplexity score.
        Lower is better for perplexity, but we invert it so higher is better
        (consistent with other metrics).
        
        Args:
            reference: Reference text
            hypothesis: Hypothesis text to evaluate
            
        Returns:
            Inverted perplexity score as a float between 0 and 1
        """
        if not reference or not hypothesis:
            return 0.0
        
        # Calculate edit distance as a proxy for perplexity
        edit_distance = difflib.SequenceMatcher(None, reference.lower(), hypothesis.lower()).ratio()
        
        # Convert to an inverted perplexity-like score
        # (1.0 means perfect match, 0.0 means completely different)
        return edit_distance
    
    def calculate_average_metrics(self) -> Dict[str, float]:
        """
        Calculate average values for all metrics across all queries.
        
        Returns:
            Dictionary with average values for each metric
        """
        if not self.query_metrics:
            return {
                "binary_accuracy": 0.0,
                "bleu": 0.0,
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0,
                "perplexity": 0.0
            }
        
        # Initialize counters
        totals = {
            "binary": 0.0,
            "bleu": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "perplexity": 0.0
        }
        
        # Sum up all metrics
        for metrics in self.query_metrics.values():
            for metric, value in metrics.items():
                totals[metric] += value
        
        # Calculate averages
        count = len(self.query_metrics)
        averages = {
            "binary_accuracy": totals["binary"] / count,
            "bleu": totals["bleu"] / count,
            "precision": totals["precision"] / count,
            "recall": totals["recall"] / count,
            "f1": totals["f1"] / count,
            "perplexity": totals["perplexity"] / count
        }
        
        return averages
    
    def print_goal_accuracy_summary(self) -> None:
        """
        Print a summary of enhanced agent goal accuracy results.
        """
        accuracy = self.calculate_accuracy()
        avg_metrics = self.calculate_average_metrics()
        
        print("\n==== ENHANCED AGENT GOAL ACCURACY SUMMARY ====")
        print(f"Total Queries Evaluated: {self.total}")
        print(f"Queries with Goal Achieved: {self.correct}")
        print(f"Binary Accuracy: {accuracy:.4f}")
        print(f"Average BLEU Score: {avg_metrics['bleu']:.4f}")
        print(f"Average F1 Score: {avg_metrics['f1']:.4f}")
        print(f"Average Precision: {avg_metrics['precision']:.4f}")
        print(f"Average Recall: {avg_metrics['recall']:.4f}")
        print(f"Average Perplexity (inverted): {avg_metrics['perplexity']:.4f}")
        
        print("\nQuery-Goal Metrics:")
        for query, metrics in self.query_metrics.items():
            print(f"  Query: '{query}'")
            print(f"  Binary Result: {metrics['binary']}")
            print(f"  BLEU Score: {metrics['bleu']:.4f}")
            print(f"  F1 Score: {metrics['f1']:.4f}")
            print(f"  Precision: {metrics['precision']:.4f}")
            print(f"  Recall: {metrics['recall']:.4f}")
            print(f"  Perplexity (inverted): {metrics['perplexity']:.4f}")
            print()
        
        print("=================================================\n")