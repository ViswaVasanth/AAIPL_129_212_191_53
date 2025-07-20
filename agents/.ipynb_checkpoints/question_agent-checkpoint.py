#!/usr/bin/python3

from tqdm import tqdm
from pathlib import Path
from typing import List, Tuple, Dict, Any

from .question_model import QAgent

import random
import json

class QuestioningAgent(object):
    r"""Agent responsible for generating questions"""
    
    def __init__(self, **kwargs):
        self.agent = QAgent(**kwargs)

    def build_inc_samples(self, inc_samples: List[Dict[str, str]], topic: str)->str:
        r"""
        Build a string of example questions from the provided samples.
        """
        if not inc_samples:
            return ""
        fmt = (
            'EXAMPLE: {}\n'
            '{{\n'
            '  "topic": "{}",\n'
            '  "question": "{}",\n'
            '  "choices": ["{}", "{}", "{}", "{}"],\n'
            '  "answer": "{}",\n'
            '  "explanation": "{}"\n'
            '}}'            
        )

        sample_str = ""
        for sample in inc_samples:
            question = sample.get("question", "")
            choices = sample.get("choices", [""] * 4)
            answer = sample.get("answer", "")
            explanation = sample.get("explanation", "")
            sample_str += fmt.format(topic, topic.split('/')[-1], question, *choices, answer, explanation) + "\n\n"
        return sample_str.strip()


    def build_prompt(self, topic: str, wadvsys: bool = True, wicl: bool = True, inc_samples: List[Dict[str, str]]|None = None) -> Tuple[str, str]:
        """Generate an MCQ based question on given topic with specified difficulty"""
        
        if wadvsys:
            sys_prompt = """You are an elite competitive exam question designer who STRICTLY follows proven templates. Your primary skill is PERFECT REPLICATION of successful question patterns, not creative innovation.

CORE PRINCIPLE: The In context learning  examples provided are your SACRED BLUEPRINTS - deviation from them creates nonsensical questions.

YOUR MINDSET:
- You are a master craftsman following proven blueprints
- Creativity in question generation leads to logical failures  
- Your expertise is in precise pattern matching and replication
- The ICL examples represent perfected logical frameworks

MANDATORY APPROACH:
- Study each ICL example's logical structure completely
- Identify the exact constraint patterns used
- Replicate the SAME logical framework with different surface details
- Never invent new constraint types or logical relationships
- Verify your question solves identically to your chosen template

QUALITY STANDARD: Someone should be able to overlay your question with an In context learning example and see identical logical DNA - only names and numbers should differ.
DIVERSITY MANDATE: No two questions you generate should be exactly same. ALWAYS VERIFY THIS


Remember: Template adherence = Success. Creative deviation = Failure.
ALWAYS THINK STEP BY STEP"""
        else:
            sys_prompt = "You are an examiner who strictly follows provided examples as templates for creating questions."
        
        # Include examples if provided
        if wicl and inc_samples:
            inc_samples_ex = self.build_inc_samples(inc_samples, topic)
            examples_section = f"\n\nEXAMPLES:\n{inc_samples_ex}\n"
        else:
            examples_section = ""
        
        tmpl = (
            "ALWAYS THINK STEP BY STEP"
    
            'CRITICAL: Generate ONLY valid JSON. Invalid JSON = DISQUALIFIED.\n\n'
            
            '🎯 EXCELLENCE THROUGH TEMPLATE MASTERY:\n'
            'The In context learning examples below represent PERFECTED logical frameworks. Your expertise lies in studying these masterpieces and creating questions that honor their elegant structure while exploring new scenarios.\n\n'
            
            '📚 DEEP TEMPLATE ANALYSIS:\n'
            'Before generating, carefully examine the ICL examples for your topic:\n'
            '• Identify the core logical pattern and constraint structure\n'
            '• Understand the step-by-step reasoning flow\n'
            '• Note how constraints interact to create a unique solution\n'
            '• Observe the explanation style and clarity\n\n'
            
            'Create a challenging yet solvable MCQ on topic: {topic}\n\n'
            
            '🔍 LOGICAL FOUNDATION PRINCIPLES:\n'
            '1. RELEVANCE: Your question must directly test the core concepts of {topic}\n'
            '2. SOLVABILITY: Every constraint must be mathematically consistent and lead to exactly one correct answer\n'
            '3. CLARITY: The question should be unambiguous with clear, well-defined relationships\n'
            '4. ELEGANCE: Follow the proven logical patterns from ICL examples\n\n'
            
            '⚖️ QUALITY ASSURANCE PROCESS:\n'
            '• Design constraints that work together harmoniously\n'
            '• Verify each relationship is logically sound\n'
            '• Ensure the problem has a clear, step-by-step solution path\n'
            '• Test that only one answer choice is definitively correct\n'
            '• Confirm all distractors are plausible but clearly wrong\n\n'

             'MANDATORY LOGICAL VALIDATION:\n'
            '1. For SEATING ARRANGEMENTS:\n'
            '   - Circular: Use EVEN numbers for clear opposites\n'
            '   - Linear: Use LESS people maximum for solvability\n'
            '   - VERIFY all constraints are satisfiable before finalizing\n'
            '   - Test your arrangement step-by-step\n\n'
            
            '2. For FAMILY TREES:\n'
            '   - NO contradictory relationships (X cannot be both parent and sibling)\n'
            '   - Use LESS people maximum for solvability\n'
            '   - Maintain generation consistency\n'
            '   - Verify each relationship is logically possible\n'
            '   - Check for impossible loops or contradictions\n\n'
            
            '3. For TRUTH-TELLER PROBLEMS:\n'
            '   - Ensure statements can be consistently assigned\n'
            '- Use LESS people maximum for solvability\n'
            '   - Avoid paradoxes unless intentionally testing them\n'
            '   - Verify logical consistency of all statements\n\n'
            "DIVERSITY MANDATE: No two questions you generate should be exactly same. ALWAYS VERIFY THIS"

            
            
            '{examples}\n\n'
            
            '🎨 CRAFTSMANSHIP GUIDELINES:\n'
            '• Draw inspiration from the ICL template structure\n'
            '• Maintain the same level of logical rigor\n'
            '• Use fresh names  while preserving the logical DNA\n'
            '• Create an explanation that guides readers through the solution\n\n'
            
            '📋 MANDATORY REQUIREMENTS:\n'
            '• Question must be directly relevant to: {topic}\n'
            '• Must be completely solvable\n'
            '• Exactly 4 options labeled A), B), C), D)\n'
            '• Only ONE option can be factually correct\n'
            '• Answer must be definitively accurate (wrong answer = DISQUALIFIED)\n'
            '• Explanation under 100 words showing clear solution steps\n\n'
            
            'EXACT OUTPUT FORMAT:\n'
            '{{\n'
            '  "topic": "[TOPIC_NAME]",\n'
            '  "question": "[YOUR_QUESTION]",\n'
            '  "choices": ["A) [OPTION1]", "B) [OPTION2]", "C) [OPTION3]", "D) [OPTION4]"],\n'
            '  "answer": "[CORRECT_OPTION]",\n'
            '  "explanation": "[BRIEF_EXPLANATION]"\n'
            '}}\n\n'
            
            '✨ FINAL EXCELLENCE CHECK:\n'
            'Is your JSON valid? Is your question relevant, logical, and solvable? Does it honor the ICL template quality?'
        )
        
        prompt = tmpl.format(topic=topic, examples=examples_section)
        return prompt, sys_prompt


    def generate_question(self, topic: Tuple[str, str]|List[Tuple[str, str]], wadvsys: bool, wicl: bool, inc_samples: Dict[str, List[Dict[str, str]]]|None, **gen_kwargs) -> Tuple[List[str], int|None, float|None]:
        """Generate a question prompt for the LLM"""
        if isinstance(topic, list):
            prompt = []
            for t in topic:
                p, sp = self.build_prompt(f"{t[0]}/{t[1]}", wadvsys, wicl, inc_samples[t[1]])
                prompt.append(p)
        else:
            prompt, sp = self.build_prompt(f"{topic[0]}/{topic[1]}", wadvsys, wicl, inc_samples[topic[1]])
        
        resp, tl, gt = self.agent.generate_response(prompt, sp, **gen_kwargs)

        if (isinstance(resp, list) and all(isinstance(r, str) for r in resp)) or isinstance(resp, str):
            return resp, tl, gt
        else:
            return '', tl, gt if not isinstance(resp, list) else [''] * len(resp), tl, gt


    def generate_batches(self, num_questions: int, topics: Dict[str, List[str]], batch_size: int = 5, wadvsys: bool=True, wicl: bool = True, inc_samples: Dict[str, List[Dict[str, str]]]|None = None, **kwargs) -> Tuple[List[str], List[int | None], List[float | None]]:
        r"""
        Generate questions in batches
        ---

        Args:
            - num_questions (int): Total number of questions to generate.
            - topics (Dict[str, List[str]]): Dictionary of topics with subtopics.
            - batch_size (int): Number of questions to generate in each batch.
            - wadvsys (bool): Whether to use advance prompt.
            - wicl (bool): Whether to include in-context learning (ICL) samples.
            - inc_samples (Dict[str, List[Dict[str, str]]]|None): In-context learning samples for the topics.
            - **kwargs: Additional keyword arguments for question generation.

        Returns:
            - Tuple[List[str], List[int | None], List[float | None]]: Generated questions, token lengths, and generation times.
        """
        extended_topics = self.populate_topics(topics, num_questions)
        questions = []
        tls, gts = [], []
        # Calculate total batches including the partial last batch
        total_batches = (len(extended_topics) + batch_size - 1) // batch_size
        pbar = tqdm(total=total_batches, desc="STEPS: ")
        
        for i in range(0, len(extended_topics), batch_size):
            batch_topics = extended_topics[i:i + batch_size]
            batch_questions = self.generate_question(batch_topics, wadvsys, wicl, inc_samples, **kwargs)
            questions.extend(batch_questions[0]), tls.append(batch_questions[1]), gts.append(batch_questions[2])
            pbar.update(1)
        pbar.close()
        return questions, tls, gts

    def count_tokens_q(self, text: str) -> int:
        """Count the number of tokens using model.tokenizer"""
        if not hasattr(self.agent, 'tokenizer'):
            raise AttributeError("The agent does not have a tokenizer attribute.")
        return len(self.agent.tokenizer.encode(text, add_special_tokens=False))

    def filter_questions(self, questions: List[str|Dict[str, str|Any]]) -> List[Dict[str, str|Any]]:
        def basic_checks(q2: Dict[str, str])->bool:
            # check required keys
            required_keys = ['topic', 'question', 'choices', 'answer']
            if all((key in q2) for key in required_keys):
                # check choices format
                checks = all(isinstance(choice, str) and len(choice) > 2 and choice[0].upper() in 'ABCD' for choice in q2['choices'])
                if isinstance(q2['choices'], list) and len(q2['choices']) == 4 and checks:
                    # check answer format
                    # Check token length
                    check_len = sum(self.count_tokens_q(q2[k]) for k in ['question', 'answer'])
                    check_len += sum(self.count_tokens_q(choice) for choice in q2['choices']) - 15
                    if check_len < 130:
                        if check_len + self.count_tokens_q(q2.get('explanation', 'None')) <= 1024:
                            # Extra Checks: (PLUS checks) len(q2['answer']) == 1 and q2['answer'].upper() in 'ABCD':
                            if isinstance(q2['answer'], str):
                                return True
            return False
        correct_format_question = []
        for i, q in enumerate(questions):
            if isinstance(q, dict):
                if basic_checks(q):
                    correct_format_question.append(q)
            elif isinstance(q, str):
                try:
                    q1 = json.loads(q)
                    if basic_checks(q1):
                        correct_format_question.append(q1)
                except json.JSONDecodeError:
                    # If JSON decoding fails, skip this answer
                    print(f"Skipping invalid JSON at index {i}: {q}")
                    continue
            else:
                continue
        if len(correct_format_question) >= 0.5 * len(questions):
            return correct_format_question
        return list()
    
    def save_questions(self, questions: Any, file_path: str|Path) -> None:
        """Save generated questions to a JSON file"""
        # Ensure dir exist
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        # Save to JSON file
        with open(file_path, 'w') as f:
            json.dump(questions, f, indent=4)
    
    def populate_topics(self, topics: Dict[str, List[str]], num_questions: int) -> List[str]:
        """Populate topics sorted by topic name to group same topics in batches"""
        if not isinstance(topics, dict):
            raise ValueError("Topics must be a dictionary with topic names as keys and lists of subtopics as values.")
        
        all_subtopics = [(t, st) for t, sublist in topics.items() for st in sublist]
        if not all_subtopics:
            raise ValueError("No subtopics found in the provided topics dictionary.")
        
        selected_topics = random.choices(all_subtopics, k=num_questions)
        
        # Sort by topic name to group same topics together in batches
        #selected_topics.sort(key=lambda x: x[0])
        
        return selected_topics

    @staticmethod
    def load_icl_samples(file_path: str|Path) -> Dict[str, List[Dict[str, str]]]:
        """Load in-context learning samples from a JSON file"""
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File {file_path} does not exist.")
        with open(file_path, 'r') as f:
            samples = json.load(f)
        if not isinstance(samples, dict):
            raise ValueError("Samples must be inside dictionary.")
        return samples

# Example usage
if __name__ == "__main__":
    import argparse
    import yaml

    # ++++++++++++++++++++++++++
    # Run: python -m agents.question_agent --num_questions 20 --output_file outputs/questions.json --batch_size 5 --verbose
    # ++++++++++++++++++++++++++

    argparser = argparse.ArgumentParser(description="Generate questions using the QuestioningAgent.")
    argparser.add_argument("--num_questions", type=int, default=200, help="Total number of questions to generate.")
    argparser.add_argument("--output_file", type=str, default="outputs/questions.json", help="Output file name to save the generated questions.")
    argparser.add_argument("--batch_size", type=int, default=5, help="Batch size for generating questions.")
    argparser.add_argument("--verbose", action="store_true", help="Enable verbose output for debugging.")
    args = argparser.parse_args()

    inc_samples = QuestioningAgent.load_icl_samples("assets/topics_example.json")

    # Load topics.json file.
    with open("assets/topics.json") as f: topics = json.load(f)
    
    agent = QuestioningAgent()
    # gen_kwargs = {"tgps_show": True, "max_new_tokens": 1024, "temperature": 0.1, "top_p": 0.9, "do_sample": True}
    gen_kwargs = {"tgps_show": True}
    with open("qgen.yaml", "r") as f: gen_kwargs.update(yaml.safe_load(f))

    question, tls, gts = agent.generate_batches(
        num_questions=args.num_questions,
        topics=topics, 
        batch_size=args.batch_size,
        wadvsys=True,
        wicl=True,
        inc_samples=inc_samples,
        **gen_kwargs
    )
    print(f"Generated {len(question)} questions!")
    if args.verbose:
        for q in question:
            print(q, flush=True)
        print("\n" + "="*50 + "\n\n")
        if gen_kwargs.get("tgps_show", False):
            print("Time taken per batch generation:", gts)
            print("Tokens generated per batch:", tls)
            print(f"Total Time Taken: {sum(gts):.3f} seconds; Total Tokens: {sum(tls)}; TGPS: {sum(tls)/sum(gts):.3f} seconds\n\n")
        print("\n" + "+"*50 + "\n")

    # check if question is JSON format
    ques = []
    for q in question:
        try:
            json.loads(q)
        except json.JSONDecodeError as e:
            print(f"Invalid JSON format in question: {q}\nError: {e}")
            # use agent itself to extract JSON: Self-Reflection
            # the dictionary is not as expected.
            # IMPROVED: Use robust JSON processor with multiple fallback strategies
            from utils.robust_json_processor import extract_question_json
            q = extract_question_json(q, agent.agent)
        ques.append(q)
    # Save the questions for later analysis
    agent.save_questions(ques, args.output_file)
    filtered_file_name = args.output_file.replace("questions.json", "filtered_questions.json")
    agent.save_questions(agent.filter_questions(ques), filtered_file_name)
    print(f"Saved to {args.output_file}!")

    # ========================================================================================
