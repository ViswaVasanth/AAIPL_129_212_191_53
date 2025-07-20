#!/usr/bin/env python3
"""
Robust JSON Processor for handling malformed LLM outputs
Combines json-repair library with smart fallback strategies
"""

import json
import re
import time
from typing import Dict, List, Any, Optional

# Try to import json-repair, fallback to basic processing if not available
try:
    from json_repair import repair_json
    HAS_JSON_REPAIR = True
except ImportError:
    HAS_JSON_REPAIR = False

class RobustJSONProcessor:
    """
    Robust JSON processor that handles malformed LLM outputs
    Uses multiple strategies: direct parsing -> json-repair -> regex extraction -> LLM fallback
    """
    
    def __init__(self, agent_model=None):
        self.agent = agent_model
        
        # Pre-compile regex patterns for performance
        self.markdown_pattern = re.compile(r'```json\s*|\s*```')
        self.json_pattern = re.compile(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', re.DOTALL)
        self.prefix_pattern = re.compile(r'^[^{]*(?=\{)', re.MULTILINE)
        self.suffix_pattern = re.compile(r'\}[^}]*$')
        
        # Field extraction patterns
        self.field_patterns = {
            'topic': [
                re.compile(r'"topic":\s*"([^"]*)"', re.IGNORECASE),
                re.compile(r"'topic':\s*'([^']*)'", re.IGNORECASE),
                re.compile(r'topic:\s*"([^"]*)"', re.IGNORECASE),
                re.compile(r'topic:\s*([^,}\n]*)', re.IGNORECASE)
            ],
            'question': [
                re.compile(r'"question":\s*"([^"]*)"', re.IGNORECASE),
                re.compile(r"'question':\s*'([^']*)'", re.IGNORECASE),
                re.compile(r'question:\s*"([^"]*)"', re.IGNORECASE)
            ],
            'answer': [
                re.compile(r'"answer":\s*"([A-D])"', re.IGNORECASE),
                re.compile(r"'answer':\s*'([A-D])'", re.IGNORECASE),
                re.compile(r'answer:\s*"([A-D])"', re.IGNORECASE),
                re.compile(r'answer:\s*([A-D])', re.IGNORECASE)
            ],
            'explanation': [
                re.compile(r'"explanation":\s*"([^"]*)"', re.IGNORECASE),
                re.compile(r"'explanation':\s*'([^']*)'", re.IGNORECASE),
                re.compile(r'"reasoning":\s*"([^"]*)"', re.IGNORECASE)
            ]
        }
        
        # Choice extraction patterns
        self.choice_patterns = [
            re.compile(r'"choices":\s*\[(.*?)\]', re.DOTALL | re.IGNORECASE),
            re.compile(r"'choices':\s*\[(.*?)\]", re.DOTALL | re.IGNORECASE),
            re.compile(r'choices:\s*\[(.*?)\]', re.DOTALL | re.IGNORECASE)
        ]
        
        self.choice_item_patterns = [
            re.compile(r'"([A-D]\)[^"]*)"'),
            re.compile(r"'([A-D]\)[^']*)'"),
            re.compile(r'([A-D]\)[^,\]]*)')
        ]
    
    def extract_json_with_smart_fallback(self, raw_response: str, is_question: bool = True) -> Dict:
        """
        Main entry point - extract JSON with multiple fallback strategies
        """
        start_time = time.time()
        
        # STEP 1: Quick cleanup and direct parse (< 0.01s)
        cleaned = self.quick_cleanup(raw_response)
        
        try:
            result = json.loads(cleaned)
            if self.validate_structure(result, is_question):
                return self._add_metadata(result, "direct_parse", time.time() - start_time)
        except json.JSONDecodeError as e:
            direct_parse_error = str(e)
        except Exception as e:
            direct_parse_error = f"Unexpected error: {str(e)}"
        
        # STEP 2: JSON repair library (< 0.1s)
        if HAS_JSON_REPAIR:
            try:
                repaired = repair_json(cleaned)
                result = json.loads(repaired)
                if self.validate_structure(result, is_question):
                    return self._add_metadata(result, "json_repair", time.time() - start_time)
            except Exception as e:
                repair_library_error = str(e)
        else:
            repair_library_error = "json-repair library not available"
        
        # STEP 3: Extract JSON candidates from text (< 0.05s)
        json_candidates = self.extract_json_candidates(raw_response)
        for candidate in json_candidates:
            try:
                if HAS_JSON_REPAIR:
                    repaired = repair_json(candidate)
                    result = json.loads(repaired)
                else:
                    result = json.loads(candidate)
                    
                if self.validate_structure(result, is_question):
                    return self._add_metadata(result, "candidate_extraction", time.time() - start_time)
            except:
                continue
        
        # STEP 4: Field-by-field regex extraction (< 0.1s)
        extracted = self.extract_fields_individually(raw_response, is_question)
        if extracted and self.validate_structure(extracted, is_question):
            return self._add_metadata(extracted, "regex_extraction", time.time() - start_time)
        
        # STEP 5: Smart LLM fallback with error context (< 2s)
        if self.agent:
            try:
                llm_result = self.smart_llm_correction(
                    raw_response, 
                    is_question,
                    errors={
                        'direct_parse': direct_parse_error,
                        'repair_library': repair_library_error
                    }
                )
                return self._add_metadata(llm_result, "llm_fallback", time.time() - start_time)
            except Exception as e:
                pass
        
        # Final fallback - create safe default with raw response for LLM extraction
        safe_default = self.create_safe_default(is_question, raw_response)
        return self._add_metadata(safe_default, "safe_default", time.time() - start_time)
    
    def quick_cleanup(self, text: str) -> str:
        """Ultra-fast pre-processing"""
        # Remove markdown artifacts
        text = self.markdown_pattern.sub('', text)
        
        # Remove text before first {
        text = self.prefix_pattern.sub('', text)
        
        # Remove text after last }
        text = self.suffix_pattern.sub('}', text)
        
        return text.strip()
    
    def extract_json_candidates(self, text: str) -> List[str]:
        """Extract potential JSON objects from text"""
        candidates = self.json_pattern.findall(text)
        
        # Sort by length (longer is usually more complete)
        return sorted(candidates, key=len, reverse=True)
    
    def extract_fields_individually(self, text: str, is_question: bool) -> Optional[Dict]:
        """Extract each field separately using regex patterns"""
        result = {}
        
        # Extract topic
        if is_question:
            result['topic'] = self._try_patterns(text, self.field_patterns['topic'])
        
        # Extract question
        if is_question:
            result['question'] = self._try_patterns(text, self.field_patterns['question'])
        
        # Extract choices (complex)
        if is_question:
            choices = self._extract_choices_robust(text)
            if choices:
                result['choices'] = choices
        
        # Extract answer
        result['answer'] = self._try_patterns(text, self.field_patterns['answer'])
        
        # Extract explanation/reasoning
        explanation = self._try_patterns(text, self.field_patterns['explanation'])
        if explanation:
            if is_question:
                result['explanation'] = explanation
            else:
                result['reasoning'] = explanation
        
        # For answers, try simple letter extraction if no structured answer found
        if not is_question and not result.get('answer'):
            letter_match = re.search(r'\b([A-D])\b', text.upper())
            if letter_match:
                result['answer'] = letter_match.group(1)
                result['reasoning'] = result.get('reasoning', 'Extracted from text')
        
        return result if result else None
    
    def _try_patterns(self, text: str, patterns: List[re.Pattern]) -> Optional[str]:
        """Try multiple regex patterns and return first match"""
        for pattern in patterns:
            match = pattern.search(text)
            if match:
                return match.group(1).strip()
        return None
    
    def _extract_choices_robust(self, text: str) -> Optional[List[str]]:
        """Handle all possible choice format errors"""
        
        # Try to find choices array
        for pattern in self.choice_patterns:
            match = pattern.search(text)
            if match:
                choices_text = match.group(1)
                
                # Extract individual choice items
                for item_pattern in self.choice_item_patterns:
                    choice_items = item_pattern.findall(choices_text)
                    if len(choice_items) >= 4:
                        # Clean up and validate choices
                        cleaned_choices = []
                        for i, choice in enumerate(choice_items[:4]):
                            choice = choice.strip()
                            expected_prefix = f"{chr(65 + i)})"
                            if not choice.startswith(expected_prefix):
                                choice = f"{expected_prefix} {choice}"
                            cleaned_choices.append(choice)
                        return cleaned_choices
        
        # Fallback: look for individual choice lines
        choice_lines = re.findall(r'([A-D]\)[^\n,}]*)', text)
        if len(choice_lines) >= 4:
            return [choice.strip() for choice in choice_lines[:4]]
        
        return None
    
    def smart_llm_correction(self, raw_response: str, is_question: bool, errors: Dict) -> Dict:
        """Targeted LLM correction using specific error information"""
        
        if is_question:
            correction_prompt = self._build_question_correction_prompt(raw_response, errors)
        else:
            correction_prompt = self._build_answer_correction_prompt(raw_response, errors)
        
        # Use the agent's LLM with optimized parameters for speed
        try:
            corrected_response = self.agent.generate_response(
                correction_prompt,
                "You are a JSON correction expert. Output ONLY valid JSON.",
                max_new_tokens=512,
                temperature=0.0,
                do_sample=False
            )
            
            # Try to parse the corrected response
            cleaned_correction = self.quick_cleanup(corrected_response)
            result = json.loads(cleaned_correction)
            
            if self.validate_structure(result, is_question):
                return result
            else:
                return self.create_safe_default(is_question)
                
        except Exception as e:
            return self.create_safe_default(is_question)
    
    def _build_question_correction_prompt(self, raw_response: str, errors: Dict) -> str:
        """Build targeted correction prompt for questions"""
        
        prompt = f"""TASK: Fix the malformed JSON response to create a valid question JSON.

ORIGINAL RESPONSE:
{raw_response[:500]}...

ERRORS ENCOUNTERED:
- Direct parsing failed: {errors.get('direct_parse', 'Unknown')}
- JSON repair library failed: {errors.get('repair_library', 'Unknown')}

REQUIRED OUTPUT FORMAT (exactly this structure):
{{
    "topic": "One of: Logical Reasoning, Puzzles, Blood Relations and Family Tree",
    "question": "The actual question text",
    "choices": ["A) option1", "B) option2", "C) option3", "D) option4"],
    "answer": "A",
    "explanation": "Brief explanation under 100 words"
}}

CRITICAL REQUIREMENTS:
1. Output ONLY valid JSON - no extra text, no markdown
2. Answer must be exactly one letter: A, B, C, or D
3. Choices must be exactly 4 items starting with A), B), C), D)
4. Keep total tokens under 100 for topic+question+choices+answer

Extract and fix the content from the original response:"""
        
        return prompt
    
    def _build_answer_correction_prompt(self, raw_response: str, errors: Dict) -> str:
        """Build targeted correction prompt for answers"""
        
        prompt = f"""TASK: Fix the malformed JSON response to create a valid answer JSON.

ORIGINAL RESPONSE:
{raw_response[:300]}...

ERRORS ENCOUNTERED:
- Direct parsing failed: {errors.get('direct_parse', 'Unknown')}
- JSON repair library failed: {errors.get('repair_library', 'Unknown')}

REQUIRED OUTPUT FORMAT (exactly this structure):
{{
    "answer": "A",
    "reasoning": "Brief reasoning under 100 words"
}}

CRITICAL REQUIREMENTS:
1. Output ONLY valid JSON - no extra text, no markdown
2. Answer must be exactly one letter: A, B, C, or D
3. Extract the actual answer choice from the original response

Extract and fix the content from the original response:"""
        
        return prompt
    
    def validate_structure(self, data: Dict, is_question: bool) -> bool:
        """Fast structure validation"""
        if not isinstance(data, dict):
            return False
            
        if is_question:
            required = ['topic', 'question', 'choices', 'answer']
            
            # Check all required fields exist
            if not all(k in data for k in required):
                return False
            
            # Check choices format
            choices = data.get('choices', [])
            if not isinstance(choices, list) or len(choices) != 4:
                return False
            
            # Check each choice starts with A), B), C), D)
            expected_prefixes = ['A)', 'B)', 'C)', 'D)']
            for i, choice in enumerate(choices):
                if not isinstance(choice, str) or not choice.startswith(expected_prefixes[i]):
                    return False
            
            # Check answer is single letter
            answer = data.get('answer', '')
            if not isinstance(answer, str) or len(answer) != 1 or answer not in 'ABCD':
                return False
                
        else:  # Answer validation
            if 'answer' not in data:
                return False
            answer = data.get('answer', '')
            if not isinstance(answer, str) or len(answer) != 1 or answer not in 'ABCD':
                return False
        
        return True
    
    def create_safe_default(self, is_question: bool, raw_response: str = "") -> Dict:
        """Create safe default when all else fails - use LLM as ultimate fallback"""
        if self.agent and raw_response:
            # Ultimate fallback: Pass the raw response and error to LLM for manual extraction
            try:
                if is_question:
                    fallback_prompt = f"""The following response failed all JSON parsing attempts. Please extract the question data manually and return ONLY valid JSON:

FAILED RESPONSE:
{raw_response}

Extract and return in this exact format:
{{
    "topic": "extracted topic or best guess",
    "question": "extracted question text",
    "choices": ["A) option1", "B) option2", "C) option3", "D) option4"],
    "answer": "A",
    "explanation": "extracted explanation or brief reasoning"
}}

Return ONLY the JSON, no other text."""
                else:
                    fallback_prompt = f"""The following response failed all JSON parsing attempts. Please extract the answer data manually and return ONLY valid JSON:

FAILED RESPONSE:
{raw_response}

Extract and return in this exact format:
{{
    "answer": "A",
    "reasoning": "extracted reasoning or brief explanation"
}}

Return ONLY the JSON, no other text."""
                
                result = self.agent.generate_response(
                    fallback_prompt,
                    "You are an expert at extracting structured data from malformed text. Return only valid JSON.",
                    max_new_tokens=512,
                    temperature=0.0,
                    do_sample=False
                )
                
                # Try to parse the LLM's response
                cleaned = self.quick_cleanup(result)
                parsed = json.loads(cleaned)
                if self.validate_structure(parsed, is_question):
                    return parsed
            except Exception as e:
                # If even the LLM fallback fails, we have no choice but to return minimal structure
                pass
        
        # Absolute last resort - minimal structure to prevent crashes
        if is_question:
            return {
                "topic": "Error",
                "question": "Failed to parse response",
                "choices": ["A) Error", "B) Error", "C) Error", "D) Error"],
                "answer": "A",
                "explanation": "Parsing failed"
            }
        else:
            return {
                "answer": "A",
                "reasoning": "Parsing failed"
            }
    
    def _add_metadata(self, result: Dict, method: str, processing_time: float, error: str = None) -> Dict:
        """Add processing metadata to result"""
        result['_processing_info'] = {
            'method': method,
            'time': processing_time,
            'error': error
        }
        return result

# Convenience functions for easy integration
def extract_question_json(raw_response: str, agent_model=None) -> Dict:
    """Convenience function for question JSON extraction"""
    processor = RobustJSONProcessor(agent_model)
    return processor.extract_json_with_smart_fallback(raw_response, is_question=True)

def extract_answer_json(raw_response: str, agent_model=None) -> Dict:
    """Convenience function for answer JSON extraction"""
    processor = RobustJSONProcessor(agent_model)
    return processor.extract_json_with_smart_fallback(raw_response, is_question=False)

def extract_answer_letter(raw_response: str, choices: List[str] = None, agent_model=None) -> str:
    """Fast answer letter extraction with fallback strategies"""
    processor = RobustJSONProcessor(agent_model)
    
    # Try JSON extraction first
    try:
        result = processor.extract_json_with_smart_fallback(raw_response, is_question=False)
        if 'answer' in result:
            return result['answer'].upper()
    except:
        pass
    
    # Fast regex extraction
    letter_match = re.search(r'\b([A-D])\b', raw_response.upper())
    if letter_match:
        return letter_match.group(1)
    
    # Content matching if choices provided
    if choices:
        for i, choice in enumerate(choices):
            choice_content = choice[3:].strip().lower() if len(choice) > 3 else choice.lower()
            if choice_content in raw_response.lower():
                return chr(65 + i)
    
    # Ultimate fallback - use LLM to extract answer
    if agent_model:
        try:
            from utils.build_prompt import option_extractor_prompt
            return agent_model.generate_response(option_extractor_prompt(raw_response, choices or []))
        except:
            pass
    
    return 'A'  # Only as last resort
