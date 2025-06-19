import random
import json
from typing import List, Tuple, Dict
from enum import Enum

class Operation(Enum):
    ADD = "+"
    SUB = "-"
    MUL = "*"

class Mod23BenchmarkGenerator:
    """
    Generates benchmark questions for integer arithmetic modulo 23.
    Supports addition, subtraction, and multiplication with configurable
    number of operands and parentheses.
    """
    
    def __init__(self, seed: int = 42):
        """
        Initialize the benchmark generator.
        
        Args:
            seed: Random seed for reproducible question generation
        """
        self.seed = seed
        self.rng = random.Random(seed)
        self.modulus = 23
    
    def _generate_number(self) -> int:
        """Generate a random number between 0 and 22 (inclusive)."""
        return self.rng.randint(0, self.modulus - 1)
    
    def _generate_operation(self) -> Operation:
        """Generate a random operation."""
        return self.rng.choice(list(Operation))
    
    def _build_expression(self, n: int, use_parentheses: bool = True) -> Tuple[str, int]:
        """
        Build a mathematical expression with n numbers.
        
        Args:
            n: Number of operands in the expression
            use_parentheses: Whether to add parentheses for grouping
            
        Returns:
            Tuple of (expression_string, correct_answer)
        """
        if n < 2:
            raise ValueError("Need at least 2 numbers for an expression")
        
        # Generate numbers and operations
        numbers = [self._generate_number() for _ in range(n)]
        operations = [self._generate_operation() for _ in range(n - 1)]
        
        # Build expression string
        expr_parts = [str(numbers[0])]
        for i, op in enumerate(operations):
            expr_parts.extend([f" {op.value} ", str(numbers[i + 1])])
        
        expression = "".join(expr_parts)
        
        # Add parentheses for more complex expressions
        if use_parentheses and n > 2 and self.rng.random() < 0.5:
            # Choose a random position to add parentheses
            paren_start = self.rng.randint(0, n - 3) * 2  # Even indices are numbers
            paren_end = self.rng.randint(paren_start + 4, len(expr_parts) - 1)
            if paren_end % 2 == 0:  # Make sure we end on a number
                # Only add parentheses if they don't wrap the entire expression
                if not (paren_start == 0 and paren_end == len(expr_parts) - 1):
                    expr_parts.insert(paren_start, "(")
                    expr_parts.insert(paren_end + 2, ")")
                    expression = "".join(expr_parts)
        
        # Calculate the correct answer
        try:
            # Evaluate the expression modulo 23
            answer = eval(expression) % self.modulus
        except:
            # Fallback: calculate step by step
            answer = self._calculate_modular_result(numbers, operations)
        
        return expression, answer
    
    def _calculate_modular_result(self, numbers: List[int], operations: List[Operation]) -> int:
        """
        Calculate the result of operations modulo 23 step by step.
        Follows standard operator precedence.
        """
        # Convert to a format we can process with operator precedence
        # First handle multiplication, then addition/subtraction left to right
        
        # Create a copy of numbers and operations to work with
        nums = numbers.copy()
        ops = operations.copy()
        
        # First pass: handle multiplication
        i = 0
        while i < len(ops):
            if ops[i] == Operation.MUL:
                result = (nums[i] * nums[i + 1]) % self.modulus
                nums[i] = result
                nums.pop(i + 1)
                ops.pop(i)
            else:
                i += 1
        
        # Second pass: handle addition and subtraction left to right
        result = nums[0]
        for i, op in enumerate(ops):
            if op == Operation.ADD:
                result = (result + nums[i + 1]) % self.modulus
            elif op == Operation.SUB:
                result = (result - nums[i + 1]) % self.modulus
        
        return result
    
    def generate_question(self, n: int = 3, use_parentheses: bool = True) -> str:
        """
        Generate a single benchmark question.
        
        Args:
            n: Number of operands in the expression
            use_parentheses: Whether to include parentheses in expressions
            
        Returns:
            Formatted question string with answer
        """
        expression, answer = self._build_expression(n, use_parentheses)
        return f"Question: What is {expression} modulo 23? Answer: {answer}"
    
    def generate_questions(self, count: int, n: int = 3, use_parentheses: bool = True) -> List[str]:
        """
        Generate multiple benchmark questions.
        
        Args:
            count: Number of questions to generate
            n: Number of operands in each expression
            use_parentheses: Whether to include parentheses in expressions
            
        Returns:
            List of formatted question strings
        """
        questions = []
        for _ in range(count):
            questions.append(self.generate_question(n, use_parentheses))
        return questions
    
    def generate_all_answer_variants(self, n: int = 3, use_parentheses: bool = True) -> List[str]:
        """
        Generate all 23 possible variants of the same question (one for each possible answer 0-22).
        Only one will be correct, the other 22 will be wrong.
        
        Args:
            n: Number of operands in the expression
            use_parentheses: Whether to include parentheses in expressions
            
        Returns:
            List of 23 question strings, each with a different answer (0-22)
        """
        expression, correct_answer = self._build_expression(n, use_parentheses)
        
        # Generate all 23 variants
        variants = []
        for answer in range(self.modulus):
            question_str = f"Question: What is {expression} modulo 23? Answer: {answer}"
            variants.append(question_str)
        
        return variants
    
    def generate_question_with_correct_flag(self, n: int = 3, use_parentheses: bool = True) -> Tuple[List[str], int]:
        """
        Generate all 23 variants of a question and return which one is correct.
        
        Args:
            n: Number of operands in the expression
            use_parentheses: Whether to include parentheses in expressions
            
        Returns:
            Tuple of (list_of_23_questions, correct_answer_index)
            where correct_answer_index indicates which question has the right answer
        """
        expression, correct_answer = self._build_expression(n, use_parentheses)
        
        # Generate all 23 variants
        variants = []
        for answer in range(self.modulus):
            question_str = f"Question: What is {expression} modulo 23? Answer: {answer}"
            variants.append(question_str)
        
        return variants, correct_answer
    
    def generate_question_pair(self, n: int = 3, use_parentheses: bool = True) -> Tuple[str, str]:
        """
        Generate a pair of questions: one with correct answer, one with wrong answer (off by 1).
        
        Args:
            n: Number of operands in the expression
            use_parentheses: Whether to include parentheses in expressions
            
        Returns:
            Tuple of (correct_question, wrong_question)
        """
        expression, correct_answer = self._build_expression(n, use_parentheses)
        
        # Generate wrong answer (off by 1, randomly +1 or -1)
        if self.rng.random() < 0.5:
            wrong_answer = (correct_answer + 1) % self.modulus
        else:
            wrong_answer = (correct_answer - 1) % self.modulus
        
        correct_q = f"Question: What is {expression} modulo 23? Answer: {correct_answer}"
        wrong_q = f"Question: What is {expression} modulo 23? Answer: {wrong_answer}"
        
        return correct_q, wrong_q

    def generate_benchmark_file(self, filename: str, count: int, n: int = 3, 
                              use_parentheses: bool = True) -> None:
        """
        Generate benchmark questions and save to file.
        
        Args:
            filename: Output filename
            count: Number of questions to generate
            n: Number of operands in each expression
            use_parentheses: Whether to include parentheses in expressions
        """
        questions = self.generate_questions(count, n, use_parentheses)
        
        with open(filename, 'w') as f:
            f.write(f"# Modular Arithmetic Benchmark (mod 23)\n")
            f.write(f"# Generated with seed: {self.seed}\n")
            f.write(f"# Number of operands: {n}\n")
            f.write(f"# Parentheses: {use_parentheses}\n")
            f.write(f"# Total questions: {count}\n\n")
            
            for i, question in enumerate(questions, 1):
                f.write(f"{i}. {question}\n")
    
    def generate_jsonl_data(self, count: int, n: int = 3, use_parentheses: bool = True) -> List[Dict]:
        """
        Generate benchmark data in JSONL format with all 23 answer options for each question.
        
        Args:
            count: Number of unique questions to generate
            n: Number of operands in each expression
            use_parentheses: Whether to include parentheses in expressions
            
        Returns:
            List of dictionaries, each representing one question-answer pair
        """
        jsonl_data = []
        
        for doc_id in range(count):
            expression, correct_answer = self._build_expression(n, use_parentheses)
            question_text = f"What is {expression} modulo 23?"
            
            # Generate all 23 answer variants for this question
            for idx in range(self.modulus):
                answer_text = str(idx)
                prompt = f"Goal: {question_text}\nAnswer: {answer_text}"
                
                # Label is the correct answer index (not binary 0/1)
                label = correct_answer
                
                jsonl_data.append({
                    "prompt": prompt,
                    "doc_id": doc_id,
                    "idx": idx,
                    "label": label
                })
        
        return jsonl_data
    
    def save_jsonl_file(self, filename: str, count: int, n: int = 3, use_parentheses: bool = True) -> None:
        """
        Generate benchmark data and save as JSONL file.
        
        Args:
            filename: Output filename (should end with .jsonl)
            count: Number of unique questions to generate
            n: Number of operands in each expression
            use_parentheses: Whether to include parentheses in expressions
        """
        data = self.generate_jsonl_data(count, n, use_parentheses)
        
        with open(filename, 'w') as f:
            for item in data:
                f.write(json.dumps(item) + '\n')
        
        print(f"Generated {len(data)} question-answer pairs ({count} questions × 23 answers) to '{filename}'")
    
    def generate_jsonl_data_with_binary_labels(self, count: int, n: int = 3, use_parentheses: bool = True) -> List[Dict]:
        """
        Generate benchmark data in JSONL format with binary labels (1 for correct, 0 for incorrect).
        
        Args:
            count: Number of unique questions to generate
            n: Number of operands in each expression
            use_parentheses: Whether to include parentheses in expressions
            
        Returns:
            List of dictionaries, each representing one question-answer pair
        """
        jsonl_data = []
        
        for doc_id in range(count):
            expression, correct_answer = self._build_expression(n, use_parentheses)
            question_text = f"What is {expression} modulo 23?"
            
            # Generate all 23 answer variants for this question
            for idx in range(self.modulus):
                answer_text = str(idx)
                prompt = f"Goal: {question_text}\nAnswer: {answer_text}"
                
                # Binary label: 1 if this is the correct answer, 0 otherwise
                label = 1 if idx == correct_answer else 0
                
                jsonl_data.append({
                    "prompt": prompt,
                    "doc_id": doc_id,
                    "idx": idx,
                    "label": label
                })
        
        return jsonl_data
    
    def reset_seed(self, new_seed: int) -> None:
        """Reset the random seed for reproducible generation."""
        self.seed = new_seed
        self.rng = random.Random(new_seed)

# Example usage
if __name__ == "__main__":
    # Create generator with seed for reproducibility
    generator = Mod23BenchmarkGenerator(seed=42)
    
    # Generate some example questions
    print("Example questions with 3 operands:")
    for i in range(5):
        print(generator.generate_question(n=3))
    
    print("\nExample questions with 5 operands:")
    for i in range(3):
        print(generator.generate_question(n=5))
    
    print("\nExample JSONL data (first few entries):")
    jsonl_data = generator.generate_jsonl_data(count=2, n=3)
    for i, item in enumerate(jsonl_data[:8]):  # Show first 8 entries
        print(json.dumps(item))
    print("... (continuing with remaining answer options)")
    
    print(f"\nGenerated {len(jsonl_data)} total entries for 2 questions")
    
    # Generate benchmark files
    generator.generate_benchmark_file("mod23_benchmark.txt", count=100, n=4)
    print("\nGenerated 100 questions with 4 operands to 'mod23_benchmark.txt'")
    
    generator.save_jsonl_file("mod23_benchmark.jsonl", count=50, n=3)
    print("Generated JSONL benchmark with 50 questions (1,150 total entries)")
    
    # Example with binary labels
    binary_data = generator.generate_jsonl_data_with_binary_labels(count=1, n=3)
    print(f"\nExample with binary labels (question with correct answer {binary_data[0]['label']}):")
    for item in binary_data[:5]:
        print(json.dumps(item))


        