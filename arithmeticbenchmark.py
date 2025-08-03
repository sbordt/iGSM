import random
import json
from typing import List, Tuple, Dict
from enum import Enum

class Operation(Enum):
    ADD = "+"
    SUB = "-"
    MUL = "*"

class ArithmeticBenchmarkGenerator:
    """
    Generates benchmark questions for integer arithmetic.
    Supports addition, subtraction, and multiplication with configurable
    number of operands, parentheses, and maximum number range.
    """
    
    def __init__(self, n_max: int = 100, seed: int = 42):
        """
        Initialize the benchmark generator.
        
        Args:
            n_max: Maximum number that can appear in problems (inclusive)
            seed: Random seed for reproducible question generation
        """
        self.n_max = n_max
        self.seed = seed
        self.rng = random.Random(seed)
    
    def _generate_number(self) -> int:
        """Generate a random number between 1 and n_max (inclusive)."""
        return self.rng.randint(1, self.n_max)
    
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
        
        # Calculate the result
        result = self._calculate_result(numbers, operations)
        
        return expression, result
    
    def _calculate_result(self, numbers: List[int], operations: List[Operation]) -> int:
        """
        Calculate the result of the arithmetic expression.
        Follows standard operator precedence (multiplication before addition/subtraction).
        
        Returns:
            The final result
        """
        # Create copies to work with
        nums = numbers.copy()
        ops = operations.copy()
        
        # First pass: handle multiplication
        i = 0
        while i < len(ops):
            if ops[i] == Operation.MUL:
                result = nums[i] * nums[i + 1]
                nums[i] = result
                nums.pop(i + 1)
                ops.pop(i)
            else:
                i += 1
        
        # Second pass: handle addition and subtraction left to right
        result = nums[0]
        for i, op in enumerate(ops):
            if op == Operation.ADD:
                result = result + nums[i + 1]
            elif op == Operation.SUB:
                result = result - nums[i + 1]
        
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
        return f"Question: What is {expression}? Answer: {answer}"
    
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
    
    def generate_multiple_choice_variants(self, n: int = 3, use_parentheses: bool = True, 
                                        num_choices: int = 4) -> Tuple[str, List[int], int]:
        """
        Generate a question with multiple choice answers.
        
        Args:
            n: Number of operands in the expression
            use_parentheses: Whether to include parentheses in expressions
            num_choices: Number of answer choices to generate
            
        Returns:
            Tuple of (expression, list_of_choices, correct_answer_index)
        """
        expression, correct_answer = self._build_expression(n, use_parentheses)
        
        # Generate wrong answers by adding/subtracting random values
        choices = [correct_answer]
        max_attempts = 100  # Prevent infinite loops
        attempts = 0
        
        while len(choices) < num_choices and attempts < max_attempts:
            attempts += 1
            # Generate a plausible wrong answer with a wider range
            offset_range = max(20, abs(correct_answer) // 5)  # Scale offset with answer size
            offset = self.rng.randint(-offset_range, offset_range)
            if offset == 0:
                offset = self.rng.choice([-1, 1])
            wrong_answer = correct_answer + offset
            
            # Accept the wrong answer if it's not already in choices
            if wrong_answer not in choices:
                choices.append(wrong_answer)
        
        # If we couldn't generate enough unique choices, fill with systematic offsets
        if len(choices) < num_choices:
            base_offset = max(1, abs(correct_answer) // 10)
            for i in range(len(choices), num_choices):
                offset = base_offset * (i - len(choices) + 1)
                if self.rng.random() < 0.5:
                    offset = -offset
                wrong_answer = correct_answer + offset
                if wrong_answer not in choices:
                    choices.append(wrong_answer)
        
        # Shuffle the choices
        self.rng.shuffle(choices)
        correct_index = choices.index(correct_answer)
        
        return expression, choices, correct_index
    
    def generate_question_pair(self, n: int = 3, use_parentheses: bool = True) -> Tuple[str, str]:
        """
        Generate a pair of questions: one with correct answer, one with wrong answer.
        
        Args:
            n: Number of operands in the expression
            use_parentheses: Whether to include parentheses in expressions
            
        Returns:
            Tuple of (correct_question, wrong_question)
        """
        expression, correct_answer = self._build_expression(n, use_parentheses)
        
        # Generate wrong answer (off by a small random amount)
        offset = self.rng.randint(1, 10)
        if self.rng.random() < 0.5:
            offset = -offset
        wrong_answer = correct_answer + offset
        
        correct_q = f"Question: What is {expression}? Answer: {correct_answer}"
        wrong_q = f"Question: What is {expression}? Answer: {wrong_answer}"
        
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
            f.write(f"# Arithmetic Benchmark (numbers 1-{self.n_max})\n")
            f.write(f"# Generated with seed: {self.seed}\n")
            f.write(f"# Number of operands: {n}\n")
            f.write(f"# Parentheses: {use_parentheses}\n")
            f.write(f"# Total questions: {count}\n\n")
            
            for i, question in enumerate(questions, 1):
                f.write(f"{i}. {question}\n")
    
    def generate_jsonl_data(self, count: int, n: int = 3, use_parentheses: bool = True,
                           num_choices: int = 4) -> List[Dict]:
        """
        Generate benchmark data in JSONL format with multiple choice answers.
        
        Args:
            count: Number of unique questions to generate
            n: Number of operands in each expression
            use_parentheses: Whether to include parentheses in expressions
            num_choices: Number of answer choices per question
            
        Returns:
            List of dictionaries, each representing one question-answer pair
        """
        jsonl_data = []
        
        for doc_id in range(count):
            expression, choices, correct_index = self.generate_multiple_choice_variants(
                n, use_parentheses, num_choices)
            question_text = f"What is {expression}?"
            
            # Generate entries for each answer choice
            for idx, choice in enumerate(choices):
                answer_text = str(choice)
                prompt = f"Question: {question_text}\nAnswer: {answer_text}"
                
                # Binary label: 1 if this is the correct answer, 0 otherwise
                label = 1 if idx == correct_index else 0
                
                jsonl_data.append({
                    "prompt": prompt,
                    "doc_id": doc_id,
                    "idx": idx,
                    "label": label,
                    "correct_answer": choices[correct_index],
                    "expression": expression
                })
        
        return jsonl_data
    
    def save_jsonl_file(self, filename: str, count: int, n: int = 3, 
                       use_parentheses: bool = True, num_choices: int = 4) -> None:
        """
        Generate benchmark data and save as JSONL file.
        
        Args:
            filename: Output filename (should end with .jsonl)
            count: Number of unique questions to generate
            n: Number of operands in each expression
            use_parentheses: Whether to include parentheses in expressions
            num_choices: Number of answer choices per question
        """
        data = self.generate_jsonl_data(count, n, use_parentheses, num_choices)
        
        with open(filename, 'w') as f:
            for item in data:
                f.write(json.dumps(item) + '\n')
        
        print(f"Generated {len(data)} question-answer pairs ({count} questions × {num_choices} answers) to '{filename}'")
        print(f"Number range: 1-{self.n_max}")
        
        # Print some statistics
        correct_answers = [item['correct_answer'] for item in data if item['label'] == 1]
        print(f"Answer range: {min(correct_answers)} to {max(correct_answers)}")
        print(f"Average answer: {sum(correct_answers) / len(correct_answers):.1f}")
    
    def reset_seed(self, new_seed: int) -> None:
        """Reset the random seed for reproducible generation."""
        self.seed = new_seed
        self.rng = random.Random(new_seed)

# Example usage
if __name__ == "__main__":
    # Create generator with seed for reproducibility
    generator = ArithmeticBenchmarkGenerator(n_max=50, seed=42)
    
    # Generate some example questions
    print(f"Example questions with numbers 1-{generator.n_max}:")
    for i in range(10):
        expression, answer = generator._build_expression(n=3)
        print(f"Expression: {expression} = {answer}")
    
    print("\nExample multiple choice question:")
    expr, choices, correct_idx = generator.generate_multiple_choice_variants(n=3, num_choices=4)
    print(f"Question: What is {expr}?")
    for i, choice in enumerate(choices):
        marker = "*" if i == correct_idx else " "
        print(f"{marker} {chr(65+i)}) {choice}")
    
    print("\nExample JSONL data (first few entries):")
    jsonl_data = generator.generate_jsonl_data(count=2, n=3, num_choices=4)
    for i, item in enumerate(jsonl_data[:8]):  # Show first 8 entries
        print(json.dumps(item))
    print("... (continuing with remaining answer options)")
    
    print(f"\nGenerated {len(jsonl_data)} total entries for 2 questions")
    
    # Generate benchmark files
    generator.save_jsonl_file("arithmetic_benchmark.jsonl", count=50, n=3, num_choices=4)
    print("Generated arithmetic benchmark with 50 questions")
    
    # Show statistics for different number ranges
    print("\nStatistics for different number ranges:")
    for n_max in [10, 50, 100, 500]:
        test_gen = ArithmeticBenchmarkGenerator(n_max=n_max, seed=42)
        test_data = test_gen.generate_jsonl_data(count=100, n=3, num_choices=4)
        correct_answers = [item['correct_answer'] for item in test_data if item['label'] == 1]
        print(f"n_max={n_max}: answers range from {min(correct_answers)} to {max(correct_answers)}, avg={sum(correct_answers)/len(correct_answers):.1f}")