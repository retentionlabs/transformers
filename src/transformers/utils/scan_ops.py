"""
Enhanced Associative Scan for Neural Memory Operations
Optimized implementation with robust PyTree handling and memory efficiency
"""

from typing import Callable, Any, Optional

from torch._higher_order_ops import associative_scan  # for future use
import torch


def _slice_tensor(
    tensor: torch.Tensor, start: int, end: Optional[int] = None,
    stride: int = 1, axis: int = 0
) -> torch.Tensor:
    """Create optimized tensor slice along specified axis"""
    slices = [slice(None)] * tensor.ndim
    slices[axis] = slice(start, end, stride)
    return tensor[tuple(slices)]


def _merge_sequences(
    even_seq: torch.Tensor, odd_seq: torch.Tensor, axis: int = 0
) -> torch.Tensor:
    """Efficiently merge even and odd sequences with proper padding"""
    even_len = even_seq.shape[axis]
    odd_len = odd_seq.shape[axis]

    # Handle different sequence lengths
    if even_len == odd_len:
        # Same length: direct interleaving
        combined = torch.stack([even_seq, odd_seq], dim=axis + 1)
        merged = torch.flatten(combined, start_dim=axis, end_dim=axis + 1)
    elif even_len == odd_len + 1:
        # Even sequence is longer by 1
        if odd_len == 0:
            # Special case: odd sequence is empty
            return even_seq
        # Interleave available parts
        min_len = odd_len
        even_part = _slice_tensor(even_seq, 0, min_len, 1, axis)
        combined = torch.stack([even_part, odd_seq], dim=axis + 1)
        interleaved = torch.flatten(combined, start_dim=axis, end_dim=axis + 1)
        # Append remaining even element
        last_even = _slice_tensor(even_seq, min_len, None, 1, axis)
        merged = torch.cat([interleaved, last_even], dim=axis)
    elif odd_len == even_len + 1:
        # Odd sequence is longer by 1
        if even_len == 0:
            # Special case: even sequence is empty
            return odd_seq
        # Interleave available parts
        min_len = even_len
        odd_part = _slice_tensor(odd_seq, 0, min_len, 1, axis)
        combined = torch.stack([even_seq, odd_part], dim=axis + 1)
        interleaved = torch.flatten(combined, start_dim=axis, end_dim=axis + 1)
        # Append remaining odd element
        last_odd = _slice_tensor(odd_seq, min_len, None, 1, axis)
        merged = torch.cat([interleaved, last_odd], dim=axis)
    else:
        # Unexpected length difference
        raise ValueError(
            f"Cannot merge sequences with lengths {even_len} and {odd_len}. "
            f"Length difference should be at most 1."
        )

    return merged


def _validate_inputs(tensors: list, axis: int) -> int:
    """Validate tensor inputs and return sequence length"""
    if not tensors:
        return 0

    ref_tensor = tensors[0]
    seq_length = ref_tensor.shape[axis]

    for i, tensor in enumerate(tensors[1:], 1):
        if tensor.shape[axis] != seq_length:
            raise ValueError(
                f"Tensor dimension mismatch at axis {axis}: "
                f"tensor[0].shape={ref_tensor.shape}, "
                f"tensor[{i}].shape={tensor.shape}"
            )

    return seq_length


def associative_scan(
    combine_fn: Callable[[Any, Any], Any],
    sequence: Any,
    dim: int = 0,
    reverse: bool = False,
    combine_mode: str = "generic"  # no needed, kept for compatibility
) -> Any:  # TODO: Use PyTree type hints for better type safety
    """
    Perform parallel associative scan using divide-and-conquer strategy

    Enhanced implementation for neural memory (for TTT, TITANS+) with:
    - Robust PyTree handling via dm-tree
    - Optimized tensor operations
    - Memory-efficient processing
    - Enhanced error reporting

    Args:
        combine_fn: Associative binary operator (a, b) -> c
        sequence: Input data (tensors, dicts, lists, or nested structures)
        dim: Axis dimension for scanning (default: 0)
        reverse: Process sequence in reverse order
        combine_mode: Mode for combining tensors ("generic" only, keep this argument for compatibility with pytorch associative_scan proto API)

    Returns:
        Scanned sequence with same structure as input

    Raises:
        TypeError: If combine_fn is not callable
        ValueError: If tensor dimensions don't match
        ImportError: If dm-tree is not available
    """
    # Input validation
    if not callable(combine_fn):
        raise TypeError(
            f"combine_fn must be callable, got {type(combine_fn).__name__}"
        )

    try:
        import tree  # TODO: Add dm-tree as a dependency to the project after discussion with the Hugging Face team
    except ImportError:
        raise ImportError(
            "dm-tree library required. Install with: pip install dm-tree"
        )

    # Handle edge cases
    if sequence is None:
        return None

    # Flatten nested structure using dm-tree
    try:
        flat_tensors = tree.flatten(sequence)
        # Store original sequence as template for reconstruction
        sequence_template = sequence
    except Exception as e:
        raise ValueError(f"Failed to flatten input structure: {e}")

    if not flat_tensors:
        if isinstance(sequence, (list, tuple)):
            return type(sequence)([])
        elif isinstance(sequence, dict):
            return {}
        else:
            return sequence

    # Apply reverse transformation
    if reverse:
        flat_tensors = [torch.flip(t, [dim]) for t in flat_tensors]

    # Validate tensor compatibility
    seq_len = _validate_inputs(flat_tensors, dim)

    def _combine_flat_tensors(left_tensors: list, right_tensors: list) -> list:
        """Apply combine function to flattened tensor lists"""
        # Reconstruct original structure for operator
        left_structured = tree.unflatten_as(sequence_template, left_tensors)
        right_structured = tree.unflatten_as(sequence_template, right_tensors)

        # Apply user-defined operator
        combined_result = combine_fn(left_structured, right_structured)

        # Flatten result back to list
        return tree.flatten(combined_result)

    # Core recursive scanning algorithm
    def _recursive_scan(tensor_list: list) -> list:
        """
        Divide-and-conquer associative scan implementation

        Algorithm:
        1. Base case: sequences of length < 2
        2. Pair-wise combine adjacent elements
        3. Recursively scan reduced sequence
        4. Compute final results and merge
        """
        current_length = tensor_list[0].shape[dim]

        # Base case: trivial sequences
        if current_length < 2:
            return tensor_list

        # Step 1: Combine adjacent pairs (downward pass)
        left_elements = [_slice_tensor(t, 0, -1, 2, dim) for t in tensor_list]
        right_elements = [_slice_tensor(t, 1, None, 2, dim) for t in tensor_list]

        paired_results = _combine_flat_tensors(left_elements, right_elements)

        # Step 2: Recursive call on reduced problem
        scanned_pairs = _recursive_scan(paired_results)

        # Step 3: Compute even-indexed results (upward pass)
        if current_length % 2 == 0:
            # Even length: use prefix of scanned pairs
            prefix_pairs = [_slice_tensor(t, 0, -1, 1, dim) for t in scanned_pairs]
            suffix_elements = [_slice_tensor(t, 2, None, 2, dim) for t in tensor_list]
            even_results = _combine_flat_tensors(prefix_pairs, suffix_elements)
        else:
            # Odd length: use full-scanned pairs
            suffix_elements = [_slice_tensor(t, 2, None, 2, dim) for t in tensor_list]
            even_results = _combine_flat_tensors(scanned_pairs, suffix_elements)

        # Step 4: Prepare final even sequence with identity element
        identity_elements = [_slice_tensor(t, 0, 1, 1, dim) for t in tensor_list]

        complete_even = []
        for identity_elem, even_elem in zip(identity_elements, even_results):
            if even_elem.numel() > 0 and identity_elem.shape[dim] > 0:
                complete_even.append(torch.cat([identity_elem, even_elem], dim=dim))
            elif even_elem.numel() > 0:
                complete_even.append(even_elem)
            else:
                complete_even.append(identity_elem)

        # Step 5: Merge even and odd sequences
        final_results = []
        for even_seq, odd_seq in zip(complete_even, scanned_pairs):
            try:
                merged = _merge_sequences(even_seq, odd_seq, dim)
                final_results.append(merged)
            except ValueError as e:
                # Debug information for merge issues
                print(f"Merge error: {e}")
                print(f"Even shape: {even_seq.shape}, Odd shape: {odd_seq.shape}")
                print(f"Original length: {current_length}")
                raise

        return final_results

    # Execute main algorithm
    scanned_tensors = _recursive_scan(flat_tensors)

    # Undo reverse transformation
    if reverse:
        scanned_tensors = [torch.flip(t, [dim]) for t in scanned_tensors]

    # Reconstruct original nested structure
    return tree.unflatten_as(sequence_template, scanned_tensors)


class MemoryOperator:
    """
    Enhanced memory operator for neural networks
    Supports gradient-based learning with momentum and forgetting
    """

    def __init__(
        self, learning_rate: float = 0.01, momentum: float = 0.9,
        decay_rate: float = 0.001, surprise_threshold: float = 0.1
    ):
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.decay_rate = decay_rate
        self.surprise_threshold = surprise_threshold

    def __call__(self, prev_memory: Any, current_input: Any) -> Any:
        """
        Apply neural memory update with adaptive learning

        Implements simplified version of memory update mechanism:
        - Surprise-based attention weighting
        - Momentum for stable learning
        - Decay for forgetting irrelevant information
        """
        if isinstance(prev_memory, dict) and isinstance(current_input, dict):
            return self._update_structured_memory(prev_memory, current_input)
        else:
            return self._update_tensor_memory(prev_memory, current_input)

    def _update_structured_memory(self, prev_state: dict, new_input: dict) -> dict:
        """Update structured memory (PyTree format)"""
        updated_memory = {}

        for key in prev_state.keys():
            if key in new_input:
                # Compute surprise metric (simplified)
                prev_val = prev_state[key]
                curr_val = new_input[key]

                surprise = torch.norm(curr_val - prev_val, dim=-1, keepdim=True)
                surprise_weight = torch.sigmoid(surprise - self.surprise_threshold)

                # Apply forgetting through decay
                decayed_memory = prev_val * (1 - self.decay_rate)

                # Momentum-based update
                update_signal = self.learning_rate * curr_val * surprise_weight
                updated_memory[key] = decayed_memory + self.momentum * update_signal
            else:
                # Pure decay for missing inputs
                updated_memory[key] = prev_state[key] * (1 - self.decay_rate)

        return updated_memory

    def _update_tensor_memory(
        self, prev_tensor: torch.Tensor, new_tensor: torch.Tensor
    ) -> torch.Tensor:
        """Update simple tensor memory"""
        # Surprise-based weighting
        surprise = torch.norm(new_tensor - prev_tensor, dim=-1, keepdim=True)
        surprise_weight = torch.sigmoid(surprise - self.surprise_threshold)

        # Memory update with decay and momentum
        decayed_prev = prev_tensor * (1 - self.decay_rate)
        weighted_update = self.learning_rate * new_tensor * surprise_weight

        return decayed_prev + self.momentum * weighted_update


if __name__ == '__main__':
    """
    Test suite and benchmarks - only runs when script is executed directly
    """

    def test_parallelism():
        """Test actual parallelism by comparing with sequential baseline"""
        print("=== Parallelism Analysis ===\n")

        import time

        # Test data
        seq_len = 4096000 // 4
        data = torch.randn(1, seq_len, 256)

        def add_op(a, b):
            return a + b

        # Method 1: Our associative scan
        start_time = time.time()
        result1 = associative_scan(add_op, data, dim=1)
        our_time = time.time() - start_time

        # Method 2: Sequential cumsum (baseline)
        start_time = time.time()
        result2 = torch.cumsum(data, dim=1)
        baseline_time = time.time() - start_time

        # Method 3: Manual sequential scan
        start_time = time.time()
        manual_result = data.clone()
        for i in range(1, seq_len):
            manual_result[:, i:i+1, :] = manual_result[:, i-1:i, :] + data[:, i:i+1, :]
        manual_time = time.time() - start_time

        print(f"Associative Scan: {our_time:.4f}s")
        print(f"PyTorch cumsum:   {baseline_time:.4f}s")
        print(f"Manual sequential: {manual_time:.4f}s")
        print(f"Speedup vs manual: {manual_time/our_time:.2f}x")
        print(f"Overhead vs cumsum: {our_time/baseline_time:.2f}x")

        # Check GPU utilization if available
        if torch.cuda.is_available():
            print(f"\nGPU available: Testing GPU parallelism...")
            data_gpu = data.cuda()

            start_time = time.time()
            result_gpu = associative_scan(add_op, data_gpu, dim=1)
            gpu_time = time.time() - start_time

            print(f"GPU time: {gpu_time:.4f}s")
            print(f"CPU vs GPU: {our_time/gpu_time:.2f}x speedup")


    def analyze_algorithm_complexity():
        """Analyze the theoretical vs actual complexity"""
        print("\n=== Algorithm Complexity Analysis ===\n")

        import math
        import time

        lengths = [100, 500, 1000, 2000, 4000]
        times = []

        def add_op(a, b):
            return a + b

        for length in lengths:
            data = torch.randn(1, length, 64)

            # Measure time
            start_time = time.time()
            _ = associative_scan(add_op, data, dim=1)
            elapsed = time.time() - start_time
            times.append(elapsed)

            # Theoretical complexity
            theoretical_ops = length * math.log2(length)

            print(
                f"Length {length:4d}: {elapsed:.4f}s | "
                f"Theoretical ops: {theoretical_ops:.0f}"
            )

        # Check if scaling matches O(n log n)
        print("\nScaling analysis:")
        for i in range(1, len(lengths)):
            actual_ratio = times[i] / times[i-1]
            length_ratio = lengths[i] / lengths[i-1]
            theoretical_ratio = (lengths[i] * math.log2(lengths[i])) / (lengths[i-1] * math.log2(lengths[i-1]))

            print(
                f"{lengths[i-1]}->{lengths[i]}: "
                f"Actual {actual_ratio:.2f}x | "
                f"Expected {theoretical_ratio:.2f}x"
            )


    def benchmark_scan_performance():
        """Benchmark different sequence lengths and data types"""
        print("=== Titans Associative Scan Benchmark ===\n")

        import time

        test_configs = [
            (100, 64, "Small"),
            (1000, 128, "Medium"),
            (4000, 256, "Large"),
            (8000, 512, "XLarge")
        ]

        for seq_len, hidden_dim, size_label in test_configs:
            print(f"Testing {size_label} sequence: {seq_len} x {hidden_dim}")

            # Generate test data
            test_tensor = torch.randn(2, seq_len, hidden_dim)

            def simple_add(a, b):
                return a + b

            # Benchmark timing
            start_time = time.time()
            result = associative_scan(simple_add, test_tensor, dim=1)
            elapsed = time.time() - start_time

            # Verify correctness
            expected = torch.cumsum(test_tensor, dim=1)
            max_error = torch.max(torch.abs(result - expected)).item()

            print(f"  Time: {elapsed:.4f}s | Max Error: {max_error:.2e}")
            print(f"  Input: {test_tensor.shape} | Output: {result.shape}\n")


    def test_titans_neural_memory():
        """Test neural memory update mechanism"""
        print("=== Testing Neural Memory System ===\n")

        # Setup test parameters
        batch_size, seq_len, memory_dim = 2, 500, 128

        # Create memory operator
        memory_op = MemoryOperator(
            learning_rate=0.02,
            momentum=0.85,
            decay_rate=0.001,
            surprise_threshold=0.2
        )

        # Initialize memory state
        initial_state = {
            'content': torch.randn(batch_size, 1, memory_dim) * 0.1,
            'attention': torch.zeros(batch_size, 1, memory_dim),
            'metadata': torch.ones(batch_size, 1, memory_dim // 4)
        }

        # Generate input sequence
        input_sequence = {
            'content': torch.randn(batch_size, seq_len, memory_dim),
            'attention': torch.randn(batch_size, seq_len, memory_dim) * 0.5,
            'metadata': torch.randn(batch_size, seq_len, memory_dim // 4) * 0.2
        }

        # Combine initial state with sequence
        full_sequence = {}
        for key in initial_state.keys():
            full_sequence[key] = torch.cat([
                initial_state[key], input_sequence[key]
            ], dim=1)

        print(f"Processing sequence with structure:")
        for key, tensor in full_sequence.items():
            print(f"  {key}: {tensor.shape}")

        # Apply associative scan for memory updates
        print("\nRunning neural memory scan...")
        memory_states = associative_scan(
            combine_fn=memory_op,
            sequence=full_sequence,
            dim=1,
            reverse=False
        )

        # Extract final memory state
        final_memory = {}
        for key, tensor in memory_states.items():
            final_memory[key] = tensor[:, -1:, :]  # Last timestep

        print("Memory update completed successfully!")
        print("\nFinal memory statistics:")
        for key, tensor in final_memory.items():
            mean_val = torch.mean(tensor).item()
            std_val = torch.std(tensor).item()
            print(f"  {key}: mean={mean_val:.4f}, std={std_val:.4f}")


    def validate_implementation():
        """Validate implementation against known results"""
        print("=== Implementation Validation ===\n")

        # Test 1: Simple cumulative sum
        print("Test 1: Cumulative Sum Validation")
        x = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0]])

        def add_op(a, b):
            return a + b

        result = associative_scan(add_op, x, dim=1)
        expected = torch.cumsum(x, dim=1)

        print(f"Input:    {x.tolist()}")
        print(f"Result:   {result.tolist()}")
        print(f"Expected: {expected.tolist()}")
        print(f"Match: {torch.allclose(result, expected)}\n")

        # Test 2: Nested structure
        print("Test 2: Nested Structure Validation")
        nested_data = {
            'values': torch.tensor([[1.0, 2.0, 3.0]]),
            'weights': torch.tensor([[0.5, 0.3, 0.2]])
        }

        def weighted_add(a, b):
            return {
                'values': a['values'] + b['values'],
                'weights': a['weights'] * 0.9 + b['weights'] * 0.1
            }

        nested_result = associative_scan(weighted_add, nested_data, dim=1)
        print(f"Nested processing successful: {type(nested_result) == dict}")
        print(f"Output shapes: {[v.shape for v in nested_result.values()]}")


    # Run all tests when script is executed directly
    validate_implementation()
    print("\n" + "="*60 + "\n")
    test_parallelism()
    print("\n" + "="*60 + "\n")
    analyze_algorithm_complexity()
    print("\n" + "="*60 + "\n")
    benchmark_scan_performance()
    print("="*60 + "\n")
    test_titans_neural_memory()
