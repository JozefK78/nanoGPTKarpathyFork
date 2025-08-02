import os
import numpy as np
import tiktoken
from multiprocessing import Pool, Manager, cpu_count
import argparse
from tqdm import tqdm

def search_chunk(args):
    """Search for a token sequence in a chunk of the dataset."""
    chunk_start, chunk_size, dataset_path, search_tokens, result_queue = args
    try:
        # Memory-map the dataset file
        data = np.memmap(dataset_path, dtype=np.uint16, mode='r')
        
        # Get the specific chunk for this worker
        chunk = data[chunk_start : chunk_start + chunk_size]
        
        # Search for the sequence
        search_len = len(search_tokens)
        for i in range(len(chunk) - search_len + 1):
            if np.array_equal(chunk[i:i+search_len], search_tokens):
                # Found a match, put the global index in the queue and stop
                result_queue.put(chunk_start + i)
                return
    except Exception as e:
        # In case of any error in the worker process
        print(f"Error in worker for chunk {chunk_start}: {e}")

def inspect_locations(dataset_path, indices, context_window=50):
    """Inspect specific locations in the dataset."""
    if not os.path.exists(dataset_path):
        print(f"Error: Dataset file not found at {dataset_path}")
        return

    enc = tiktoken.get_encoding("gpt2")
    data = np.memmap(dataset_path, dtype=np.uint16, mode='r')

    for index in indices:
        if index >= len(data):
            print(f"Error: Index {index} is out of bounds for dataset with length {len(data)}")
            continue

        print(f"\n--- Inspecting index {index} with context window {context_window} ---")

        start_context = max(0, index - context_window)
        end_context = min(len(data), index + context_window)

        context_tokens = data[start_context:end_context]
        print("Context text:")
        print(enc.decode(context_tokens.tolist()))
        print("--- END INSPECTION ---")

def find_first_occurrence(dataset_path, search_string, context_window=50):
    """
    Find the first occurrence of a search string in the tokenized dataset
    using multiprocessing.
    """
    if not os.path.exists(dataset_path):
        print(f"Error: Dataset file not found at {dataset_path}")
        return

    # Initialize tokenizer
    enc = tiktoken.get_encoding("gpt2")
    search_tokens = np.array(enc.encode(search_string), dtype=np.uint16)
    
    # Get file size and determine chunking
    file_size_bytes = os.path.getsize(dataset_path)
    total_tokens = file_size_bytes // 2  # Each token is 2 bytes (uint16)
    
    num_processes = cpu_count()
    chunk_size = total_tokens // num_processes
    
    # Use a Manager queue to get results from worker processes
    with Manager() as manager:
        result_queue = manager.Queue()
        
        pool = Pool(processes=num_processes)
        first_occurrence_index = None
        try:
            # Prepare arguments for each worker
            overlap = len(search_tokens) - 1
            tasks = []
            for i in range(num_processes):
                start = i * chunk_size
                size = chunk_size + overlap if i < num_processes - 1 else total_tokens - start
                tasks.append((start, size, dataset_path, search_tokens, result_queue))

            # Run tasks in parallel and show progress
            print("Scanning dataset...")
            with tqdm(total=total_tokens, desc="Scanning") as pbar:
                result = pool.map_async(search_chunk, tasks)
                pool.close() # No more tasks will be submitted to the pool

                # Monitor the queue for the first result while updating the progress bar
                while not result.ready():
                    if not result_queue.empty():
                        first_occurrence_index = result_queue.get()
                        print("\nFirst match found. Terminating other workers...")
                        break # Exit the monitoring loop
                    
                    # Heuristic progress update
                    pbar.update(chunk_size // 10)
                    result.wait(timeout=0.1)
                
                # After loop, check one last time if a result came in
                if first_occurrence_index is None and result.ready():
                    if not result_queue.empty():
                        first_occurrence_index = result_queue.get()

        except KeyboardInterrupt:
            print("\nKeyboard interrupt received. Terminating workers...")
            first_occurrence_index = None # Discard any partial results
        
        finally:
            # This block ensures that the pool is always cleaned up
            print("Terminating worker processes...")
            pool.terminate()
            pool.join()
            print("Worker processes terminated.")

        if first_occurrence_index is not None:
            print(f"\nFound first occurrence of '{search_string}' at token index: {first_occurrence_index}")
            
            # Now, let's get the context around this occurrence
            data = np.memmap(dataset_path, dtype=np.uint16, mode='r')
            
            # Find the full extent of the repetitive pattern
            pattern_end_index = first_occurrence_index + len(search_tokens)
            while pattern_end_index < len(data) and data[pattern_end_index] == search_tokens[-1]:
                pattern_end_index += 1
            
            pattern_length = pattern_end_index - first_occurrence_index
            print(f"Length of the problematic sequence: {pattern_length} tokens")

            # Get context before and after
            start_context = max(0, first_occurrence_index - context_window)
            end_context = min(len(data), pattern_end_index + context_window)
            
            before_tokens = data[start_context:first_occurrence_index]
            after_tokens = data[pattern_end_index:end_context]
            
            print("\n--- Context Before ---")
            print(enc.decode(before_tokens.tolist()))
            
            print("\n--- Corrupted Sequence ---")
            print(enc.decode(data[first_occurrence_index:pattern_end_index].tolist()))

            print("\n--- Context After ---")
            print(enc.decode(after_tokens.tolist()))
        else:
            print(f"Sequence '{search_string}' not found in the dataset.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Scan a tokenized dataset for a specific string.")
    parser.add_argument("path", type=str, help="Path to the .bin dataset file.")
    parser.add_argument("--search", type=str, default="!" * 100, help="The string to search for.")
    parser.add_argument("--context", type=int, default=50, help="Number of tokens for context window.")
    parser.add_argument("--inspect", type=int, nargs='+', default=None, help="Inspect one or more specific token indices instead of searching.")
    
    args = parser.parse_args()

    if args.inspect is not None:
        inspect_locations(args.path, args.inspect, args.context)
    else:
        find_first_occurrence(args.path, args.search, args.context)