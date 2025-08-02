import tiktoken

class Debug:
    def __init__(self, master_process, debug_batches=False):
        self.master_process = master_process
        self.debug_batches = debug_batches
        if self.debug_batches:
            self.enc = tiktoken.get_encoding("gpt2")

    def inspect_batch(self, iter_num, X, trigger_iters):
        if self.debug_batches and self.master_process and iter_num in trigger_iters:
            print(f"--- DEBUG: iter_num {iter_num} ---")
            try:
                decoded_text = self.enc.decode(X[0].tolist())
                print("Decoded text:", decoded_text)
            except Exception as e:
                print(f"Could not decode tokens: {e}")
            print("--- END DEBUG ---")