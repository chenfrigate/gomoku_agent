"""
utils.py -- common utility functions for the Gomoku project
"""
import pickle
from replay_buffer import ReplayBuffer

def inspect_pkl(path: str, num_samples: int = 5):
    """
    Load a pickle file and print its type, length (if iterable),
    and a sample of its contents. If the loaded object is a ReplayBuffer,
    unwrap and inspect its internal buffer list.

    Args:
        path (str): Path to the .pkl file.
        num_samples (int): Number of elements to sample when iterable.

    Returns:
        The loaded object (list or other), or None if load failed.
    """
    try:
        with open(path, 'rb') as f:
            data = pickle.load(f)
    except Exception as e:
        print(f"❌ Failed to load pickle file '{path}': {e}")
        return None

    # If it's a full ReplayBuffer instance, extract its buffer list
    if isinstance(data, ReplayBuffer):
        print(f"🔍 Unwrapped ReplayBuffer object, inspecting its buffer list.")
        buffer_list = data.buffer
    else:
        buffer_list = data

    # Print basic info
    print(f"🔍 Inspecting object from '{path}': {type(buffer_list)}")
    try:
        length = len(buffer_list)
        print(f"   Length: {length}")
    except Exception:
        length = None
        print("   Length: <not available>")

    # If iterable and non-empty, show a small sample
    if hasattr(buffer_list, '__iter__') and length and length > 0:
        try:
            # Convert to list for indexing if necessary
            seq = list(buffer_list)
            sample = seq[:num_samples]
            print(f"   Sample (first {len(sample)} items):")
            for idx, elem in enumerate(sample):
                print(f"     [{idx}] type={type(elem)}, repr={repr(elem)}")
        except Exception as e:
            print(f"   Could not sample elements: {e}")

    return buffer_list



def inspect_policy_structure(buffer_list, index: int = 0):
    """
    Inspect the structure of the policy element in the buffer.

    Args:
        buffer_list: list of (state, policy, value) tuples.
        index (int): which sample to inspect (default 0).
    """
    try:
        state, policy, value = buffer_list[index]
    except Exception as e:
        print(f"❌ Cannot unpack sample at index {index}: {e}")
        return
    print(f"🔍 Inspecting policy of sample #{index}")
    print(f"   policy type: {type(policy)}")
    # List of tuples -> likely sparse (idx, prob)
    if isinstance(policy, list):
        if len(policy) > 0 and isinstance(policy[0], tuple) and len(policy[0]) == 2:
            print("   Detected sparse representation: list of (index, probability) pairs")
            print(f"   Example pairs: {policy[:5]}")
        else:
            print(f"   Detected full-length vector of length {len(policy)}")
            print(f"   First 10 values: {policy[:10]}")
    # Numpy array or torch tensor
    elif hasattr(policy, 'shape'):
        try:
            shape = policy.shape
        except Exception:
            shape = None
        print(f"   Detected array-like (shape={shape}), repr first elements: {repr(policy)[:100]}")
    else:
        print(f"   Unknown policy format, repr: {repr(policy)[:100]}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python utils.py <path_to_pkl> [num_samples]")
        sys.exit(1)
    path = sys.argv[1]
    num = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    buf = inspect_pkl(path, num)
    # 同时默认检查第一条的 policy 结构
    if buf is not None:
        inspect_policy_structure(buf, 0)
