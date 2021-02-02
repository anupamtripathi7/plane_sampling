import gc
import logging
import numpy as np
import torch

def log_tensor_sizes(threshold=1048576, elem_size=4):
    total_size = 0
    gc.collect()
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                size = np.prod(obj.size())
                total_size += size

                if size > threshold:
                    referrers = gc.get_referrers(obj)
                    # names = reference_names(referrers)
                    # print(names)
                    logging.debug(f"referrers: {len(referrers)}, size: {humanbytes(size * elem_size)}, type({type(obj)}), shape {obj.size()}")
        except Exception as e:
            pass

    logging.debug(f"total size {humanbytes(total_size * elem_size)}")


def humanbytes(B):
    'Return the given bytes as a human friendly KB, MB, GB, or TB string'
    B = float(B)
    KB = float(1024)
    MB = float(KB ** 2)  # 1,048,576
    GB = float(KB ** 3)  # 1,073,741,824
    TB = float(KB ** 4)  # 1,099,511,627,776

    if B < KB:
        return '{0} {1}'.format(B, 'Bytes' if 0 == B > 1 else 'Byte')
    elif KB <= B < MB:
        return '{0:.2f} KB'.format(B / KB)
    elif MB <= B < GB:
        return '{0:.2f} MB'.format(B / MB)
    elif GB <= B < TB:
        return '{0:.2f} GB'.format(B / GB)
    elif TB <= B:
        return '{0:.2f} TB'.format(B / TB)


def reference_names(var):
    return [k for k, v in locals().iteritems() if v == var]
