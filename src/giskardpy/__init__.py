# USE_SYMENGINE = False
USE_SYMENGINE = True

# BACKEND = None
# BACKEND = 'cse'
# BACKEND = 'numpy'
# BACKEND = 'cython'

BACKEND = 'llvm'
# BACKEND = 'lambda'

def print_wrapper(msg):
    print(msg)
