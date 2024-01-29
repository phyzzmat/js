import scipy.stats as s
import numpy as np
from distributions import generate_distribution
from operators import generate_binary_op, generate_unary_op


def generate_tree(config, depth=0):
    difficulty = config['difficulty'] * 0.5 ** depth
    p_unary = config['p_unary']
    root = Node()
    split = s.uniform.rvs() < difficulty
    is_unary = s.uniform.rvs() < p_unary
    if split:
        root.is_terminal = False
        if is_unary:
            root.right = None
            root.is_unary = True
            root.left = generate_tree(config, depth + 1)
            root.inner = generate_unary_op(config, root.left.inner)
        else:
            root.left = generate_tree(config, depth + 1)
            root.right = generate_tree(config, depth + 1)
            root.inner = generate_binary_op(
                config, root.left.inner, root.right.inner)
        return root
    root.is_terminal = True
    root.inner = generate_distribution(config)
    return root


class Node:

    def __init__(self) -> None:
        self.is_unary = self.is_terminal = False
        self.left = self.right = None
        self.inner = None

    def __call__(self) -> np.ndarray:
        return self.inner()

    def __str__(self):
        if self.is_terminal:
            return self.inner.__str__()
        if self.is_unary:
            return f"[ {self.inner.__str__()} : {self.left.__str__()}]"
        return "(" + self.left.__str__() + " " + self.inner.__str__() + " " + self.right.__str__() + ")"
