🧮 Scalar Autograd Engine (From Scratch)

This notebook implements a minimal automatic differentiation engine for scalar values, inspired by how frameworks like PyTorch Autograd work internally — but built completely from scratch for learning purposes.

The goal is to understand backpropagation, computation graphs, and gradient flow at the most fundamental level.

📌 What this project does

Implements a Value class representing a scalar with gradient tracking

Builds a computation graph dynamically using Python operator overloading

Supports basic operations:

Addition (+)

Multiplication (*)

Power (**)

Manually defines local backward functions for each operation

Accumulates gradients via reverse-mode automatic differentiation

Visualizes the computation graph using Graphviz

This is a learning-focused implementation, not a performance-oriented one.

🧠 Core Idea

Every scalar value is treated as a node in a graph:

Value nodes store:

numerical data

gradient

Operation nodes represent computations (+, *, etc.)

Edges encode dependencies between values

During backpropagation:

Gradients flow from output to inputs

Each node applies the chain rule

Gradients are accumulated at leaf nodes

✨ Features

✔ Custom scalar autograd engine

✔ Dynamic computation graph creation

✔ Operator overloading (__add__, __mul__, __pow__)

✔ Manual backward functions

✔ Gradient accumulation

✔ Clean Graphviz visualization of the graph

✔ Educational and easy to extend

📂 File Structure
Practice/
├── ScalarDerivative.ipynb   # Main notebook (implementation + examples)
└── README.md

🚀 Example Usage
a = Value(5, label="A")
b = Value(6, label="B")
c = a * b
d = c + a

d.backward()


This builds a computation graph internally and computes gradients for a and b.

📊 Graph Visualization

The computation graph can be rendered using Graphviz, showing:

Rectangular nodes → scalar values (data, grad)

Circular nodes → operations (+, *)

Directed edges → data flow

This makes gradient flow explicit and intuitive.

🧪 Why this project exists

This project is meant to help you:

Understand how autograd works internally

Demystify backpropagation

See how computation graphs are built and traversed

Bridge the gap between math and deep learning frameworks

If you understand this notebook, you understand the core of PyTorch autograd.

⚠️ Limitations (Intentional)

Scalars only (no tensors)

No broadcasting

No vectorization

No performance optimizations

These limitations are intentional to keep the logic transparent and educational.

🔮 Possible Extensions

Add more operations (tanh, relu, exp)

Implement topological sorting explicitly

Add tensor support

Visualize backward pass separately

Compare gradients with PyTorch

📚 Inspiration

PyTorch Autograd

micrograd by Andrej Karpathy

Computational graph theory

🏁 Final Note

This notebook is not about efficiency — it’s about clarity.

“If you can build autograd for scalars, you truly understand backpropagation.”