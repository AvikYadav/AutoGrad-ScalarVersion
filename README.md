# 🔬 Scalar Autograd Engine (From Scratch)

A **minimal automatic differentiation engine for scalar values**, built from the ground up to understand how **backpropagation and computation graphs** actually work.

This project mirrors the core ideas behind **PyTorch Autograd**, but strips everything down to the bare essentials so the mechanics are completely transparent.

---

## Highlights

-  Custom **scalar autograd engine**
-  Dynamic **computation graph construction**
-  Operator overloading (`+`, `*`, `**`)
-  Manual backward functions (chain rule)
-  Gradient accumulation
-  **Graphviz visualization** of the computation graph
-  Designed purely for learning & clarity

---

##  Core Concept

Each scalar value is treated as a node in a **directed acyclic graph (DAG)**.

- **Value nodes** store:
  - numerical data
  - gradient
- **Operation nodes** represent computations (`+`, `*`, `**`)
- **Edges** encode dependencies between values

During backpropagation:
- gradients flow **from output → inputs**
- each operation applies the **chain rule**
- gradients accumulate at leaf nodes


## Example

```python
a = Value(5, label="A")
b = Value(6, label="B")

c = a * b
d = c + a

d.backward()
```

This builds the computation graph dynamically and computes gradients for a and b.

**Graph Visualization**

The computation graph can be rendered using Graphviz, where:

 Rectangular nodes → scalar values (data, grad)

 Circular nodes → operations (+, *)

 Directed edges → data flow

This makes gradient propagation explicit and visual, which is extremely useful for understanding backprop.

 **Why this exists**

This project is meant to help you:

truly understand how autograd works internally

demystify backpropagation

see computation graphs instead of just equations

connect math → code → deep learning frameworks

If you understand this notebook, you understand the core of modern deep learning frameworks.

 Limitations (Intentional)

Scalars only (no tensors)

No broadcasting

No vectorization

No performance optimizations

These constraints keep the implementation simple, readable, and educational.

⭐ If this helped you, consider starring the repo!
