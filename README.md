# GNN, Graph Neural Network (GNNs) library and tools for GoMLX

[![GoDev](https://img.shields.io/badge/go.dev-reference-007d9c?logo=go&logoColor=white)](https://pkg.go.dev/github.com/gomlx/gnn?tab=doc)

> [!Note]
> 🚧 EXPERIMENTAL and IN DEVELOPMENT: It's no where near from a complete GNN library (like [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/), [TensorFlow GNN](https://github.com/tensorflow/gnn) and others. It's being populated and the need arises.
>
> **Contributions are most welcome**. 🚧

## What is in it?

* `geometry.RadiusEdges`: returns edges between the source and target points that are within a give radius.
  It works for arbitrary dimensions (2D, 3D, etc.).
* `geometry.NearestEdges`: returns the edges between each source point and its closest target point.
  It works for arbitrary dimensions (2D, 3D, etc.).
* `graph.UnionEdges`: returns the union from a list of edge sets.
* `graph.SortEdgesBySource`: sort edges by source id. 
