# BareTensor Agent Guidelines

- Learning first: optimize for understanding and correctness, not speed of delivery.
- TinyGPT scope first: prioritize features needed to build TinyGPT end-to-end.
- PyTorch parity: keep API shape, semantics, and behavior as close to PyTorch as practical.
- Forward evolution over compatibility: backward compatibility is optional; prefer breaking changes when they produce a cleaner API, simpler internals, better scalability, and better developer ergonomics.
- No deprecated API surface: stay close to PyTorch, but do not implement deprecated methods.
- Professional quality: write production-grade code (clean, maintainable) comparable to mature libraries.
- Build for scale: implement with clean, extensible architecture to avoid avoidable technical debt.
- Modern codebase: use current language features, tooling, and engineering practices.
- After each code change, explain thoroughly what changed and why.
