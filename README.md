# Applied Computer Science — Projects

52 from-scratch, production-grade systems that implement the concepts taught in
the companion book **[Applied Computer Science](https://github.com/jchu0/applied-cs-book)**.
Where a book chapter says **"Build it →"**, it points here.

- **Python** projects use FastAPI + pytest; **Rust** projects use trait-based
  design with `Result` error handling + Criterion benchmarks; the one **Go**
  system uses gRPC + protobuf.
- Projects are grouped in tiers — foundation & backend (01–10), distributed
  systems (11–20), ML/AI core (21–37), advanced ML (38–49), data infrastructure
  (50–52).
- [`CONCEPT_TO_PROJECT_MAP.md`](CONCEPT_TO_PROJECT_MAP.md) is the bidirectional
  bridge between the book's chapters and these projects.

Each project has its own `README` / `BLUEPRINT.md` and build instructions
(`pip install -e .` / `cargo build` / `make`). Status per project lives in
[`PROJECTS_STATUS.md`](PROJECTS_STATUS.md).
