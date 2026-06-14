# Design QA

Reference: `/Users/psyche/Desktop/스크린샷 2026-06-15 오전 1.29.25.png`

Prototype: `http://127.0.0.1:7860/`

Viewport/state: desktop empty-job state after UI redesign.

Findings:
- P0/P1/P2: none blocking. The redesigned demo uses the requested three-pane document workflow: left job list, center PDF canvas, right result panel with preview/HTML/JSON tabs.
- P3: the reference screenshot shows a completed document with dense parsed content, while the verified local state has no uploaded job yet. The completed state uses the same center PDF and right result panel regions once a job finishes.
- P3: top-left product labeling remains `vlm-parser demo` instead of `Agent 1`; this keeps the current product identity while borrowing the target layout.

Final result: passed.
