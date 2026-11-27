# Contributing to Whydra

- Keep each PR focused on one problem that can be reviewed and tested on its own.
- Keep each commit small and readable. Use Conventional Commits, for example
  `fix(graph): preserve endpoint order`.
- One branch, one issue: create or select the issue before branching.
  Every PR must link its issue; both must have assignees. Use sub-issues
  for nested work; child PRs may target parent branches. Merge child PRs
  and close child issues before completing the parent.
- Name branches `<type>/<issue-number>-<kebab-case-description>`.
  Types: feat, fix, refactor, perf, docs, test, chore.
  Example: `fix/123-endpoint-order`.
- Use short sentences and plain words in commit messages, PRs, and reviews.
  Say what changed, why, and how you tested it.
- Open PRs as drafts until they are ready for review and merge.
  Mark a PR as ready only when implementation is complete, all required
  checks pass, and no known blockers remain.
- Preserve public imports, graph semantics, algorithm results, and optional
  dependencies. Agree on breaking changes with a maintainer. Mark the commit
  with `!` and document how to migrate.
- Add a regression test for each bug. New algorithms start as experimental.
  Cite the paper, state assumptions and limits, and show the expected graph
  in a reproducible example.
- Keep research runners, comparison benchmarks, datasets, and generated results
  outside the library. For speed claims, compare timings and outputs on the
  same input. Record third-party sources, licenses, changes, and notices.
  [Code provenance](docs/CODE_PROVENANCE.md)
- Install the hooks. Run `pre-commit run --all-files` and `python -m pytest`.
  Check that CI passes, including docs and package builds.
- Never commit or push temporary files, one-off scripts, or other disposable
  artifacts.
- You may add tests without prior approval. Get explicit maintainer approval
  before modifying, removing, or disabling existing tests, or making changes
  to checks, pre-commit hooks, or CI workflows. Never weaken or bypass safeguards.
