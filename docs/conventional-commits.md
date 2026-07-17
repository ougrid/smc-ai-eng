# Conventional commits — practice for this repo

Sources: [conventionalcommits.org v1.0.0](https://www.conventionalcommits.org/en/v1.0.0/#specification),
[qoomon's cheatsheet gist](https://gist.github.com/qoomon/5dfcdf8eec66a051ecd85625518cfd13),
[qoomon/git-conventional-commits](https://github.com/qoomon/git-conventional-commits).

## Format

```
<type>[optional scope][!]: <description>

[optional body]

[optional footer(s)]
```

- **type**: required, lowercase.
- **scope**: optional, a noun in parentheses naming the affected area — in this repo, one of
  the module names used in `E11. Commit plan` in `docs/technical-execution-plan.md`:
  `db`, `scripts`, `api`, `auth`, `web`, `agent`, `docs`. Don't put issue IDs in the scope.
- **`!`**: append right before the colon for a breaking change (e.g. `feat(api)!: ...`).
- **description**: imperative present tense ("add", not "added"/"adds"), no capitalized first
  letter, no trailing period.
- **body**: optional, separated from the description by one blank line. Explains *why*, and
  contrasts with prior behavior — not a restatement of the diff.
- **footer**: optional, separated by one blank line. `Token: value` or `Token #value` (token
  uses `-` instead of spaces, e.g. `Refs #12`, `Acked-by: ...`). `BREAKING CHANGE: <description>`
  (or `BREAKING-CHANGE`, treated as a synonym) is a footer and is the alternative to the `!`
  marker — use one or the other, not neither, whenever a commit breaks a public contract.

## Types used in this repo

| Type | Use for |
|---|---|
| `feat` | new capability (SemVer MINOR) |
| `fix` | bug fix (SemVer PATCH) |
| `refactor` | behavior-preserving rewrite |
| `perf` | performance-motivated refactor |
| `test` | adding/correcting tests only |
| `docs` | documentation only (README, `docs/*`, this file) |
| `build` | dependencies, build tooling, Dockerfiles |
| `chore` | scaffolding, `.gitignore`, config with no behavior change |
| `ci` | CI pipeline changes (if/when CI is added) |

Any breaking change (`feat`/`fix`/`refactor` that changes a contract another layer depends
on — an API response shape, a DB column, an env var name) → SemVer MAJOR → mark with `!` and/or
a `BREAKING CHANGE:` footer.

## Examples (matching this repo's modules)

```
feat(auth): add register/login/me endpoints

feat(agent): add route node with fail-closed intent gate

fix(agent): tune score floor to cut false-negative refusals

test(auth): cover token expiry and tampered-signature rejection

docs: add conventional-commits practice doc

chore: add docker compose data stack

feat(api)!: change /api/chat to stream UI message parts

BREAKING CHANGE: /api/chat no longer returns a single JSON body;
clients must consume the SSE stream (UI message stream v1).
```

## Commit often

Prefer small, single-purpose commits over large batched ones — one logical change per commit
(one module, one concern), following the module-by-module breakdown already sequenced in
`docs/technical-execution-plan.md` § E11 (Commit plan). Commit as soon as a unit lands and
passes its own tests, rather than accumulating unrelated changes across modules into one commit.
This keeps history bisectable and makes each commit's diff match its message.

Still governed by the CLAUDE.md-level git safety rules: only commit when the user asks, never
force-push or rewrite published history, and never bypass hooks.
