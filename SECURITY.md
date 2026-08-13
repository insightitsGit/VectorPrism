# Security Policy

## Supported versions

| Version | Supported |
|---------|-----------|
| `0.1.x` (main / PyPI) | Yes |
| Older / untagged forks | Best-effort |

## Reporting a vulnerability

**Do not open a public GitHub issue for security bugs.**

Prefer GitHub’s private channel:

→ **[Report a vulnerability](https://github.com/insightitsGit/VectorPrism/security/advisories/new)**

If that form is unavailable, contact the org maintainers via the GitHub profile — still keep details private until a fix is released.

Please include:
- Description and impact
- Reproduction steps / PoC (non-destructive)
- Affected version / commit

We aim to acknowledge within **72 hours** and ship a fix or mitigation as soon as practical.

## Scope

In scope:
- Remote code execution, path traversal, or secret leakage in VectorPrism code
- Unsafe deserialization of checkpoints / JSONL loaders
- Privilege issues in ingest/search CLIs when pointed at shared DBs

Out of scope:
- Issues in third-party deps alone (report upstream; we will bump)
- Social engineering / phishing
- Denial of service against public demo infrastructure you do not own

## Safe harbor

Good-faith research that follows this policy and avoids privacy violations / service disruption is welcome. We will not pursue legal action for such reports.

## Checkpoints

Default `load_checkpoint` / truth-classifier load uses `torch.load(..., weights_only=True)` — tensors and state_dicts only. Use `--unsafe-pickle` only for trusted legacy local `.pt` files that still require full pickle.
