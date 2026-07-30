# Security Policy

## Status

llmcomposer is a pre-1.0 research exploration, designed to run as a local,
single-user tool. This policy exists so the reporting path is in place, and so
users know what the project does and does not defend against.

## Reporting a vulnerability

**Please do not open a public issue.**

Use **GitHub Security Advisories** → the *Security* tab → *Report a
vulnerability*. This creates a private thread visible only to maintainers.

If that is unavailable, open a public issue containing only "requesting a
private channel for a security report" and no details.

### What to expect

This is a small open-source research project with no paid support and no
service-level agreement. As a statement of intent, not a commitment:

| | Target |
|---|---|
| Acknowledgement | within a few days |
| Initial assessment | within a week |
| Fix for a confirmed critical issue | prioritized over all other work |
| Public disclosure | after a fix ships, coordinated with you |

Reporters are credited in release notes unless they prefer otherwise.

## Scope

**In scope:**

- Script execution in the studio via model output — the agent's replies and
  scores are untrusted text that the browser renders
- API credentials (`ANTHROPIC_API_KEY`, `LOGFIRE_TOKEN`) leaking into
  responses, exports, or the browser
- Path traversal or arbitrary file access through the HTTP API
- Supply-chain issues in the dependency set or release pipeline

**Out of scope:**

- Exposing the server beyond localhost — llmcomposer binds `127.0.0.1` and
  running it on a public interface is not a supported deployment
- Denial of service against the user's own machine (a chatty model spending
  the user's own tokens is a cost concern, not a vulnerability)
- Attacks requiring root or physical access
- Vulnerabilities in the upstream model providers or in abcjs itself,
  unless llmcomposer's use of them creates the exposure
- Missing hardening with no demonstrated impact

## Security posture

- **Local by default.** The server binds loopback; there is no
  authentication because there is no remote surface.
- **Model output is untrusted input.** Scores pass through a strict ABC
  validator before rendering, and replies are rendered as text, not HTML.
- **No secrets at rest.** Credentials are read from the environment per
  process; nothing is written to disk.
- **No telemetry** beyond opt-in Logfire tracing, which activates only when
  `LOGFIRE_TOKEN` is explicitly set.

## Supported versions

Pre-release: only the tip of `main` receives fixes.
