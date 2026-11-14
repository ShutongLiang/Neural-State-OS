# Security policy

Neural State OS is a local research preview. The reference servers have no
authentication layer and should remain bound to loopback.

Please report vulnerabilities involving request parsing, arbitrary file access,
untrusted artifact loading, or remote-control boundaries through a private
GitHub security advisory when available. Do not include private model data,
credentials, or conversation content in a public issue.

External sensor and robot integrations are not part of the public release.
Any downstream actuator integration must add independent authentication,
rate-limiting, fail-safe behavior, and hardware-level safety controls.
