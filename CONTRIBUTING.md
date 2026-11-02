# Contributing

Issues and pull requests are welcome.

- `npm ci && npx tsc --noEmit && npx cdk synth -q` must pass.
- `python3 -m pytest -q tests` must pass (`pip install pytest boto3` first).
- New backends: see "Adding a backend" in THIRD_PARTY_MODELS.md. Include the model's license in the PR description.
- Keep tools fast (every tool returns in under a second; heavy work belongs in the worker).
- No em dashes in AWS resource names or descriptions.
- By contributing you agree your contribution is licensed under Apache-2.0.
