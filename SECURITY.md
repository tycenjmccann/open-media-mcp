# Security

## Reporting a vulnerability
Please report security issues privately through GitHub's "Report a vulnerability" (Security tab), not in a
public issue. We aim to acknowledge within 3 business days.

## Security model (what the stack does for you)
- **No public endpoint.** The only entry point is an Amazon Bedrock AgentCore Gateway with `AWS_IAM` (SigV4) or
  `CUSTOM_JWT` (your IdP) inbound auth. The stack refuses an unauthenticated gateway.
- **No inbound network path to the GPU.** Workers pull jobs from SQS; their security group has no ingress rules.
  Admin access is AWS Systems Manager Session Manager only. IMDSv2 is required.
- **Verified caller identity.** A Gateway REQUEST interceptor stamps the authenticated caller onto every tool call
  and overwrites anything a client sends in its place. Jobs, uploads, quotas and listings are scoped to that tenant.
- **Encryption.** A customer managed KMS key (rotation on) encrypts the S3 bucket, DynamoDB table, SQS queues and
  SNS topic. The bucket blocks public access and denies non-TLS requests. EBS volumes are encrypted.
- **Least privilege.** Each role is scoped to its prefix or resource; autoscaling writes are tag-conditioned to
  this stack's group.
- **Content safety.** Prompts go through an Amazon Bedrock Guardrail before they reach the GPU; input images and
  sampled output frames go through Amazon Rekognition moderation. Flagged renders land in `quarantine/` with no
  download URL.
- **Audit.** Every tool call and job transition is logged with the tenant and principal (CloudWatch Logs), job
  records keep the guardrail and moderation results, and each output has a provenance manifest (input hashes,
  model, seed, worker, output hash). Gateway API calls are in CloudTrail.

## Your responsibilities
- Model licenses and acceptable-use terms (see THIRD_PARTY_MODELS.md).
- Who may call the gateway (IAM policies on `bedrock-agentcore:InvokeGateway`, or your IdP's client and audience).
- Reviewing quarantined outputs and tuning the guardrail and moderation thresholds to your policy.
