# IAM for the `minillm-agent` user

**Current setup (decided 2026-11-14): the user has full access.** Attached policies:

| Policy | Type | Purpose |
|---|---|---|
| `AdministratorAccess` | AWS managed | Everything, including IAM (list/edit policies) |
| `AWSMCPSignInOAuthAccessPolicy` | AWS managed | Sign-in for the AWS MCP server |

The scoped policy below is kept as the least-privilege alternative; it is redundant while
`AdministratorAccess` is attached.

| Policy | Type | Purpose |
|---|---|---|
| [`minillm-agent-policy.json`](minillm-agent-policy.json) | customer managed `minillm-agent-policy` | EC2, SSM, S3, quota requests, read-only costs, DynamoDB, RDS, Lambda (+ its basic execution role), Bedrock invoke, Free Tier activity status |
| `AWSMCPSignInOAuthAccessPolicy` | AWS managed | Sign-in for the AWS MCP server |

It is a customer managed policy (6,144-character limit) because IAM caps all inline policies on one
user at 2,048 characters in total, and this policy is about 3,700.

Every statement uses `"Resource": "*"`. Limits come only from conditions:
- launch any instance type, only in `us-east-1`;
- start, stop, reboot, terminate only instances tagged `project=minillm` (tag them at launch);
- `iam:PassRole` only to Lambda; `iam:AttachRolePolicy` only for `AWSLambdaBasicExecutionRole`;
  service-linked roles only for Service Quotas and RDS.

With `AdministratorAccess` the user can change IAM too. Claude still asks before starting or
launching instances, before terminating anything, and before any purchase (e.g. Capacity Blocks).

## When scope grows
- An EC2 instance role (so instances reach S3 without keys) would need `iam:PassRole` to EC2 and its
  own role; not set up now.
