# GitHub Actions deployment

This GitHub Actions workflow tests the Rust service, builds a Linux release
binary, uploads it to a private S3 release prefix, then tells AWS Systems
Manager to install it on the production EC2 instance. The instance validates
the SHA-256, restarts `maskarapay`, checks `/healthz`, and restarts active
`cloudflared`.

The ONNX model remains in `/opt/maskarapay/assets` on the EC2 instance. CI
never downloads or exports the model, so it cannot recreate the previous
PyTorch disk-space failure.

```text
main push
    |
GitHub Actions: test -> Linux build -> SHA-256 -> private S3
    |
AWS Systems Manager Run Command
    |
EC2: verify -> install -> health check -> restart Cloudflare Tunnel
```

The workflow is [`.github/workflows/deploy.yml`](../.github/workflows/deploy.yml).
Pull requests only verify and build. A push to `main`, or a manual dispatch,
deploys to the `production` environment.

## 1. Prepare the EC2 instance once

The EC2 instance must already run both MaskaraPay and `cloudflared`. Install
the AWS CLI:

```bash
sudo apt-get update
sudo apt-get install -y awscli
aws --version
```

Keep this production override so a deployment can never attempt a Python or
Torch model export:

```ini
# /etc/systemd/system/maskarapay.service.d/override.conf
[Service]
Environment=MASKARA_SKIP_MODEL_DOWNLOAD=1
```

Its instance profile needs `AmazonSSMManagedInstanceCore` plus this inline
policy. Substitute the private deployment bucket name:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": "s3:GetObject",
      "Resource": "arn:aws:s3:::DEPLOY_BUCKET/maskarapay/releases/*"
    }
  ]
}
```

## 2. Create the deployment bucket

In the S3 console, create a general-purpose bucket in the same Region as the
instance, for example `maskarapay-releases-ACCOUNT_ID`. Keep **Block all public
access** enabled. This bucket contains service releases only; do not use a
public model bucket for deployments. Set an S3 lifecycle rule to expire the
`maskarapay/releases/` prefix after 30 days.

## 3. Give GitHub Actions temporary AWS access

In **AWS IAM > Identity providers**, add OpenID Connect provider:

| Field | Value |
| --- | --- |
| Provider URL | `https://token.actions.githubusercontent.com` |
| Audience | `sts.amazonaws.com` |

Create an IAM role named, for example, `github-maskarapay-deploy` with this
trust policy. It limits role assumption to this repository's `production`
environment:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Principal": {
        "Federated": "arn:aws:iam::AWS_ACCOUNT_ID:oidc-provider/token.actions.githubusercontent.com"
      },
      "Action": "sts:AssumeRoleWithWebIdentity",
      "Condition": {
        "StringEquals": {
          "token.actions.githubusercontent.com:aud": "sts.amazonaws.com",
          "token.actions.githubusercontent.com:sub": "repo:Somu878/maskarapay:environment:production"
        }
      }
    }
  ]
}
```

Attach this inline permissions policy to that role. Replace every uppercase
placeholder before saving:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "WriteReleaseArtifacts",
      "Effect": "Allow",
      "Action": "s3:PutObject",
      "Resource": "arn:aws:s3:::DEPLOY_BUCKET/maskarapay/releases/*"
    },
    {
      "Sid": "DeployToOneInstance",
      "Effect": "Allow",
      "Action": "ssm:SendCommand",
      "Resource": [
        "arn:aws:ssm:AWS_REGION::document/AWS-RunShellScript",
        "arn:aws:ec2:AWS_REGION:AWS_ACCOUNT_ID:instance/EC2_INSTANCE_ID"
      ]
    },
    {
      "Sid": "ReadDeploymentStatus",
      "Effect": "Allow",
      "Action": "ssm:GetCommandInvocation",
      "Resource": "*"
    }
  ]
}
```

## 4. Configure the GitHub repository

1. Open **Settings > Environments** and create `production`. Under deployment
   branches, allow `main`. Do not add a required reviewer if deployments should
   be fully automatic.
2. Open **Settings > Secrets and variables > Actions > Variables** and add:

| Variable | Value |
| --- | --- |
| `AWS_REGION` | `ap-south-1` |
| `AWS_ROLE_ARN` | `arn:aws:iam::AWS_ACCOUNT_ID:role/github-maskarapay-deploy` |
| `DEPLOY_BUCKET` | The private bucket from step 2 |
| `EC2_INSTANCE_ID` | The EC2 instance ID, for example `i-0123456789abcdef0` |
| `API_HOST` | `api.somuso.fun` |

No GitHub Actions secrets or AWS access keys are required. An AWS role ARN,
region, bucket name, instance ID, and public host name are identifiers rather
than credentials, so repository variables are appropriate. Do **not** create
or share `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, a Cloudflare tunnel
token, S3 pre-signed URLs, or private keys.

3. Protect `main` with a branch ruleset so only reviewed changes can merge.
   The workflow deploys only after that protected branch changes.

## 5. First deployment

Commit this workflow and push it to `main`. In **Actions**, open **Build and
deploy MaskaraPay**. The first run will appear in the `production` environment.
The deployment job prints the Systems Manager command result. A successful
result ends with `MaskaraPay deployment completed`.

If the new binary fails its local health check, the instance restores the
previous binary, restarts the service, and marks the GitHub workflow failed.

## Sources

- [GitHub Actions OIDC in AWS](https://docs.github.com/en/actions/how-tos/secure-your-work/security-harden-deployments/oidc-in-aws)
- [GitHub Actions variables](https://docs.github.com/en/actions/reference/workflows-and-actions/variables)
- [AWS Systems Manager Run Command](https://docs.aws.amazon.com/systems-manager/latest/userguide/running-commands.html)
