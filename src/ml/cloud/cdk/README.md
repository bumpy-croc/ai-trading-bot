# ML Cloud Training CDK

CDK stack to provision the S3 bucket and IAM roles needed for `atb train cloud`.

## Resources

- S3 bucket for training data and model artifacts.
- SageMaker execution role scoped to the bucket and CloudWatch Logs.
- Managed policy for the training user that can submit jobs and pass the execution role.

## Prerequisites

- AWS CDK v2 installed.
- AWS credentials with permissions to deploy CloudFormation stacks.

## Deploy

```bash
cd src/ml/cloud/cdk
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cdk bootstrap
cdk deploy
```

## Outputs

After deploy, capture the outputs and set your environment variables:

- `TrainingBucketName` -> `SAGEMAKER_S3_BUCKET`
- `SageMakerExecutionRoleArn` -> `SAGEMAKER_ROLE_ARN`

The stack also outputs `TrainingUserPolicyArn`. Attach that managed policy to the IAM user that will run cloud training commands.
