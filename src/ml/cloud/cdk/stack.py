"""CDK stack for ML cloud training infrastructure."""

from __future__ import annotations

from aws_cdk import CfnOutput, Stack
from aws_cdk import aws_iam as iam
from aws_cdk import aws_s3 as s3
from constructs import Construct

SAGEMAKER_SERVICE_PRINCIPAL = "sagemaker.amazonaws.com"

ACTION_SETS = {
    "logs": [
        "logs:CreateLogGroup",
        "logs:CreateLogStream",
        "logs:PutLogEvents",
    ],
    "s3_bucket": [
        "s3:ListBucket",
    ],
    "s3_objects": [
        "s3:GetObject",
        "s3:PutObject",
    ],
    "sagemaker_jobs": [
        "sagemaker:CreateTrainingJob",
        "sagemaker:DescribeTrainingJob",
        "sagemaker:ListTrainingJobs",
        "sagemaker:StopTrainingJob",
    ],
}


class MlCloudTrainingStack(Stack):
    """Provision S3 storage and IAM roles for ML cloud training."""

    def __init__(
        self,
        scope: Construct,
        construct_id: str,
        training_user_name: str | None = None,
        **kwargs: object,
    ) -> None:
        """Initialize the ML cloud training stack.

        Args:
            scope: CDK construct scope.
            construct_id: Stack identifier.
            training_user_name: IAM username to attach training policy to (optional).
            **kwargs: CDK Stack keyword arguments.
        """
        super().__init__(scope, construct_id, **kwargs)

        training_bucket = s3.Bucket(
            self,
            "MlTrainingArtifacts",
            versioned=True,
            encryption=s3.BucketEncryption.S3_MANAGED,
            block_public_access=s3.BlockPublicAccess.BLOCK_ALL,
            enforce_ssl=True,
        )

        execution_role = iam.Role(
            self,
            "SageMakerExecutionRole",
            assumed_by=iam.ServicePrincipal(SAGEMAKER_SERVICE_PRINCIPAL),
            description="SageMaker execution role for ML cloud training.",
        )

        execution_role.add_to_policy(
            iam.PolicyStatement(
                actions=ACTION_SETS["s3_bucket"],
                resources=[training_bucket.bucket_arn],
            )
        )
        execution_role.add_to_policy(
            iam.PolicyStatement(
                actions=ACTION_SETS["s3_objects"],
                resources=[training_bucket.arn_for_objects("*")],
            )
        )
        execution_role.add_to_policy(
            iam.PolicyStatement(
                actions=ACTION_SETS["logs"],
                resources=[self._sagemaker_log_group_arn()],
            )
        )

        training_user_policy = iam.ManagedPolicy(
            self,
            "MlCloudTrainingUserPolicy",
            description="Least-privilege policy for submitting ML cloud training jobs.",
            statements=[
                iam.PolicyStatement(
                    actions=ACTION_SETS["sagemaker_jobs"],
                    resources=["*"],
                ),
                iam.PolicyStatement(
                    actions=["iam:PassRole"],
                    resources=[execution_role.role_arn],
                ),
                iam.PolicyStatement(
                    actions=ACTION_SETS["s3_bucket"],
                    resources=[training_bucket.bucket_arn],
                ),
                iam.PolicyStatement(
                    actions=ACTION_SETS["s3_objects"],
                    resources=[training_bucket.arn_for_objects("*")],
                ),
                iam.PolicyStatement(
                    actions=ACTION_SETS["logs"],
                    resources=[self._sagemaker_log_group_arn()],
                ),
            ],
        )

        # Attach policy to IAM user if specified
        if training_user_name:
            training_user = iam.User.from_user_name(
                self,
                "TrainingUser",
                user_name=training_user_name,
            )
            training_user_policy.attach_to_user(training_user)

        CfnOutput(
            self,
            "TrainingBucketName",
            value=training_bucket.bucket_name,
        )
        CfnOutput(
            self,
            "SageMakerExecutionRoleArn",
            value=execution_role.role_arn,
        )
        CfnOutput(
            self,
            "TrainingUserPolicyArn",
            value=training_user_policy.managed_policy_arn,
        )

    def _sagemaker_log_group_arn(self) -> str:
        """Return the CloudWatch Logs ARN pattern for SageMaker log groups."""
        return f"arn:aws:logs:{self.region}:{self.account}:log-group:/aws/sagemaker/*"
