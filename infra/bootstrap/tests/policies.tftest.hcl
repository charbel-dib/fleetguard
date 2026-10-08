mock_provider "aws" {
  mock_data "aws_caller_identity" {
    defaults = { account_id = "123456789012" }
  }
}
variables {
  github_oidc_repository     = "owner@123/repo@456"
  budget_email               = "test@example.com"
  monthly_budget_usd         = 30
  existing_oidc_provider_arn = "arn:aws:iam::123456789012:oidc-provider/token.actions.githubusercontent.com"
}
run "private_versioned_storage_and_exact_subjects" {
  command = plan
  assert {
    condition     = aws_ecr_repository.app.image_tag_mutability == "IMMUTABLE"
    error_message = "Candidates must not be overwritten by mutable tags."
  }
  assert {
    condition     = alltrue([for item in aws_s3_bucket_versioning.storage : item.versioning_configuration[0].status == "Enabled"])
    error_message = "Models and state must be versioned."
  }
  assert {
    condition     = alltrue([for item in aws_s3_bucket_public_access_block.storage : item.block_public_policy && item.restrict_public_buckets && item.block_public_acls && item.ignore_public_acls])
    error_message = "Models/state must remain private."
  }
  assert {
    condition     = jsondecode(aws_iam_role.github["staging"].assume_role_policy).Statement[0].Condition.StringEquals["token.actions.githubusercontent.com:sub"] == "repo:owner@123/repo@456:environment:staging"
    error_message = "Use the exact observed OIDC subject, without broad wildcards."
  }
}
run "reuse_existing_oidc_provider" {
  command = plan
  variables { existing_oidc_provider_arn = "arn:aws:iam::123456789012:oidc-provider/token.actions.githubusercontent.com" }
  assert {
    condition     = length(aws_iam_openid_connect_provider.github) == 0
    error_message = "Do not duplicate an account's GitHub OIDC provider."
  }
}
run "create_oidc_provider_when_absent" {
  command = plan
  variables { existing_oidc_provider_arn = null }
  assert {
    condition     = length(aws_iam_openid_connect_provider.github) == 1
    error_message = "Create exactly one GitHub provider when the account has none."
  }
}
