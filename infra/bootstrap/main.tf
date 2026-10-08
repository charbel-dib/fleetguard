terraform {
  required_version = ">= 1.13, < 2.0"
  backend "local" {}
  required_providers {
    aws = { source = "hashicorp/aws", version = "~> 6.0" }
  }
}

provider "aws" {
  region = var.region
  default_tags {
    tags = { Project = var.project, ManagedBy = "Terraform" }
  }
}

variable "region" {
  type    = string
  default = "eu-west-3"
}
variable "project" {
  type    = string
  default = "fleetguard"
  validation {
    condition     = can(regex("^[a-z][a-z0-9-]{2,19}$", var.project))
    error_message = "Use a 3–20 character lowercase project name."
  }
}
variable "github_oidc_repository" {
  type        = string
  description = "Exact repo portion of the observed OIDC sub, including immutable IDs if present."
  validation {
    condition     = can(regex("^[A-Za-z0-9_.-]+(@[0-9]+)?/[A-Za-z0-9_.-]+(@[0-9]+)?$", var.github_oidc_repository))
    error_message = "Copy the exact owner/repo or owner@ID/repo@ID portion from the identity workflow."
  }
}
variable "existing_oidc_provider_arn" {
  type        = string
  default     = null
  description = "Reuse an existing GitHub OIDC provider in this AWS account when present."
}
variable "budget_email" {
  type = string
  validation {
    condition     = can(regex("^[^@ ]+@[^@ ]+\\.[^@ ]+$", var.budget_email))
    error_message = "Supply your email for monthly AWS account budget alerts."
  }
}
variable "monthly_budget_usd" {
  type = number
  validation {
    condition     = var.monthly_budget_usd > 0
    error_message = "Choose a positive monthly alert budget, not a spending cap."
  }
}
resource "aws_budgets_budget" "account" {
  name         = "${var.project}-account-monthly-alert"
  budget_type  = "COST"
  limit_amount = tostring(var.monthly_budget_usd)
  limit_unit   = "USD"
  time_unit    = "MONTHLY"
  notification {
    comparison_operator        = "GREATER_THAN"
    threshold                  = 80
    threshold_type             = "PERCENTAGE"
    notification_type          = "ACTUAL"
    subscriber_email_addresses = [var.budget_email]
  }
}

data "aws_caller_identity" "current" {}
locals {
  account      = data.aws_caller_identity.current.account_id
  bucket_names = { models = "${var.project}-${local.account}-${var.region}-models", state = "${var.project}-${local.account}-${var.region}-state" }
  provider_arn = var.existing_oidc_provider_arn != null ? var.existing_oidc_provider_arn : aws_iam_openid_connect_provider.github[0].arn
}

resource "aws_s3_bucket" "storage" {
  for_each      = local.bucket_names
  bucket        = each.value
  force_destroy = false
}
resource "aws_s3_bucket_versioning" "storage" {
  for_each = aws_s3_bucket.storage
  bucket   = each.value.id
  versioning_configuration {
    status = "Enabled"
  }
}
resource "aws_s3_bucket_server_side_encryption_configuration" "storage" {
  for_each = aws_s3_bucket.storage
  bucket   = each.value.id
  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}
resource "aws_s3_bucket_public_access_block" "storage" {
  for_each                = aws_s3_bucket.storage
  bucket                  = each.value.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}
resource "aws_s3_bucket_policy" "tls" {
  for_each = aws_s3_bucket.storage
  bucket   = each.value.id
  policy = jsonencode({
    Version   = "2012-10-17"
    Statement = [{ Effect = "Deny", Principal = "*", Action = "s3:*", Resource = [each.value.arn, "${each.value.arn}/*"], Condition = { Bool = { "aws:SecureTransport" = "false" } } }]
  })
}
resource "aws_ecr_repository" "app" {
  name                 = var.project
  image_tag_mutability = "IMMUTABLE"
  image_scanning_configuration {
    scan_on_push = true
  }
  encryption_configuration {
    encryption_type = "AES256"
  }
  force_delete = false
}
# No image expiry policy: retaining named candidates preserves rollback availability.
resource "aws_iam_openid_connect_provider" "github" {
  count          = var.existing_oidc_provider_arn == null ? 1 : 0
  url            = "https://token.actions.githubusercontent.com"
  client_id_list = ["sts.amazonaws.com"]
}
resource "aws_iam_role" "github" {
  for_each = toset(["cloud-build", "staging", "production"])
  name     = "${var.project}-${each.key}-github"
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect = "Allow", Action = "sts:AssumeRoleWithWebIdentity", Principal = { Federated = local.provider_arn }
      Condition = { StringEquals = {
        "token.actions.githubusercontent.com:aud" = "sts.amazonaws.com"
        "token.actions.githubusercontent.com:sub" = "repo:${var.github_oidc_repository}:environment:${each.key}"
      } }
    }]
  })
}
resource "aws_iam_role_policy" "build" {
  role = aws_iam_role.github["cloud-build"].id
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      { Effect = "Allow", Action = ["ecr:GetAuthorizationToken"], Resource = "*" },
      { Effect = "Allow", Action = ["ecr:BatchCheckLayerAvailability", "ecr:InitiateLayerUpload", "ecr:UploadLayerPart", "ecr:CompleteLayerUpload", "ecr:PutImage", "ecr:BatchGetImage", "ecr:GetDownloadUrlForLayer", "ecr:DescribeImages"], Resource = aws_ecr_repository.app.arn },
      { Effect = "Allow", Action = ["s3:GetObjectVersion"], Resource = "${aws_s3_bucket.storage["models"].arn}/models/*" }
    ]
  })
}
resource "aws_iam_role_policy" "deploy" {
  for_each = toset(["staging", "production"])
  role     = aws_iam_role.github[each.key].id
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      { Effect = "Allow", Action = ["ecs:DescribeServices", "ecs:UpdateService"], Resource = "arn:aws:ecs:${var.region}:${local.account}:service/${var.project}-${each.key}/${var.project}-${each.key}" },
      { Effect = "Allow", Action = ["ecs:DescribeTaskDefinition"], Resource = "arn:aws:ecs:${var.region}:${local.account}:task-definition/${var.project}-${each.key}:*" },
      { Effect = "Allow", Action = ["ecs:RegisterTaskDefinition"], Resource = "*" },
      { Effect = "Allow", Action = ["ecr:DescribeImages"], Resource = aws_ecr_repository.app.arn },
      { Effect = "Allow", Action = ["iam:PassRole"], Resource = ["arn:aws:iam::${local.account}:role/${var.project}-${each.key}-execution", "arn:aws:iam::${local.account}:role/${var.project}-${each.key}-task"], Condition = { StringEquals = { "iam:PassedToService" = "ecs-tasks.amazonaws.com" } } }
    ]
  })
}

output "models_bucket" {
  value = aws_s3_bucket.storage["models"].id
}
output "state_bucket" {
  value = aws_s3_bucket.storage["state"].id
}
output "repository_url" {
  value = aws_ecr_repository.app.repository_url
}
output "github_roles" {
  value = { for key, role in aws_iam_role.github : key => role.arn }
}
