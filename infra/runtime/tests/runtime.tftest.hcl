mock_provider "aws" {
  mock_data "aws_caller_identity" { defaults = { account_id = "123456789012" } }
  mock_data "aws_availability_zones" { defaults = { names = ["eu-west-3a", "eu-west-3b"] } }
}
variables {
  environment = "staging"
  candidate = {
    image         = "123456789012.dkr.ecr.eu-west-3.amazonaws.com/fleetguard@sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
    git_sha       = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
    bundle_sha256 = "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc"
  }
}
run "private_origin_and_frozen_unprivileged_runtime" {
  command = plan
  assert {
    condition     = aws_lb.app.internal && aws_cloudfront_distribution.app.default_cache_behavior[0].viewer_protocol_policy == "redirect-to-https"
    error_message = "The origin must be private and the viewer must use HTTPS."
  }
  assert {
    condition     = aws_cloudfront_vpc_origin.app.vpc_origin_endpoint_config[0].origin_protocol_policy == "http-only"
    error_message = "Origin transport is explicitly HTTP inside the VPC; do not claim end-to-end TLS."
  }
  assert {
    condition     = aws_ecs_service.app.deployment_circuit_breaker[0].enable && aws_ecs_service.app.deployment_circuit_breaker[0].rollback
    error_message = "ECS must reject failing rollouts and restore the previous revision."
  }
  assert {
    condition     = jsondecode(aws_ecs_task_definition.app.container_definitions)[0].readonlyRootFilesystem && jsondecode(aws_ecs_task_definition.app.container_definitions)[0].user == "10001:10001"
    error_message = "The application must remain unprivileged with a read-only root filesystem."
  }
  assert {
    condition     = jsondecode(aws_ecs_task_definition.app.container_definitions)[0].environment[0].value == "false"
    error_message = "Synthetic fixtures must never become the cloud demo model."
  }
}
run "reject_invalid_image_digest" {
  command = plan
  variables {
    candidate = { image = "123456789012.dkr.ecr.eu-west-3.amazonaws.com/fleetguard@sha256:short", git_sha = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb", bundle_sha256 = "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc" }
  }
  expect_failures = [var.candidate]
}
