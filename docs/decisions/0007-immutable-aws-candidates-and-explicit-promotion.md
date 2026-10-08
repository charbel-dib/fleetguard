# ADR 0007: Immutable AWS candidates and explicit promotion

Status: accepted for implementation; live AWS acceptance pending.

AWS was selected by the user for cloud engineering practice and portfolio evidence. Separate
versioned S3 model storage, private ECR and ECS Fargate make the artifact/service boundaries explicit.
CloudFront supplies a default HTTPS hostname and a private VPC origin to an internal ALB. HTTP remains
inside the VPC; a purchased domain/ACM origin certificate is not required for this initial demo.

The frozen trusted model is baked into an image together with the built frontend and serving-only
Python environment. Promotion identifies that image by digest. The application needs no AWS API
permissions or runtime model download, and a model/network storage outage cannot change its decision.
Image size/storage and rebuilding for a model change are the accepted tradeoffs.

GitHub OIDC roles separate publication from staging/production service updates. Exact observed
subjects handle both legacy and immutable repository-ID claims. Production requires the same
candidate's verified staging receipt. Failed rollout or public contract check restores and verifies
the previous task definition; a staging-only failed-startup drill supplies real rollback evidence.

Provisioning is reviewed separately through Terraform; normal PR CI never receives AWS credentials.
The infrastructure and code are delivered before asking the user to perform account setup/provisioning.
No cloud endpoint, AWS price, resource capacity or rollback result is claimed until measured.
