terraform {
  required_version = ">= 1.6"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 6.0"
    }
  }
}
provider "aws" {
  region = var.region
}
data "aws_partition" "current" {}
locals {
  bucket = "arn:${data.aws_partition.current.partition}:s3:::${var.artifact_bucket}"
}
resource "aws_cloudwatch_log_group" "app" {
  name              = "/medical-imaging/${var.name}"
  retention_in_days = 30
}
resource "aws_iam_role" "app" {
  name = "${var.name}-instance"
  assume_role_policy = jsonencode({
    Version   = "2012-10-17"
    Statement = [{ Effect = "Allow", Principal = { Service = "ec2.amazonaws.com" }, Action = "sts:AssumeRole" }]
  })
}
resource "aws_iam_role_policy" "app" {
  role = aws_iam_role.app.id
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      { Effect = "Allow", Action = "s3:GetObject", Resource = "${local.bucket}/${var.artifact_prefix}/*" },
      { Effect = "Allow", Action = "s3:ListBucket", Resource = local.bucket,
      Condition = { StringEquals = { "s3:prefix" = "${var.artifact_prefix}/" } } },
      { Effect = "Allow", Action = "secretsmanager:GetSecretValue", Resource = var.backend_secret_arn },
      { Effect = "Allow", Action = ["logs:CreateLogStream", "logs:PutLogEvents"],
      Resource = "${aws_cloudwatch_log_group.app.arn}:log-stream:*" }
    ]
  })
}
resource "aws_iam_role_policy_attachment" "ssm" {
  role       = aws_iam_role.app.name
  policy_arn = "arn:${data.aws_partition.current.partition}:iam::aws:policy/AmazonSSMManagedInstanceCore"
}
resource "aws_iam_instance_profile" "app" {
  name = var.name
  role = aws_iam_role.app.name
}
resource "aws_security_group" "app" {
  name        = var.name
  vpc_id      = var.vpc_id
  description = "Inference host behind HTTPS load balancer"
}
resource "aws_vpc_security_group_ingress_rule" "frontend" {
  security_group_id            = aws_security_group.app.id
  referenced_security_group_id = var.alb_security_group_id
  ip_protocol                  = "tcp"
  from_port                    = 8501
  to_port                      = 8501
}
resource "aws_vpc_security_group_egress_rule" "app" {
  for_each          = var.egress_rules
  security_group_id = aws_security_group.app.id
  cidr_ipv4         = each.value.cidr
  ip_protocol       = "tcp"
  from_port         = each.value.port
  to_port           = each.value.port
}
resource "aws_instance" "app" {
  ami                         = var.ami_id
  instance_type               = var.instance_type
  subnet_id                   = var.subnet_id
  vpc_security_group_ids      = [aws_security_group.app.id]
  associate_public_ip_address = false
  iam_instance_profile        = aws_iam_instance_profile.app.name
  monitoring                  = true
  metadata_options {
    http_tokens                 = "required"
    http_put_response_hop_limit = 2
  }
  root_block_device {
    encrypted   = true
    volume_type = "gp3"
    volume_size = 60
  }
  user_data_replace_on_change = true
  user_data = templatefile("${path.module}/bootstrap.sh.tftpl", {
    script = filebase64("${path.module}/../scripts/deploy.sh")
    args = join(" ", [for value in [
      var.region, var.backend_secret_arn, "s3://${var.artifact_bucket}/${var.artifact_prefix}",
      aws_cloudwatch_log_group.app.name, var.backend_image, var.frontend_image
    ] : base64encode(value)])
  })
  tags       = { Name = var.name }
  depends_on = [aws_iam_role_policy.app, aws_iam_role_policy_attachment.ssm]
}
resource "aws_lb_target_group_attachment" "frontend" {
  target_group_arn = var.alb_target_group_arn
  target_id        = aws_instance.app.id
  port             = 8501
}
resource "aws_cloudwatch_metric_alarm" "host" {
  alarm_name          = "${var.name}-host-status"
  namespace           = "AWS/EC2"
  metric_name         = "StatusCheckFailed"
  statistic           = "Maximum"
  period              = 60
  evaluation_periods  = 2
  comparison_operator = "GreaterThanOrEqualToThreshold"
  threshold           = 1
  dimensions          = { InstanceId = aws_instance.app.id }
  alarm_actions       = var.alarm_actions
}
output "instance_id" {
  value = aws_instance.app.id
}
