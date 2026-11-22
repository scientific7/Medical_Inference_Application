variable "region" {
  type = string
}

variable "vpc_id" {
  type = string
}

variable "subnet_id" {
  type = string
}

variable "alb_security_group_id" {
  type = string
}

variable "alb_target_group_arn" {
  type = string
}

variable "ami_id" {
  type = string
}

variable "artifact_bucket" {
  type = string
}

variable "backend_secret_arn" {
  type = string
}

variable "backend_image" {
  type = string
  validation {
    condition     = can(regex("^[A-Za-z0-9._:/-]+@sha256:[a-f0-9]{64}$", var.backend_image))
    error_message = "Supply a digest-pinned image reference."
  }
}

variable "frontend_image" {
  type = string
  validation {
    condition     = can(regex("^[A-Za-z0-9._:/-]+@sha256:[a-f0-9]{64}$", var.frontend_image))
    error_message = "Supply a digest-pinned image reference."
  }
}

variable "name" {
  type    = string
  default = "medical-imaging-inference"
}
variable "instance_type" {
  type    = string
  default = "m7i.large"
}
variable "artifact_prefix" {
  type    = string
  default = "approved"
  validation {
    condition     = can(regex("^[A-Za-z0-9_-]+(/[A-Za-z0-9_-]+)*$", var.artifact_prefix))
    error_message = "Use a nonempty prefix without leading or trailing slashes."
  }
}
variable "egress_rules" {
  type = map(object({ cidr = string, port = number }))
}
variable "alarm_actions" {
  type    = list(string)
  default = []
}
