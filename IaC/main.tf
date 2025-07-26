terraform {
  required_version = ">= 1.4"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 6.0"
    }
  }
}

provider "aws" {
  region = var.aws_region
}


# -----------------------------------------------------------------------------
# S3 Bucket
# -----------------------------------------------------------------------------
resource "aws_s3_bucket" "mlops_capstone_bucket" {
  bucket = var.s3_bucket_name

  tags = {
    Name = var.s3_bucket_tag_name
  }
}

# -----------------------------------------------------------------------------
# Networking prerequisites
# -----------------------------------------------------------------------------
data "aws_vpc" "default" {
  default = true
}

data "aws_subnet" "default" {
  vpc_id            = data.aws_vpc.default.id
  availability_zone = var.availability_zone
}

# -----------------------------------------------------------------------------
# Security Group allowing SSH
# -----------------------------------------------------------------------------
resource "aws_security_group" "mlops_capstone_sg" {
  name        = var.aws_security_group_name
  description = "Security Group for MLOPS Capstone Project"
  vpc_id      = data.aws_vpc.default.id

  dynamic "ingress" {
    for_each = var.ingress_rules
    content {
      description = ingress.value.description
      from_port   = ingress.value.from_port
      to_port     = ingress.value.to_port
      protocol    = ingress.value.protocol
      cidr_blocks = ingress.value.cidr_blocks
    }
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = var.egress_protocol
    cidr_blocks = [var.allowed_ssh_cidr]
  }

  tags = {
    Name = var.aws_security_group_tag_name
  }
}

# -----------------------------------------------------------------------------
# IAM Role & Instance Profile for S3 Full Access 
# -----------------------------------------------------------------------------
resource "aws_iam_role" "ec2_s3_access" {
  name = var.iam_role_name
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Action    = "sts:AssumeRole"
      Effect    = "Allow"
      Principal = { Service = "ec2.amazonaws.com" }
    }]
  })
}

resource "aws_iam_role_policy_attachment" "s3_full" {
  role       = aws_iam_role.ec2_s3_access.name
  policy_arn = var.s3_policy_arn
}

resource "aws_iam_instance_profile" "mlopsz_2025_capstoneproject_instance_profile" {
  name = var.iam_instance_profile_name
  role = aws_iam_role.ec2_s3_access.name
}

# -----------------------------------------------------------------------------
# EC2 Instance
# -----------------------------------------------------------------------------
resource "aws_instance" "mlops_zoomcamp_project_VM" {
  ami                    = var.instance_ami # Ubuntu 24.04 LTS
  instance_type          = var.instance_type
  availability_zone      = var.availability_zone
  subnet_id              = data.aws_subnet.default.id
  vpc_security_group_ids = [aws_security_group.mlops_capstone_sg.id]
  key_name               = var.key_name # existing key-pair
  iam_instance_profile   = aws_iam_instance_profile.mlopsz_2025_capstoneproject_instance_profile.name

  # Root disk (50 GiB gp3)
  root_block_device {
    volume_type           = var.volume_type
    volume_size           = var.root_volume_size
    delete_on_termination = true
  }


  credit_specification {
    cpu_credits = "unlimited"
  }

  tags = {
    Name        = var.aws_instance_tag_name
    Environment = var.environment
  }
}

# -----------------------------------------------------------------------------
# Outputs
# -----------------------------------------------------------------------------
output "s3_bucket_name" {
  description = "Name of the newly created S3 bucket"
  value       = aws_s3_bucket.mlops_capstone_bucket.bucket
}

output "ec2_public_ip" {
  description = "Public IPv4 address of the new EC2 instance"
  value       = aws_instance.mlops_zoomcamp_project_VM.public_ip
}
