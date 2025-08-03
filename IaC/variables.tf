variable "aws_region" {
  type    = string
  default = "eu-north-1"
}

variable "availability_zone" {
  type    = string
  default = "eu-north-1a"
}

variable "key_name" {
  type        = string
  description = "Name of the existing EC2 key-pair in AWS"
  default     = "sidd4ML"
}

variable "instance_ami" {
  type        = string
  description = "AMI ID for the EC2 instance"
  default     = "ami-0c1ac8a41498c1a9c"
}

variable "instance_type" {
  type    = string
  default = "t3.xlarge"
}

variable "root_volume_size" {
  type    = number
  default = 50
}

variable "volume_type" {
  type    = string
  default = "gp3"
}

variable "allowed_ssh_cidr" {
  type    = string
  default = "0.0.0.0/0"
}

variable "ingress_protocol" {
  type    = string
  default = "tcp"
}

variable "egress_protocol" {
  type    = string
  default = "-1"
}


variable "s3_policy_arn" {
  type    = string
  default = "arn:aws:iam::aws:policy/AmazonS3FullAccess"
}

variable "environment" {
  type    = string
  default = "dev"
}
variable "s3_bucket_name" {
  type        = string
  description = "Name of the S3 bucket"
  default     = "mlops-zoomcamp-bike-sharing-bucket"
}

variable "aws_instance_tag_name" {
  type    = string
  default = "mlops-zoomcamp-EC2_VM"
}

variable "iam_instance_profile_name" {
  type    = string
  default = "mlopsz-instance-profile-2025-capstoneproject"
}

variable "iam_role_name" {
  type    = string
  default = "project-VM-S3-access"
}

variable "aws_security_group_tag_name" {
  type    = string
  default = "mlops-capstone-sg-terraform"
}

variable "aws_security_group_name" {
  type    = string
  default = "mlops-capstone-sg"
}

variable "s3_bucket_tag_name" {
  type    = string
  default = "mlops-zoomcamp-bike-sharing-bucket-terraform"
}

variable "ingress_rules" {
  type = list(object({
    description = string
    from_port   = number
    to_port     = number
    protocol    = string
    cidr_blocks = list(string)
  }))
  default = [
    {
      description = "SSH"
      from_port   = 22
      to_port     = 22
      protocol    = "tcp"
      cidr_blocks = ["0.0.0.0/0"] # <- literal default
    },
    {
      description = "MLFlow"
      from_port   = 5000
      to_port     = 5000
      protocol    = "tcp"
      cidr_blocks = ["0.0.0.0/0"]
    },
    {
      description = "Prefect"
      from_port   = 4200
      to_port     = 4201
      protocol    = "tcp"
      cidr_blocks = ["0.0.0.0/0"]
    },
    {
      description = "Evidently-AI"
      from_port   = 8000
      to_port     = 8000
      protocol    = "tcp"
      cidr_blocks = ["0.0.0.0/0"]
    },
    {
      description = "Grafana"
      from_port   = 3000
      to_port     = 3000
      protocol    = "tcp"
      cidr_blocks = ["0.0.0.0/0"]
    },
    {
      description = "Fast-API"
      from_port   = 8010
      to_port     = 8010
      protocol    = "tcp"
      cidr_blocks = ["0.0.0.0/0"]
    },
    {
      description = "Streamlit"
      from_port   = 8501
      to_port     = 8501
      protocol    = "tcp"
      cidr_blocks = ["0.0.0.0/0"]
    }
  ]
}
