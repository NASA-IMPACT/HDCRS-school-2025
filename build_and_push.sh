#! /bin/bash

export ECR_URL="${AWS_ACCOUNT_ID}.dkr.ecr.us-west-2.amazonaws.com"
export ECR_IMAGE_NAME=hdcrs:latest
aws ecr get-login-password --region us-west-2 | \
docker login --password-stdin --username AWS $ECR_URL

# docker build . --platform linux/amd64 -t $ECR_URL/eo_training:latest

# docker push $ECR_URL/eo_training:latest

docker build -f Dockerfile.inference --platform linux/amd64 -t $ECR_URL/$ECR_IMAGE_NAME


docker push $ECR_URL/$ECR_IMAGE_NAME
