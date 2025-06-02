%%sh
docker build . --platform linux/amd64 -t sagemaker-build

export ECR_URL="<account-number>.dkr.ecr.us-west-2.amazonaws.com"

aws ecr get-login-password --region us-west-2 | \
  docker login --password-stdin --username AWS $ECR_URL

docker tag sagemaker-build $ECR_URL/eo_training:latest

docker push $ECR_URL/eo_training:latest

docker build -f Dockerfile.inference --platform linux/amd64 -t sagemaker-inference

export ECR_URL="<account-number>.dkr.ecr.us-west-2.amazonaws.com"

aws ecr get-login-password --region us-west-2 | \
  docker login --password-stdin --username AWS $ECR_URL

docker tag sagemaker-inference $ECR_URL/prithvi_inference:latest

docker push $ECR_URL/prithvi_inference:latest
