# Deployment Guide

## Overview

PetMind AI — Personalized Pet Care Advisor API is deployed as a containerized FastAPI service on AWS.

The current deployment architecture uses:

- **Amazon RDS for PostgreSQL** as the relational database
- **pgvector** inside PostgreSQL for vector similarity search
- **Amazon ECR** to store the Docker image
- **Amazon ECS with Fargate** to run the API container
- **Amazon CloudWatch Logs** for runtime logging and monitoring
- **IAM** for task execution permissions

This guide documents the minimum viable production deployment used for the course deliverable.

---

## Architecture Summary

### Runtime flow

1. The API container runs on **ECS Fargate**
2. The container image is pulled from **Amazon ECR**
3. The API connects to **Amazon RDS PostgreSQL**
4. PostgreSQL stores both structured data and vector embeddings
5. Application logs are written to **CloudWatch Logs**
6. The public health endpoint can be accessed through the task public IP

### Main AWS services used

- **RDS PostgreSQL**
- **ECR**
- **ECS Fargate**
- **CloudWatch Logs**
- **IAM**

---

## Local Build and Validation

Before deploying to AWS, the application is validated locally.

### Build and run locally

```bash id="m9rlzg"
docker compose up --build
```

### Validate the health endpoint locally

```bash id="m9rlzg"
curl http://127.0.0.1:8000/api/v1/health
```

### Run tests locally

```bash id="m9rlzg"
make test
```

### Run coverage locally

```bash id="m9rlzg"
make coverage
```

---

## Docker Image

The application is packaged using a multi-stage Dockerfile.

### Build image locally

```bash id="m9rlzg"
docker build -t petmind-ai-api:latest .
```

The final image exposes port 8000 and includes a health check against:

```bash id="m9rlzg"
/api/v1/health
```

---

## Database Deployment (Amazon RDS)

### Engine

- PostgreSQL

### Purpose

RDS stores:

- chunked knowledge documents
- metadata
- pgvector embeddings
- auxiliary relational data

### Notes

- The database is reachable from the ECS workload
- The vector extension is enabled
- Migrations are managed through Alembic

### Migration execution

After configuring the database connection string, migrations are applied with:

```bash id="m9rlzg"
alembic upgrade head
```

---

## Container Registry (Amazon ECR)

An ECR private repository is used to store the deployable image.

### Typical flow

- Create ECR repository
- Authenticate Docker against ECR
- Tag the image
- Push the image

### Example commands

```bash id="m9rlzg"
docker tag petmind-ai-api:latest <ECR_REPOSITORY_URI>:latest
docker push <ECR_REPOSITORY_URI>:latest
```

---

## Compute Deployment (Amazon ECS with Fargate)

The API is deployed as a single-container ECS task on Fargate.

### Cluster

- petmind-ai-cluster

### Service

- petmind-ai-service

### Task definition

- family: petmind-ai-api
- launch type: FARGATE
- runtime: Linux / x86_64
- container port: 8000

### Container configuration

- The task definition includes:
- environment variables for app configuration
- database connection string
- OpenAI settings
- API key
- logging configuration
- health check command

### Health check

The container health check validates:

```bash id="m9rlzg"
http://localhost:8000/api/v1/health
```

### Networking

The ECS service is configured with:

- awsvpc network mode
- public IP assignment enabled for MVP validation
- security group allowing inbound access to port 8000
- outbound access to reach RDS and external APIs

---

## Logging (CloudWatch)

The ECS task sends application logs to Amazon CloudWatch Logs.

### Purpose

CloudWatch is used for:

- startup and shutdown logs
- error debugging
- health check troubleshooting
- API runtime visibility

### Log group

The deployment uses a log group similar to:

```bash id="m9rlzg"
/ecs/petmind-ai-api
```

---

## IAM

The deployment uses an ECS task execution role to allow the runtime to:

- pull private images from ECR
- write logs to CloudWatch

### Role

- `ecsTaskExecutionRole`

### Required permissions

The execution role includes the AWS-managed policy for ECS task execution.

---

## Environment Variables

The application is configured through environment variables.

### Main categories

- API configuration
- OpenAI configuration
- PostgreSQL connection
- RAG chunking and retrieval thresholds
- logging
- feature flags

### Important note

For the MVP deployment, environment variables were configured directly in the task definition for simplicity.

A future hardening step should move secrets such as:

- OPENAI_API_KEY
- API_KEY
- DATABASE_URL

to a managed secret solution such as:

- AWS Secrets Manager
- AWS Systems Manager Parameter Store

---

## Deployment Validation

After deployment, the following validations were performed:

### Health endpoint

The public health endpoint responded successfully with:

- service status = healthy
- database status = healthy
- environment = production

### Database validation

The ECS task was able to connect to RDS successfully.

### Logging validation

Application logs were visible in CloudWatch.

### Functional validation

The API served the production endpoints:

- GET /api/v1/health
- POST /api/v1/query
- POST /api/v1/ingest

---

## Production URL

Add the current deployed public URL or public IP here:

```
http://<PUBLIC_IP_OR_BASE_URL>:8000
```

Example validation endpoint:

```
http://<PUBLIC_IP_OR_BASE_URL>:8000/api/v1/health
```

---

## Current Limitations

The current deployment is intentionally minimal and appropriate for a course MVP.

Known limitations:

- secrets are not yet managed through a dedicated secret store
- public IP exposure is used for validation simplicity
- no load balancer is configured yet
- no autoscaling is configured yet
- no private VPC-only service exposure is configured yet
- CD automation is not yet fully finalized

---

## Suggested Next Improvements

The natural next steps for production hardening are:

- move secrets to Secrets Manager or Parameter Store
- add a load balancer
- restrict network exposure
- automate deployment through GitHub Actions
- add autoscaling policies
- strengthen observability and metrics
- add secret rotation and tighter IAM policies
