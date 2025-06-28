# AWS VPC Setup Guide for AI Trading Bot

## Overview

A VPC (Virtual Private Cloud) provides network isolation and enhanced security for your trading bot. This guide walks you through creating a production-ready VPC architecture.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                         VPC (10.0.0.0/16)                   │
├─────────────────────────────┬───────────────────────────────┤
│      Public Subnet          │      Private Subnet           │
│      (10.0.1.0/24)         │      (10.0.2.0/24)          │
│                            │                               │
│   ┌─────────────────┐     │   ┌──────────────────┐       │
│   │   NAT Gateway   │     │   │   Trading Bot    │       │
│   │                 │     │   │   EC2 Instance   │       │
│   └─────────────────┘     │   └──────────────────┘       │
│            │               │            │                  │
│   ┌─────────────────┐     │            │                  │
│   │ Internet Gateway│◄────┼────────────┘                  │
│   └─────────────────┘     │                               │
└────────────┴───────────────┴───────────────────────────────┘
```

## Method 1: AWS Console (Recommended for Beginners)

### Step 1: Create VPC

1. Go to **VPC Dashboard** in AWS Console
2. Click **"Create VPC"**
3. Configure:
   - **Name tag**: `ai-trader-vpc`
   - **IPv4 CIDR**: `10.0.0.0/16`
   - **IPv6 CIDR block**: No IPv6 CIDR block
   - **Tenancy**: Default
4. Click **"Create VPC"**

### Step 2: Create Subnets

#### Public Subnet (for NAT Gateway)
1. Go to **Subnets** → **"Create subnet"**
2. Configure:
   - **VPC**: Select `ai-trader-vpc`
   - **Subnet name**: `ai-trader-public-subnet`
   - **Availability Zone**: Choose your preferred AZ (e.g., `us-east-1a`)
   - **IPv4 CIDR block**: `10.0.1.0/24`
3. Click **"Create subnet"**

#### Private Subnet (for Trading Bot)
1. Click **"Create subnet"** again
2. Configure:
   - **VPC**: Select `ai-trader-vpc`
   - **Subnet name**: `ai-trader-private-subnet`
   - **Availability Zone**: Same as public subnet
   - **IPv4 CIDR block**: `10.0.2.0/24`
3. Click **"Create subnet"**

### Step 3: Create Internet Gateway

1. Go to **Internet Gateways** → **"Create internet gateway"**
2. **Name tag**: `ai-trader-igw`
3. Click **"Create internet gateway"**
4. Select the new IGW → **Actions** → **"Attach to VPC"**
5. Select `ai-trader-vpc` and attach

### Step 4: Create NAT Gateway

1. Go to **NAT Gateways** → **"Create NAT gateway"**
2. Configure:
   - **Name**: `ai-trader-nat`
   - **Subnet**: Select `ai-trader-public-subnet`
   - **Elastic IP allocation ID**: Click **"Allocate Elastic IP"**
3. Click **"Create NAT gateway"**

### Step 5: Configure Route Tables

#### Public Route Table
1. Go to **Route Tables**
2. Find the route table associated with your VPC
3. **Name it**: `ai-trader-public-rt`
4. Select it → **Routes** tab → **"Edit routes"**
5. Add route:
   - **Destination**: `0.0.0.0/0`
   - **Target**: Select Internet Gateway (`ai-trader-igw`)
6. **Subnet associations** tab → **"Edit subnet associations"**
7. Select `ai-trader-public-subnet`

#### Private Route Table
1. **"Create route table"**
2. Configure:
   - **Name tag**: `ai-trader-private-rt`
   - **VPC**: `ai-trader-vpc`
3. Select it → **Routes** tab → **"Edit routes"**
4. Add route:
   - **Destination**: `0.0.0.0/0`
   - **Target**: Select NAT Gateway (`ai-trader-nat`)
5. **Subnet associations** tab → **"Edit subnet associations"**
6. Select `ai-trader-private-subnet`

### Step 6: Create Security Groups

#### Trading Bot Security Group
1. Go to **Security Groups** → **"Create security group"**
2. Configure:
   - **Security group name**: `ai-trader-sg`
   - **Description**: `Security group for AI trading bot`
   - **VPC**: `ai-trader-vpc`
3. **Inbound rules**:
   - **SSH**: Port 22, Source: Your IP only
   - **HTTPS**: Port 443, Source: 0.0.0.0/0 (for API calls)
4. **Outbound rules**:
   - Leave default (all traffic allowed)

## Method 2: AWS CLI (Automated)

Create a file `create-vpc.sh`:

```bash
#!/bin/bash

# Variables
REGION="us-east-1"
VPC_CIDR="10.0.0.0/16"
PUBLIC_SUBNET_CIDR="10.0.1.0/24"
PRIVATE_SUBNET_CIDR="10.0.2.0/24"
AVAILABILITY_ZONE="${REGION}a"

echo "🌐 Creating VPC for AI Trading Bot..."

# Create VPC
VPC_ID=$(aws ec2 create-vpc \
    --cidr-block $VPC_CIDR \
    --tag-specifications "ResourceType=vpc,Tags=[{Key=Name,Value=ai-trader-vpc}]" \
    --query 'Vpc.VpcId' \
    --output text)

echo "✅ Created VPC: $VPC_ID"

# Enable DNS hostnames
aws ec2 modify-vpc-attribute \
    --vpc-id $VPC_ID \
    --enable-dns-hostnames

# Create public subnet
PUBLIC_SUBNET_ID=$(aws ec2 create-subnet \
    --vpc-id $VPC_ID \
    --cidr-block $PUBLIC_SUBNET_CIDR \
    --availability-zone $AVAILABILITY_ZONE \
    --tag-specifications "ResourceType=subnet,Tags=[{Key=Name,Value=ai-trader-public-subnet}]" \
    --query 'Subnet.SubnetId' \
    --output text)

echo "✅ Created public subnet: $PUBLIC_SUBNET_ID"

# Create private subnet
PRIVATE_SUBNET_ID=$(aws ec2 create-subnet \
    --vpc-id $VPC_ID \
    --cidr-block $PRIVATE_SUBNET_CIDR \
    --availability-zone $AVAILABILITY_ZONE \
    --tag-specifications "ResourceType=subnet,Tags=[{Key=Name,Value=ai-trader-private-subnet}]" \
    --query 'Subnet.SubnetId' \
    --output text)

echo "✅ Created private subnet: $PRIVATE_SUBNET_ID"

# Create Internet Gateway
IGW_ID=$(aws ec2 create-internet-gateway \
    --tag-specifications "ResourceType=internet-gateway,Tags=[{Key=Name,Value=ai-trader-igw}]" \
    --query 'InternetGateway.InternetGatewayId' \
    --output text)

# Attach IGW to VPC
aws ec2 attach-internet-gateway \
    --internet-gateway-id $IGW_ID \
    --vpc-id $VPC_ID

echo "✅ Created and attached Internet Gateway: $IGW_ID"

# Allocate Elastic IP for NAT Gateway
EIP_ALLOC_ID=$(aws ec2 allocate-address \
    --domain vpc \
    --tag-specifications "ResourceType=elastic-ip,Tags=[{Key=Name,Value=ai-trader-nat-eip}]" \
    --query 'AllocationId' \
    --output text)

# Create NAT Gateway
NAT_GW_ID=$(aws ec2 create-nat-gateway \
    --subnet-id $PUBLIC_SUBNET_ID \
    --allocation-id $EIP_ALLOC_ID \
    --tag-specifications "ResourceType=nat-gateway,Tags=[{Key=Name,Value=ai-trader-nat}]" \
    --query 'NatGateway.NatGatewayId' \
    --output text)

echo "✅ Created NAT Gateway: $NAT_GW_ID (this may take a few minutes to become available)"

# Wait for NAT Gateway to be available
echo "⏳ Waiting for NAT Gateway to become available..."
aws ec2 wait nat-gateway-available --nat-gateway-ids $NAT_GW_ID

# Create route table for public subnet
PUBLIC_RT_ID=$(aws ec2 create-route-table \
    --vpc-id $VPC_ID \
    --tag-specifications "ResourceType=route-table,Tags=[{Key=Name,Value=ai-trader-public-rt}]" \
    --query 'RouteTable.RouteTableId' \
    --output text)

# Add route to Internet Gateway
aws ec2 create-route \
    --route-table-id $PUBLIC_RT_ID \
    --destination-cidr-block 0.0.0.0/0 \
    --gateway-id $IGW_ID

# Associate public subnet with public route table
aws ec2 associate-route-table \
    --subnet-id $PUBLIC_SUBNET_ID \
    --route-table-id $PUBLIC_RT_ID

echo "✅ Created public route table: $PUBLIC_RT_ID"

# Create route table for private subnet
PRIVATE_RT_ID=$(aws ec2 create-route-table \
    --vpc-id $VPC_ID \
    --tag-specifications "ResourceType=route-table,Tags=[{Key=Name,Value=ai-trader-private-rt}]" \
    --query 'RouteTable.RouteTableId' \
    --output text)

# Add route to NAT Gateway
aws ec2 create-route \
    --route-table-id $PRIVATE_RT_ID \
    --destination-cidr-block 0.0.0.0/0 \
    --nat-gateway-id $NAT_GW_ID

# Associate private subnet with private route table
aws ec2 associate-route-table \
    --subnet-id $PRIVATE_SUBNET_ID \
    --route-table-id $PRIVATE_RT_ID

echo "✅ Created private route table: $PRIVATE_RT_ID"

# Create security group
SG_ID=$(aws ec2 create-security-group \
    --group-name ai-trader-sg \
    --description "Security group for AI trading bot" \
    --vpc-id $VPC_ID \
    --tag-specifications "ResourceType=security-group,Tags=[{Key=Name,Value=ai-trader-sg}]" \
    --query 'GroupId' \
    --output text)

# Add SSH rule (replace with your IP)
MY_IP=$(curl -s https://checkip.amazonaws.com)
aws ec2 authorize-security-group-ingress \
    --group-id $SG_ID \
    --protocol tcp \
    --port 22 \
    --cidr ${MY_IP}/32

# Add HTTPS outbound (for API calls)
aws ec2 authorize-security-group-egress \
    --group-id $SG_ID \
    --protocol tcp \
    --port 443 \
    --cidr 0.0.0.0/0

echo "✅ Created security group: $SG_ID"

# Save configuration
cat > vpc-config.json << EOF
{
    "vpc_id": "$VPC_ID",
    "public_subnet_id": "$PUBLIC_SUBNET_ID",
    "private_subnet_id": "$PRIVATE_SUBNET_ID",
    "internet_gateway_id": "$IGW_ID",
    "nat_gateway_id": "$NAT_GW_ID",
    "security_group_id": "$SG_ID",
    "region": "$REGION",
    "availability_zone": "$AVAILABILITY_ZONE"
}
EOF

echo "
✅ VPC setup complete!

Configuration saved to vpc-config.json

Next steps:
1. Launch your EC2 instance in the private subnet
2. Use the security group: $SG_ID
3. The instance will have internet access via NAT Gateway

To launch an instance in this VPC:
aws ec2 run-instances \\
    --image-id ami-0c02fb55956c7d316 \\
    --instance-type t3.medium \\
    --key-name your-key-pair \\
    --subnet-id $PRIVATE_SUBNET_ID \\
    --security-group-ids $SG_ID \\
    --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=ai-trader-prod}]'
"
```

## Best Practices for Trading Bot VPC

### 1. Security Considerations

- **Private Subnet**: Always deploy the trading bot in the private subnet
- **Minimal Access**: Only open required ports (SSH for management)
- **IP Whitelisting**: Restrict SSH access to your specific IP
- **VPC Flow Logs**: Enable for security monitoring

### 2. High Availability (Optional)

For production, consider multi-AZ deployment:

```bash
# Create additional subnets in different AZ
aws ec2 create-subnet \
    --vpc-id $VPC_ID \
    --cidr-block 10.0.3.0/24 \
    --availability-zone ${REGION}b \
    --tag-specifications "ResourceType=subnet,Tags=[{Key=Name,Value=ai-trader-private-subnet-2}]"
```

### 3. Cost Optimization

- **NAT Gateway**: ~$45/month - consider NAT instance for dev/test
- **Elastic IP**: Free when attached, $3.60/month if not used
- **Data Transfer**: Minimize cross-AZ traffic

### 4. Monitoring

Enable VPC Flow Logs:

```bash
# Create CloudWatch Logs group
aws logs create-log-group --log-group-name /aws/vpc/flowlogs

# Create IAM role for Flow Logs (create flow-logs-trust-policy.json first)
aws iam create-role \
    --role-name flowlogsRole \
    --assume-role-policy-document file://flow-logs-trust-policy.json

# Enable Flow Logs
aws ec2 create-flow-logs \
    --resource-type VPC \
    --resource-ids $VPC_ID \
    --traffic-type ALL \
    --log-destination-type cloud-watch-logs \
    --log-group-name /aws/vpc/flowlogs \
    --deliver-logs-permission-arn arn:aws:iam::YOUR-ACCOUNT-ID:role/flowlogsRole
```

## Connecting to Instances in Private Subnet

Since your trading bot will be in a private subnet, you have several options:

### Option 1: Bastion Host (Jump Box)
```bash
# Launch a small instance in public subnet as bastion
# Then SSH through it:
ssh -J ec2-user@bastion-ip ubuntu@private-instance-ip
```

### Option 2: AWS Systems Manager Session Manager (Recommended)
```bash
# No need for SSH keys or bastion hosts
aws ssm start-session --target instance-id
```

### Option 3: VPN
Set up AWS Client VPN for secure direct access to private resources.

## Cleanup (If Needed)

To delete all VPC resources:

```bash
# Delete NAT Gateway first (to stop charges)
aws ec2 delete-nat-gateway --nat-gateway-id $NAT_GW_ID

# Release Elastic IP
aws ec2 release-address --allocation-id $EIP_ALLOC_ID

# Delete subnets, route tables, etc.
# ... (full cleanup script available)
```

## Integration with Trading Bot

Update your EC2 launch command to use the VPC:

```bash
aws ec2 run-instances \
    --image-id ami-0c02fb55956c7d316 \
    --instance-type t3.medium \
    --key-name your-key-pair \
    --subnet-id $PRIVATE_SUBNET_ID \
    --security-group-ids $SG_ID \
    --iam-instance-profile Name=ai-trader-profile \
    --user-data file://user-data.sh \
    --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=ai-trader-prod}]'
```

## Summary

Your VPC provides:
- **Network Isolation**: Complete control over your network
- **Enhanced Security**: Private subnets keep trading bot isolated
- **Scalability**: Easy to add more instances or services
- **Compliance**: Meets security best practices

The trading bot in the private subnet can:
- ✅ Make outbound API calls to Binance
- ✅ Send logs to CloudWatch
- ✅ Access AWS services
- ❌ Cannot be directly accessed from internet (more secure)

This setup ensures your trading bot operates in a secure, isolated environment while maintaining the connectivity it needs for trading operations. 