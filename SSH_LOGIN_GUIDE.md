# SSH Login Guide for AWS EC2 Instance

This guide will help you connect to the AWS EC2 GPU instance for the Shawshank project.

## Prerequisites

Before connecting, you'll need:
1. **SSH Key File**: `229project.pem` (should be in Downloads folder or provided separately)
2. **Instance Public IP**: Get this from AWS Console (EC2 > Instances > vllm-gpu-instance)
3. **Terminal/Command Line**: macOS Terminal, Windows PowerShell, or Linux terminal

## Step 1: Locate Your Key File

The SSH key file should be named `229project.pem`. Check if it's in your Downloads folder:

```bash
# Check if key file exists in Downloads
ls ~/Downloads/229project.pem
```

If the file is in a different location, note the full path.

## Step 2: Set Correct Permissions

SSH requires the key file to have restricted permissions for security. Set them:

```bash
# If key is in Downloads
chmod 400 ~/Downloads/229project.pem

# Or if you moved it to ~/.ssh/
chmod 400 ~/.ssh/229project.pem
```

**Important**: Without correct permissions, SSH will refuse to use the key.

## Step 3: Get the Instance IP Address

1. Go to [AWS EC2 Console](https://console.aws.amazon.com/ec2/)
2. Click on "Instances" in the left sidebar
3. Find the instance named `vllm-gpu-instance`
4. Copy the **Public IPv4 address** (e.g., `18.236.65.13`)

**Note**: The IP address may change if the instance is stopped and restarted. Always check the console for the current IP.

## Step 4: Connect via SSH

### Option A: Key File in Downloads (Easiest)

```bash
ssh -i ~/Downloads/229project.pem ec2-user@YOUR_INSTANCE_IP
```

Replace `YOUR_INSTANCE_IP` with the actual IP address from Step 3.

**Example**:
```bash
ssh -i ~/Downloads/229project.pem ec2-user@18.236.65.13
```

### Option B: Key File in ~/.ssh/ (Standard Location)

If you've moved the key to the standard location:

```bash
ssh -i ~/.ssh/229project.pem ec2-user@YOUR_INSTANCE_IP
```

## Step 5: First-Time Connection

On your first connection, you'll see a security warning:

```
The authenticity of host '18.236.65.13 (18.236.65.13)' can't be established.
ED25519 key fingerprint is SHA256:...
Are you sure you want to continue connecting (yes/no/[fingerprint])?
```

**Answer `yes`** and press Enter. This is normal for first-time connections to a new server.

## Step 6: Verify You're Connected

Once connected, you should see:

```
[ec2-user@ip-172-31-27-155 ~]$
```

You're now logged into the AWS instance!

## Quick Reference

**Connection Command** (replace IP with current instance IP):
```bash
ssh -i ~/Downloads/229project.pem ec2-user@18.236.65.13
```

**Key Details**:
- **Key file**: `229project.pem`
- **User**: `ec2-user` (for Amazon Linux 2023)
- **Region**: `us-west-2`
- **Instance**: `vllm-gpu-instance` (g5.xlarge)

## Troubleshooting

### Issue: "Permission denied (publickey)"

**Solutions**:
1. Check key file permissions: `chmod 400 ~/Downloads/229project.pem`
2. Verify you're using the correct key file path
3. Make sure you're using the correct username: `ec2-user` (not `ubuntu`)

### Issue: "Identity file not accessible: No such file or directory"

**Solutions**:
1. Check if the key file exists: `ls ~/Downloads/229project.pem`
2. Use the full path: `/Users/YOUR_USERNAME/Downloads/229project.pem`
3. Make sure the file name is exactly `229project.pem` (case-sensitive)

### Issue: "Connection timed out"

**Solutions**:
1. Check if the instance is running in AWS Console
2. Verify the security group allows SSH (port 22) from your IP
3. Check if the IP address has changed (get new IP from console)
4. Verify you're using the correct public IP (not private IP)

### Issue: "Host key verification failed"

**Solution**: This happens if you answered "no" to the host key prompt. Just run the SSH command again and answer "yes" when prompted.

### Issue: "WARNING: UNPROTECTED PRIVATE KEY FILE!"

**Solution**: Set correct permissions:
```bash
chmod 400 ~/Downloads/229project.pem
```

## Disconnecting

To disconnect from the instance, simply type:
```bash
exit
```

Or press `Ctrl+D`

## Next Steps After Connecting

Once you're connected, you can:

1. **Verify GPU is available**:
   ```bash
   nvidia-smi
   ```

2. **Check Python version**:
   ```bash
   python3 --version
   ```

3. **Navigate to project directory** (if already cloned):
   ```bash
   cd ~/shawshank
   ```

4. **Clone the repository** (if not already there):
   ```bash
   git clone https://github.com/esi2tekky/shawshank.git
   cd shawshank
   ```

## Security Notes

- **Never share your private key file** (`229project.pem`) - it's like a password
- **Keep the key file secure** - don't commit it to git or share it publicly
- The key file gives full access to the instance - treat it carefully
- If the key is compromised, you can create a new key pair in AWS and update the instance

## Getting Help

If you encounter issues:
1. Check the troubleshooting section above
2. Verify the instance is running in AWS Console
3. Check that your IP address hasn't changed
4. Make sure you have the latest key file

---

**Last Updated**: Based on instance `vllm-gpu-instance` in region `us-west-2`

