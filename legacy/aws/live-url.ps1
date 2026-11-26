# live-url.ps1 - print the current live URL without opening the AWS console.
# Needs a one-time setup:  winget install Amazon.AWSCLI  then  aws configure
# (region from your account, e.g. ap-southeast-2)
param(
    [string]$Region = "ap-southeast-2",   # where your dr-api instance lives
    [string]$Name   = "dr-api"
)

$aws = Get-Command aws -ErrorAction SilentlyContinue
if (-not $aws) {
    Write-Host "AWS CLI not installed. One-time setup:" -ForegroundColor Yellow
    Write-Host "  winget install Amazon.AWSCLI"
    Write-Host "  aws configure        # paste your access key + secret + region"
    Write-Host "Then re-run this script."
    exit 1
}

$txt = aws ec2 describe-instances --region $Region `
    --filters "Name=tag:Name,Values=$Name" "Name=instance-state-name,Values=running" `
    --query "Reservations[].Instances[].PublicIpAddress" --output text 2>$null

if (-not $txt -or $txt.ToUpper() -eq "NONE") {
    Write-Host "No RUNNING instance tagged '$Name' in region $Region."
    Write-Host "If you stopped it: EC2 console -> Start instance, then re-run."
    exit 1
}

$ip = $txt.Trim().Split()[0]
Write-Host "PUBLIC IP : $ip" -ForegroundColor Green
Write-Host "API       : http://$ip/"
Write-Host "Swagger   : http://$ip/docs   <- demo URL to open"
