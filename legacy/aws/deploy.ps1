# Deploy the current image to ECS (used by CI, runnable manually too).
#   .\scripts\deploy.ps1 -Stage staging -Tag <git-sha>
#   .\scripts\deploy.ps1 -Stage prod    -Tag <git-sha>
param(
    [Parameter(Mandatory)][ValidateSet('staging','prod')][string]$Stage,
    [Parameter(Mandatory)][string]$Tag,
    [string]$Cluster = 'dr-cluster',
    [string]$Region = 'us-east-1',
    [string]$Repo = 'dr-api'
)

$svc = "dr-api-$Stage"
Write-Host "==> image : $Repo:$Tag"
Write-Host "==> service: $svc (cluster $Cluster)"

# 1. task def with the new image tag
$td = Get-Content "configs/ecs-task-definition.json" -Raw |
      ConvertFrom-Json
$td.containerDefinitions[0].image = $td.containerDefinitions[0].image -replace ':.*$', ":$Tag"
$tmp = [System.IO.Path]::GetTempFileName() + ".json"
$td | ConvertTo-Json -Depth 20 | Set-Content $tmp

# 2. register + update (rollout is rolling; ECS keeps old tasks until healthy)
$newTd = aws ecs register-task-definition --cli-input-json (Get-Content $tmp -Raw) |
         ConvertFrom-Json
$rev = $newTd.taskDefinition.revision
aws ecs update-service --cluster $Cluster --service $svc `
    --task-definition "$($td.family):$rev" --force-new-deployment | Out-Null

aws ecs wait services-stable --cluster $Cluster --services $svc
Write-Host "==> $svc stable on revision $rev"

# 3. smoke test
$ctx = @{ staging = 'STAGING_URL'; prod = 'PROD_URL' }[$Stage]
$url = [Environment]::GetEnvironmentVariable($ctx)
if ($url) {
    Write-Host "==> smoke: $url/health"
    Invoke-RestMethod "$url/health" | ConvertTo-Json -Compress
    Invoke-RestMethod "$url/ready"  | ConvertTo-Json -Compress
} else {
    Write-Host "(set $ctx to enable smoke test)"
}
