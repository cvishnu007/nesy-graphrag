$packageRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$manifestPath = Join-Path $packageRoot "MANIFEST.json"

$files = Get-ChildItem -LiteralPath $packageRoot -Recurse -File |
    Where-Object { $_.FullName -ne $manifestPath } |
    Sort-Object FullName |
    ForEach-Object {
        [ordered]@{
            path = $_.FullName.Substring($packageRoot.Length + 1).Replace("\", "/")
            bytes = $_.Length
            sha256 = (Get-FileHash -LiteralPath $_.FullName -Algorithm SHA256).Hash.ToLowerInvariant()
        }
    }

$manifest = [ordered]@{
    manifest_version = "1.0"
    package = "teammate2"
    source_commit = "ddc7fe3"
    methodology = "AI-generated reference annotations; not human ground truth"
    file_count_excluding_manifest = @($files).Count
    files = @($files)
    explicit_exclusions = @(
        ".git and repository history internals",
        ".env and .env.example files",
        "credentials and API keys",
        "virtual environments and .tools",
        "Python, pytest, and model caches",
        "full corpus data including data/s2_clean.json",
        "Chroma and Neo4j database stores",
        "unrelated Teammate 1 and user files",
        "pre-existing corrupted .env.example",
        "pre-existing stray accidental file"
    )
}

$manifest | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath $manifestPath -Encoding utf8
Write-Output "manifest_files=$(@($files).Count)"
