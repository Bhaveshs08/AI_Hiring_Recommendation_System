# === Project Keys (replace with your real ones) ===
OPENAI_API_KEY = "YOUR_OPENAI_KEY_HERE"
$env:PINECONE_API_KEY = "YOUR_PINECONE_KEY_HERE"
$env:PINECONE_ENV = "us-east-1"
$env:PINECONE_INDEX = "polaris"
$env:RESUME_NAMESPACE = "Resumes"
$env:JD_NAMESPACE = "Job_Descriptions"
$env:EMBED_MODEL = "text-embedding-3-large"
$env:LLM_MODEL = "gpt-4o-mini"
Write-Host "Polaris environment variables set for this session." -ForegroundColor Green
# Ensure venv env vars & prompt are set
$venv = Join-Path $PWD 'venv'
if (Test-Path $venv) {
    $env:VIRTUAL_ENV = $venv
    $env:PATH = "$venv\Scripts;$env:PATH"
    function global:prompt { if ($env:VIRTUAL_ENV) { "(`$(Split-Path $env:VIRTUAL_ENV -Leaf)`) " + (Get-Location) + "> " } else { (Get-Location) + "> " } }
}
