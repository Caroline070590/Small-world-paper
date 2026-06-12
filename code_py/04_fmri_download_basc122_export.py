#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Auto-converted from the Jupyter notebook ``Preprocessing-ABIDE-final.ipynb``.

Code cells are reproduced verbatim. Markdown cells are kept as comments.
Jupyter shell-magic lines (if any) are commented out so the file is valid
Python; they are preserved for reference. Cell boundaries are marked with
``# In[...]`` to match the original notebook ordering.
"""


# In[ ]:

import os

os.environ["AWS_REGION"] = "eu-central-1"          # or your region
os.environ["S3_BUCKET"] = "YOUR_BUCKET_NAME"       # <-- required
os.environ["COBRE_PREFIX"] = "COBRE/"              # adjust if different
os.environ["BUREAU_PREFIX"] = "ADHD_BUREAU/"       # adjust if different
os.environ["LOCAL_DATA_DIR"] = "DATA"
os.environ["EXPORT_SCRIPT"] = "export_basc_timeseries.py"
# os.environ["S3_REQUEST_PAYER"] = "requester"     # only if needed
#!/usr/bin/env python3
"""
Download COBRE + ADHD_BUREAU from AWS S3 using boto3, then run the main BASC export script.

Requirements:
  pip install boto3 botocore

Auth:
  - AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY (+ AWS_SESSION_TOKEN if needed), OR
  - AWS profile/SSO, OR
  - EC2/ECS/IAM role

Env vars you set (examples):
  export AWS_REGION=eu-central-1
  export S3_BUCKET=my-neuro-datasets
  export COBRE_PREFIX=COBRE/
  export BUREAU_PREFIX=ADHD_BUREAU/
  export LOCAL_DATA_DIR=DATA

Optional:
  export S3_REQUEST_PAYER=requester   # if bucket is "Requester Pays"
"""

import os
import sys
import hashlib
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import boto3
from botocore.exceptions import ClientError


def _safe_mkdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _should_download(key: str) -> bool:
    # Keep only what your pipeline needs: NIfTI + CSV + JSON (+ TSV optionally)
    k = key.lower()
    return (
        k.endswith(".nii")
        or k.endswith(".nii.gz")
        or k.endswith(".csv")
        or k.endswith(".json")
        or k.endswith(".tsv")
    )


def s3_download_prefix(
    *,
    bucket: str,
    prefix: str,
    out_dir: Path,
    max_workers: int = 12,
    skip_existing: bool = True,
    verify_hash_if_metadata_present: bool = False,
    request_payer: str | None = None,
) -> None:
    """
    Download all objects under s3://bucket/prefix into out_dir, preserving relative paths.

    Notes on verification:
      - S3 ETag is not reliable for multipart uploads (common for big NIfTI).
      - If your objects have metadata like x-amz-meta-sha256, you can enable verify_hash_if_metadata_present.
    """
    _safe_mkdir(out_dir)
    s3 = boto3.client("s3")

    paginator = s3.get_paginator("list_objects_v2")
    list_kwargs = dict(Bucket=bucket, Prefix=prefix)
    if request_payer:
        list_kwargs["RequestPayer"] = request_payer

    keys: list[str] = []
    for page in paginator.paginate(**list_kwargs):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if key.endswith("/"):
                continue
            if _should_download(key):
                keys.append(key)

    if not keys:
        print(f"[S3] No matching files under s3://{bucket}/{prefix}")
        return

    print(f"[S3] Found {len(keys)} files under s3://{bucket}/{prefix}")
    print(f"[S3] Download -> {out_dir.resolve()}")

    def _download_one(key: str) -> tuple[str, str]:
        rel = key[len(prefix):].lstrip("/") if key.startswith(prefix) else key
        local_path = out_dir / rel
        _safe_mkdir(local_path.parent)

        if skip_existing and local_path.exists() and local_path.stat().st_size > 0:
            return (key, "skipped")

        get_kwargs = dict(Bucket=bucket, Key=key)
        if request_payer:
            get_kwargs["RequestPayer"] = request_payer

        # If hash metadata is available, optionally verify after download
        expected_sha256 = None
        if verify_hash_if_metadata_present:
            try:
                head = s3.head_object(**get_kwargs)
                md = head.get("Metadata", {}) or {}
                expected_sha256 = md.get("sha256")  # expects x-amz-meta-sha256
            except ClientError:
                expected_sha256 = None

        # Download
        s3.download_file(bucket, key, str(local_path), ExtraArgs={"RequestPayer": request_payer} if request_payer else None)

        if expected_sha256:
            got = _sha256(local_path)
            if got != expected_sha256:
                raise RuntimeError(f"SHA256 mismatch for {local_path}: expected={expected_sha256} got={got}")

        return (key, "downloaded")

    n_dl = 0
    n_skip = 0
    failures: list[tuple[str, str]] = []

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = {ex.submit(_download_one, k): k for k in keys}
        for fut in as_completed(futs):
            key = futs[fut]
            try:
                _, status = fut.result()
                if status == "downloaded":
                    n_dl += 1
                else:
                    n_skip += 1
            except Exception as e:
                failures.append((key, f"{type(e).__name__}: {e}"))

    print(f"[S3] Done. downloaded={n_dl} skipped={n_skip} failed={len(failures)}")
    if failures:
        print("[S3] Failures (first 20):")
        for k, msg in failures[:20]:
            print(f"  - {k}: {msg}")
        raise SystemExit(2)


def main() -> None:
    bucket = os.environ.get("S3_BUCKET", "").strip()
    if not bucket:
        print("ERROR: set S3_BUCKET")
        sys.exit(2)

    cobre_prefix = os.environ.get("COBRE_PREFIX", "COBRE/").strip()
    bureau_prefix = os.environ.get("BUREAU_PREFIX", "ADHD_BUREAU/").strip()
    region = os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION")
    if region:
        os.environ["AWS_DEFAULT_REGION"] = region  # boto3 fallback

    local_data_dir = Path(os.environ.get("LOCAL_DATA_DIR", "DATA")).resolve()
    request_payer = os.environ.get("S3_REQUEST_PAYER")  # set to "requester" if needed

    # Download datasets
    s3_download_prefix(
        bucket=bucket,
        prefix=cobre_prefix,
        out_dir=local_data_dir / "COBRE",
        max_workers=12,
        skip_existing=True,
        verify_hash_if_metadata_present=False,
        request_payer=request_payer,
    )

    s3_download_prefix(
        bucket=bucket,
        prefix=bureau_prefix,
        out_dir=local_data_dir / "ADHD_BUREAU",
        max_workers=12,
        skip_existing=True,
        verify_hash_if_metadata_present=False,
        request_payer=request_payer,
    )

    # Now run your actual BASC export script (the one that starts with the future import).
    # Set PROJECT_ROOT so your script can discover datasets under it.
    export_script = os.environ.get("EXPORT_SCRIPT", "export_basc_timeseries.py")
    print(f"[RUN] python {export_script}")
    os.execvp(sys.executable, [sys.executable, export_script])


if __name__ == "__main__":
    main()
