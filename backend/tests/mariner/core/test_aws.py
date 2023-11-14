from pathlib import Path

from mariner.core.aws import Bucket, download_head, is_in_s3, upload_s3_file
from mariner.utils import compress_file


def test_download_head(tmp_path: Path):
    # Test with uncompressed CSV
    import pandas as pd

    key = "datasets-test-download-head-final.csv"
    bucket = Bucket.Datasets
    if not is_in_s3(key, bucket):
        csv_path = tmp_path / "tmp.csv"
        df = pd.DataFrame({"A": list(range(20)), "B": list(range(20))}).to_csv(
            csv_path, index=False
        )
        with open(csv_path, "rb") as f:
            upload_s3_file(f, bucket, key)

    # Download first 10 lines
    data = download_head(bucket.value, key, 10, chunk_size=16)
    data.seek(0)
    df = pd.read_csv(data)
    assert isinstance(df, pd.DataFrame)

    # Test with compressed CSV
    key = f"datasets-test-download-head-compressed-final.csv"
    if not is_in_s3(key, bucket):
        csv_path = tmp_path / "tmp.csv"
        df = pd.DataFrame({"A": list(range(20)), "B": list(range(20))}).to_csv(
            csv_path, index=False
        )
        with open(csv_path, mode="rb+") as f:
            compressed_file = compress_file(f)
            upload_s3_file(compressed_file, bucket, key)
    # WONT WORK WITH CHUNK_SIZE SMALL
    data = download_head(bucket.value, key, nlines=10)
    data.seek(0)
    df = pd.read_csv(data)
    assert isinstance(df, pd.DataFrame)
