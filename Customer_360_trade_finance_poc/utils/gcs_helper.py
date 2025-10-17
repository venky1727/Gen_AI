from google.cloud import storage
import pandas as pd
import tempfile
 
def download_blob_as_df(gcs_path: str) -> pd.DataFrame:
    assert gcs_path.startswith("gs://")
    bucket_name, blob_path = gcs_path[5:].split("/", 1)
 
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_path)
 
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        blob.download_to_filename(temp_file.name)
        df = pd.read_csv(temp_file.name)
 
    return df
 