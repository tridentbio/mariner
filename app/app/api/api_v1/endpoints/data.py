from io import BufferedWriter
import boto3

from fastapi.param_functions import Depends, Query
from fastapi.routing import APIRouter
from starlette.responses import StreamingResponse

from app import schemas
from app.api import deps
from app.core.config import settings


router = APIRouter()


def iters3file(filename: str, bucket_name: str, object_key: str):
    s3 = boto3.client(
       "s3",
       region_name=settings.AWS_REGION,
       aws_access_key_id=settings.AWS_SECRET_KEY_ID,
       aws_secret_access_key=settings.AWS_SECRET_KEY,
    )
    with open(filename, 'wb') as f:
        s3.download_fileobj(bucket_name, object_key, f)
        yield from f

@router.get("/data", response_class=StreamingResponse, dependencies=[Depends(deps.get_current_active_superuser)])
def get_s3_data(
    bucket_name: str = Query(..., alias="bucketName"),
    object_key: str = Query(..., alias="objectKey"),
) -> StreamingResponse:
    """
    Proxy to AWS bucket resource after applying access control
    """
    return  StreamingResponse(iters3file('testy', bucket_name, object_key), media_type="text/csv")
