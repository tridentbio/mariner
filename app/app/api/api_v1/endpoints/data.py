import boto3
from fastapi.param_functions import Depends, Query
from fastapi.routing import APIRouter
from starlette.responses import StreamingResponse

from app.api import deps
from app.core.config import settings

router = APIRouter()


@router.get(
    "/data",
    response_class=StreamingResponse,
    dependencies=[Depends(deps.get_current_active_user)],
)
def get_s3_data(
    object_key: str = Query(..., alias="objectKey"),
) -> StreamingResponse:
    """
    Proxy to AWS bucket resource after applying access control
    """
    s3 = boto3.client(
        "s3",
        region_name=settings.AWS_REGION,
        aws_access_key_id=settings.AWS_SECRET_KEY_ID,
        aws_secret_access_key=settings.AWS_SECRET_KEY,
    )
    s3_res = s3.get_object(Bucket=settings.AWS_DATASETS, Key=object_key)
    return StreamingResponse(s3_res["Body"].iter_chunks(), media_type="text/csv")
