import datetime
import pandas
import io
from uuid import uuid4
from fastapi.datastructures import UploadFile
from sqlalchemy.orm.session import Session

from ..user.schema import User
from .schema import DatasetsQuery, DatasetCreateRepo, DatasetCreate
from .crud import repo

# TODO: move to somewhere appropriate
DATASET_BUCKET = 'datasets-bucket'

def make_key():
    return str(uuid4())

def get_my_datasets(db: Session, current_user: User, query: DatasetsQuery):
    if not current_user.id:
        raise Exception('wtf')
    query.created_by_id = current_user.id
    datasets, total = repo.get_many_paginated(db, query)
    return datasets, total

def create_dataset(db: Session, current_user: User, file: UploadFile, data: DatasetCreate):

    # parse csv bytes as json
    file_raw = file.file.read()
    df = pandas.read_csv(io.BytesIO(file_raw))
    stats = df.describe(include='all').to_dict()

    key = make_key()
    # Upload to s3 bucket
    # s3 = boto3.client('s3')
    # TODO: Here we should encrypt if bucket is not encrypted by AWS already
    # s3.upload_fileobj(file, DATASET_BUCKET, key)
    dataset = repo.create(db, DatasetCreateRepo(
        columns=len(df.columns),
        rows=len(df),
        name=data.name,
        description=data.description,
        bytes=len(file_raw), # maybe should be the encrypted size instead,
        created_at=datetime.datetime.now(),
        stats=stats,
        data_url=key,
        created_by_id=current_user.id
    ))
    return dataset

