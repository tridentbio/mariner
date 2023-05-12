"""make experiment mlflow id nullable

Revision ID: 72a6c72e32b9
Revises: 40fadbde09b7
Create Date: 2023-05-08 20:20:14.666967

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '72a6c72e32b9'
down_revision = '40fadbde09b7'
branch_labels = None
depends_on = None


def upgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    op.alter_column('experiment', 'mlflow_id',
               existing_type=sa.VARCHAR(),
               nullable=True)
    # ### end Alembic commands ###


def downgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    op.alter_column('experiment', 'mlflow_id',
               existing_type=sa.VARCHAR(),
               nullable=False)
    # ### end Alembic commands ###