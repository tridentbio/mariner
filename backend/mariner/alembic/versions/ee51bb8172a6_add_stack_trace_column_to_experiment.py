"""add stack_trace column to experiment

Revision ID: ee51bb8172a6
Revises: 4452555d97de
Create Date: 2022-08-29 19:35:55.171733

"""
import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision = "ee51bb8172a6"
down_revision = "4452555d97de"
branch_labels = None
depends_on = None


def upgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    op.add_column(
        "experiment", sa.Column("stack_trace", sa.String(), nullable=True)
    )
    # ### end Alembic commands ###


def downgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    op.drop_column("experiment", "stack_trace")
    # ### end Alembic commands ###
