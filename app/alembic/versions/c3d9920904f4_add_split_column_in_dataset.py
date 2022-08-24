"""add split column in dataset

Revision ID: c3d9920904f4
Revises: fda7857a2aa2
Create Date: 2022-08-24 17:29:40.730959

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = 'c3d9920904f4'
down_revision = 'b0aeaff2380a'
branch_labels = None
depends_on = None


def upgrade():
    op.add_column('dataset', sa.Column('split_column', sa.String(), nullable=True))


def downgrade():
    op.drop_column('dataset', 'split_column')
