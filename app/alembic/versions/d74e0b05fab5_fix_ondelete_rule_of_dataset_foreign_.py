"""fix ondelete rule of dataset foreign key in model table

Revision ID: d74e0b05fab5
Revises: 45d3f4178195
Create Date: 2022-07-22 02:47:36.604361

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = 'd74e0b05fab5'
down_revision = '45d3f4178195'
branch_labels = None
depends_on = None


def upgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    op.drop_constraint('model_dataset_id_fkey', 'model', type_='foreignkey')
    op.create_foreign_key(None, 'model', 'dataset', ['dataset_id'], ['id'], ondelete='SET NULL')
    # ### end Alembic commands ###


def downgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    op.drop_constraint(None, 'model', type_='foreignkey')
    op.create_foreign_key('model_dataset_id_fkey', 'model', 'dataset', ['dataset_id'], ['id'], ondelete='CASCADE')
    # ### end Alembic commands ###
