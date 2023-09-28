"""remove uniqueness of dataset name

Revision ID: 25191563d370
Revises: e3a60c031614
Create Date: 2023-09-25 20:33:15.609974

"""
from alembic import op

# revision identifiers, used by Alembic.
revision = "25191563d370"
down_revision = "e3a60c031614"
branch_labels = None
depends_on = None


def upgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    op.drop_index("ix_dataset_name", table_name="dataset")
    op.create_index(op.f("ix_dataset_name"), "dataset", ["name"], unique=False)
    # ### end Alembic commands ###


def downgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    op.drop_index(op.f("ix_dataset_name"), table_name="dataset")
    op.create_index("ix_dataset_name", "dataset", ["name"], unique=False)
    # ### end Alembic commands ###