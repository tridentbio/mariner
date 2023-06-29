"""Add column description to modelversion

Revision ID: a8fca2463072
Revises: 8b6bfa79b5e9
Create Date: 2022-12-06 13:24:15.136292

"""
import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision = "a8fca2463072"
down_revision = "8b6bfa79b5e9"
branch_labels = None
depends_on = None


def upgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    op.drop_table("deployment")
    op.add_column(
        "modelversion", sa.Column("description", sa.String(), nullable=True)
    )
    # ### end Alembic commands ###


def downgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    op.drop_column("modelversion", "description")
    op.create_table(
        "deployment",
        sa.Column("name", sa.VARCHAR(), autoincrement=False, nullable=False),
        sa.Column(
            "model_version_id",
            sa.INTEGER(),
            autoincrement=False,
            nullable=True,
        ),
        sa.Column(
            "created_by_id", sa.INTEGER(), autoincrement=False, nullable=True
        ),
        sa.ForeignKeyConstraint(
            ["created_by_id"],
            ["user.id"],
            name="deployment_created_by_id_fkey",
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["model_version_id"],
            ["modelversion.id"],
            name="deployment_model_version_id_fkey",
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("name", name="deployment_pkey"),
    )
    # ### end Alembic commands ###
