"""change_parameters_column_to_bigint

Revision ID: 38ecc7cdf47a
Revises: a23abb777e0f
Create Date: 2025-04-28 17:30:13.340272

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = '38ecc7cdf47a'
down_revision: Union[str, None] = 'a23abb777e0f'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.alter_column('models', 'parameters',
                   existing_type=sa.Integer(), 
                   type_=sa.BigInteger(),
                   existing_nullable=True)  


def downgrade() -> None:
    """Downgrade schema."""
    op.alter_column('models', 'parameters',
                   existing_type=sa.BigInteger(),
                   type_=sa.Integer(), 
                   existing_nullable=True)  