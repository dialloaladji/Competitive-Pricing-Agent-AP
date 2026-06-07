"""add electrical specs columns to products

Revision ID: 001_electrical_specs
Revises:
Create Date: 2026-06-05 01:35:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = '001_electrical_specs'
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column('products', sa.Column('product_type', sa.String(length=100), nullable=True))
    op.add_column('products', sa.Column('voltage_v', sa.Integer(), nullable=True))
    op.add_column('products', sa.Column('current_a', sa.Integer(), nullable=True))
    op.add_column('products', sa.Column('poles', sa.Integer(), nullable=True))
    op.add_column('products', sa.Column('curve', sa.String(length=10), nullable=True))
    op.add_column('products', sa.Column('breaking_capacity_ka', sa.Float(), nullable=True))
    op.add_column('products', sa.Column('phase', sa.String(length=20), nullable=True))
    op.add_column('products', sa.Column('power_w', sa.Float(), nullable=True))
    op.add_column('products', sa.Column('mounting', sa.String(length=50), nullable=True))
    op.add_column('products', sa.Column('standard', sa.String(length=50), nullable=True))
    op.add_column('products', sa.Column('usage', sa.String(length=50), nullable=True))
    op.alter_column('products', 'currency', server_default='EUR')


def downgrade() -> None:
    op.alter_column('products', 'currency', server_default='USD')
    op.drop_column('products', 'usage')
    op.drop_column('products', 'standard')
    op.drop_column('products', 'mounting')
    op.drop_column('products', 'power_w')
    op.drop_column('products', 'phase')
    op.drop_column('products', 'breaking_capacity_ka')
    op.drop_column('products', 'curve')
    op.drop_column('products', 'poles')
    op.drop_column('products', 'current_a')
    op.drop_column('products', 'voltage_v')
    op.drop_column('products', 'product_type')
