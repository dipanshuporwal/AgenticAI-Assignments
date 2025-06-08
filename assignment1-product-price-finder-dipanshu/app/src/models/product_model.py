from typing import Optional
from pydantic import BaseModel, Field


class Product(BaseModel):
    product_id: Optional[str] = Field(
        default=None,
        description="Unique idenitfier for the product(product number)",
    )
    product_name: Optional[str] = Field(
        default=None, description="The name of the product"
    )
    description: Optional[str] = Field(
        default=None,
        description="Brief description or key features of the product",
    )
    tentative_price_in_usd: Optional[str] = Field(
        default=None, description="Price of the product in USD"
    )
    category: Optional[str] = Field(
        default=None,
        description="Product category such as electronics, clothing, etc.",
    )
    rating: Optional[float] = Field(
        default=None,
        ge=0,
        le=5,
        description="Average customer rating (0 to 5)",
    )
