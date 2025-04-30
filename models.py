from pydantic import BaseModel, Field, validator
from typing import List, Dict, Optional, Union, Any

class Area(BaseModel):
    value: Optional[float] = None
    unit: Optional[str] = None

class Dimensions(BaseModel):
    length: Optional[float] = None
    width: Optional[float] = None
    unit: Optional[str] = None

class Price(BaseModel):
    rent: Optional[float] = None
    deposit: Optional[float] = None
    sale_price: Optional[float] = None
    price_per_unit: Optional[float] = None
    currency: Optional[str] = "INR"

class Amenities(BaseModel):
    items: Optional[List[str]] = Field(default_factory=list)

class BaseProperty(BaseModel):
    property_category: Optional[str] = None
    property_type: str
    intent: str
    area: Optional[Area] = None
    location: Optional[str] = None
    price: Optional[Price] = None
    
    @validator('property_category', pre=True, always=True)
    def set_property_category(cls, v, values, **kwargs):
        # This will be overridden in subclasses
        return v or "unknown"

class ResidentialProperty(BaseProperty):
    property_category: Optional[str] = "residential"
    configuration: Optional[str] = None
    bathrooms: Optional[str] = None
    floor: Optional[str] = None
    total_floors: Optional[str] = None
    facing: Optional[str] = None
    view: Optional[str] = None
    amenities: Optional[Amenities] = None
    furnishing_status: Optional[str] = None
    
    @validator('property_category', pre=True, always=True)
    def set_property_category(cls, v, values, **kwargs):
        return "residential"

class CommercialProperty(BaseProperty):
    property_category: Optional[str] = "commercial"
    floor: Optional[str] = None
    furnishing_status: Optional[str] = None
    amenities: Optional[Amenities] = None
    
    @validator('property_category', pre=True, always=True)
    def set_property_category(cls, v, values, **kwargs):
        return "commercial"

class LandProperty(BaseProperty):
    property_category: Optional[str] = "land"
    dimensions: Optional[Dimensions] = None
    
    @validator('property_category', pre=True, always=True)
    def set_property_category(cls, v, values, **kwargs):
        return "land"