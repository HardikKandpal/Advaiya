from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Union

# --- Basic Message Info ---
class BasicMessageInfo(BaseModel):
    id: Optional[str] = None
    date: Optional[str] = None  # YYYY-MM-DD
    time: Optional[str] = None  # HH:MM
    source_number: Optional[str] = None
    file_chat_name: Optional[str] = Field(None, alias="file/chat_name")

# --- Contact Info ---
class ContactInfo(BaseModel):
    contact_name: Optional[str] = None
    contact_number: Optional[str] = None
    agency_name: Optional[str] = None

# --- Area, Price, Amenities, Additional Space ---
class Area(BaseModel):
    value: Optional[float] = None
    unit: Optional[str] = None
    area_title: Optional[str] = None  # e.g., "carpet area", "builtup area"
    # Add: support for "other" area types
    description: Optional[str] = None

class Price(BaseModel):
    sale_price: Optional[float] = None
    rent: Optional[float] = None
    deposit: Optional[float] = None
    currency: Optional[str] = "INR"

class Amenities(BaseModel):
    list: Optional[List[str]] = Field(default_factory=list)
    parking_type: Optional[List[str]] = None
    parking_count: Optional[int] = None
    # Add: support for amenities like swimming pool, gym, security, etc.
    available_amenities: Optional[List[str]] = None

class AdditionalSpace(BaseModel):
    type: Optional[str] = None  # e.g., "balcony", "sundeck", "private terrace", "loft", "mezzanine", "otla"
    value: Optional[float] = None
    unit: Optional[str] = None
    description: Optional[str] = None

class CommercialConfiguration(BaseModel):
    pillarless_space: Optional[str] = None  # "Yes"/"No"
    cabin_count: Optional[int] = None
    conference_room_count: Optional[int] = None
    workstations: Optional[int] = None
    pantry: Optional[str] = None  # "Yes"/"None"
    back_door: Optional[int] = None
    # Add: support for more flexible/dynamic configuration fields
    other: Optional[dict] = None

class LandLocation(BaseModel):
    village: Optional[str] = None
    taluka: Optional[str] = None
    district: Optional[str] = None
    state: Optional[str] = None
    land_area: Optional['LandArea'] = None
    demarcation: Optional[str] = None  # "Yes"/"No"
    distance_from_pune: Optional[str] = None
    # Add: support for survey_no, gat_no, address breakdown
    survey_no: Optional[str] = None
    gat_no: Optional[str] = None
    address: Optional[str] = None

class RegulatoryDetails(BaseModel):
    zone: Optional[str] = None
    zone_details: Optional[str] = None
    title: Optional[str] = None  # e.g., "Freehold", "Leasehold", "Khaisa", etc.
    encumbrances: Optional[str] = None
    plan_sanction_authority: Optional[str] = None
    plan_type: Optional[str] = None
    plan_id: Optional[str] = None
    # Add: reservation, other_reservation_details
    reservation: Optional[str] = None
    other_reservation_details: Optional[str] = None

class PhysicalDetails(BaseModel):
    shape: Optional[str] = None
    topography: Optional[str] = None
    gt_line: Optional[str] = None  # "Yes"/"No"
    gas: Optional[str] = None
    cable: Optional[str] = None
    waterbody: Optional[str] = None
    distance_from_road: Optional[str] = None
    reservation: Optional[str] = None
    # Add: utilities (water supply, etc.)
    utilities: Optional[dict] = None

# --- Location ---
class LocationComponents(BaseModel):
    street: Optional[str] = None
    landmark: Optional[str] = None
    city: Optional[str] = None
    district: Optional[str] = None
    taluka: Optional[str] = None
    village: Optional[str] = None
    state: Optional[str] = None
    country: Optional[str] = None
    survey_no: Optional[str] = None
    gat_no: Optional[str] = None

class Location(BaseModel):
    address: Optional[str] = None
    components: Optional[LocationComponents] = None

# --- Residential Property ---
class ResidentialProperty(BaseModel):
    property_category: str = "residential"
    property_type: Optional[str] = None  # Flat/Apartments, Bungalow/Villa, etc.
    configuration: Optional[str] = None  # 1BHK, 2BHK, etc.
    area: Optional[Area] = None
    bathrooms: Optional[int] = None
    floor: Optional[int] = None
    total_floors: Optional[int] = None
    furnishing_status: Optional[str] = None
    facing: Optional[str] = None
    view: Optional[str] = None
    ownership: Optional[str] = None
    possession: Optional[str] = None
    maintenance_charges: Optional[float] = None
    amenities: Optional[Amenities] = None
    additional_space: Optional[AdditionalSpace] = None
    project_name: Optional[str] = None
    location: Optional[Location] = None
    price: Optional[Price] = None
    other_details: Optional[str] = None
    # Add: intent (sale/rent), required for schema
    intent: Optional[str] = None

class CommercialProperty(BaseModel):
    property_category: str = "commercial"
    property_type: Optional[str] = None  # Office, Shop, Warehouse, etc.
    configuration: Optional[CommercialConfiguration] = None
    area: Optional[Area] = None
    bathrooms: Optional[int] = None
    floor: Optional[int] = None
    total_floors: Optional[int] = None
    furnishing_status: Optional[str] = None
    facing: Optional[str] = None
    view: Optional[str] = None
    ownership: Optional[str] = None
    possession: Optional[str] = None
    maintenance_charges: Optional[float] = None
    amenities: Optional[Amenities] = None
    additional_space: Optional[AdditionalSpace] = None
    project_name: Optional[str] = None
    location: Optional[Location] = None
    price: Optional[Price] = None
    frontage: Optional[str] = None
    footfall: Optional[str] = None
    rent_potential: Optional[float] = None
    establishment: Optional[str] = None
    other_details: Optional[str] = None
    # Add: intent (sale/rent), required for schema
    intent: Optional[str] = None

class LandProperty(BaseModel):
    property_category: str = "land"
    location: Optional[LandLocation] = None
    regulatory_details: Optional[RegulatoryDetails] = None
    physical_details: Optional[PhysicalDetails] = None
    price: Optional[Price] = None
    other_details: Optional[str] = None
    # Add: intent (sale/rent), required for schema
    intent: Optional[str] = None

# --- Inquiry ---
class RealEstateInquiry(BaseModel):
    inquiry_type: Optional[str] = None  # rental, purchase, etc.
    preferred_location: Optional[str] = None
    budget: Optional[Price] = None
    other_details: Optional[str] = None

# --- Message Category & Intent ---
class MessageCategory(BaseModel):
    type: Optional[str] = None  # e.g., "listing", "inquiry"
    details: Optional[Dict[str, Any]] = None

class Intent(BaseModel):
    type: Optional[str] = None  # e.g., "property_for_sale", "property_for_rent"
    property_use: Optional[str] = None  # e.g., "residential", "commercial", "open plot/land", "paying guest"

# --- Budget Range for Inquiries ---
class BudgetRange(BaseModel):
    min_range: Optional[float] = None
    max_range: Optional[float] = None
    currency: Optional[str] = "INR"

class InquiryBudget(BaseModel):
    sale: Optional[BudgetRange] = None
    rental: Optional[BudgetRange] = None

# --- Paying Guest Property ---
class PayingGuestProperty(BaseModel):
    property_category: str = "paying_guest"
    occupancy_type: Optional[str] = None  # "Single", "Sharing"
    area: Optional[Area] = None
    amenities: Optional[Amenities] = None
    price: Optional[Price] = None
    location: Optional[Location] = None
    other_details: Optional[str] = None

# --- Joint Venture Details ---
class JointVentureDetails(BaseModel):
    ratio: Optional[str] = None
    security_deposit: Optional[str] = None
    other_details: Optional[str] = None

# --- Update StructuredMessageContent to support new property types ---
class StructuredMessageContent(BaseModel):
    property_listings: Optional[Dict[str, Union[
        ResidentialProperty, CommercialProperty, LandProperty, PayingGuestProperty
    ]]] = None
    real_estate_inquiries: Optional[Dict[str, RealEstateInquiry]] = None
    joint_venture: Optional[Dict[str, JointVentureDetails]] = None

# --- Update Main Structured Format ---
class MessageStructuredFormat(BaseModel):
    basic_message_info: Optional[BasicMessageInfo] = None
    contact_info: Optional[ContactInfo] = None
    message_category: Optional[MessageCategory] = None
    intent: Optional[Intent] = None
    structured_message_content: Optional[StructuredMessageContent] = None
    inquiry_budget: Optional[InquiryBudget] = None

# --- Example Usage ---
# To create a message with two property listings and one inquiry:
# msg = MessageStructuredFormat(
#     basic_message_info=BasicMessageInfo(id="001", date="2023-07-09", ...),
#     contact_info=ContactInfo(contact_name="Sunil Gare", ...),
#     structured_message_content=StructuredMessageContent(
#         property_listings={
#             "property_listing_A": ResidentialProperty(...),
#             "property_listing_B": CommercialProperty(...)
#         },
#         real_estate_inquiries={
#             "real_estate_inquiry_A": RealEstateInquiry(...)
#         }
#     )
# )