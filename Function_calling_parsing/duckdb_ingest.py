import orjson
import duckdb
import pandas as pd

# Load your JSON output
with open("/kaggle/working/your_output_file7.json", "rb") as f:
    data = orjson.loads(f.read())

con = duckdb.connect(database="/kaggle/working/properties.db")

con.execute("""
CREATE TABLE IF NOT EXISTS properties (
    id TEXT PRIMARY KEY,
    property_category TEXT,
    property_type TEXT,
    intent TEXT,
    location TEXT,
    area_value TEXT,
    area_unit TEXT,
    price TEXT,
    contact TEXT,
    additional_details TEXT,
    processing_time DOUBLE
)
""")

insert_query = """
INSERT OR REPLACE INTO properties (
    id, property_category, property_type, intent, location,
    area_value, area_unit, price, contact, additional_details, processing_time
) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
"""

for item in data:
    msg_id = item.get("basic_message_info", {}).get("id")
    processing_time = item.get("processing_time")
    smc = item.get("structured_message_content", {})
    for key, prop in smc.items():
        unique_id = f"{msg_id}_{key}"  # Use property key for uniqueness
        area = prop.get("area") or {}
        price = prop.get("price") or {}
        con.execute(
            insert_query,
            [
                unique_id,
                prop.get("property_category"),
                prop.get("property_type"),
                prop.get("intent"),
                prop.get("location"),
                area.get("value"),
                area.get("unit"),
                orjson.dumps(price).decode() if price else None,
                ", ".join(prop.get("contact", [])) if prop.get("contact") else None,
                ", ".join(prop.get("additional_details", [])) if prop.get("additional_details") else None,
                processing_time,
            ]
        )

# Query example
results = con.execute("SELECT * FROM properties LIMIT 5").fetchall()
for row in results:
    print(row)

# Display the table as a DataFrame
df = con.execute("SELECT * FROM properties").df()
print(df)