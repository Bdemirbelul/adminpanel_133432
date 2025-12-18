import os
from scrapers import company2, company5

os.makedirs("data", exist_ok=True)

# Remax
company2.run(output_dir="data")
# Dialog
company5.run(output_dir="data")

