# Databricks notebook source
# MAGIC %md
# MAGIC # 02 — Synthetic Data
# MAGIC
# MAGIC Creates the `northstar.compliance` Delta Lake tables with three planted violations:
# MAGIC
# MAGIC | # | Type | Customer | Description | Policy Clause |
# MAGIC |---|------|----------|-------------|---------------|
# MAGIC | 1 | CTR | James Whitfield (C001) | Single cash deposit of $12,500 | 2.1.1 |
# MAGIC | 2 | Structuring / SAR | Arash Mohammadi (C002, High-risk) | 4 cash deposits of $9,200–$9,900 within 7 days | 2.2.1 |
# MAGIC | 3 | OFAC | Linda Chen (C003) | Wire transfer of $15,000 to Iran | 5.2.1 |
# MAGIC
# MAGIC All data is fully synthetic. See `scripts/northstar_synthetic_data.py` for a standalone
# MAGIC (job-runnable) version of this notebook.

# COMMAND ----------

import random
from datetime import date, datetime, timedelta
from pyspark.sql import Row

random.seed(42)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 1 — Catalog and schema

# COMMAND ----------

spark.sql("CREATE CATALOG IF NOT EXISTS northstar")
spark.sql("CREATE SCHEMA IF NOT EXISTS northstar.compliance")
spark.sql("USE northstar.compliance")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2 — Customers table (100 customers)
# MAGIC
# MAGIC Violation seeds C001/C002/C003 are fixed; the remaining 97 are randomly generated.

# COMMAND ----------

first_names = [
    "James", "Linda", "Marcus", "Priya", "David", "Sophie", "Kevin", "Fatima", "Robert", "Angela",
    "Michael", "Sarah", "John", "Emily", "Chris", "Amanda", "Daniel", "Jessica", "Matthew", "Ashley",
    "Andrew", "Melissa", "Joshua", "Stephanie", "Ryan", "Nicole", "Justin", "Elizabeth", "Brandon", "Heather",
    "Tyler", "Amber", "Jacob", "Megan", "Nicholas", "Rachel", "Nathan", "Samantha", "Eric", "Katherine",
    "Jonathan", "Christine", "Adam", "Deborah", "Patrick", "Rebecca", "Benjamin", "Sharon", "Sean", "Laura",
    "Raj", "Aisha", "Wei", "Yuki", "Carlos", "Maria", "Ahmed", "Fatou", "Ivan", "Olga",
    "Kwame", "Amara", "Hassan", "Leila", "Diego", "Isabella", "Tariq", "Nadia", "Emre", "Ayşe",
    "Liam", "Emma", "Noah", "Olivia", "Ethan", "Ava", "Mason", "Sophia", "Logan", "Mia",
    "Lucas", "Charlotte", "Oliver", "Amelia", "Aiden", "Harper", "Elijah", "Evelyn", "James", "Abigail",
    "Alexander", "Emily", "Henry", "Elizabeth", "Sebastian", "Sofia", "Jack", "Ella", "Owen", "Scarlett",
]

last_names = [
    "Whitfield", "Mohammadi", "Chen", "Brown", "Okafor", "Tremblay", "Park", "Al-Hassan", "Kazinski", "Torres",
    "Smith", "Johnson", "Williams", "Jones", "Garcia", "Miller", "Davis", "Rodriguez", "Martinez", "Wilson",
    "Anderson", "Taylor", "Thomas", "Hernandez", "Moore", "Martin", "Jackson", "Thompson", "White", "Lopez",
    "Lee", "Gonzalez", "Harris", "Clark", "Lewis", "Robinson", "Walker", "Perez", "Hall", "Young",
    "Allen", "Sanchez", "Wright", "King", "Scott", "Green", "Baker", "Adams", "Nelson", "Carter",
    "Patel", "Sharma", "Kumar", "Singh", "Shah", "Gupta", "Khan", "Ahmed", "Ali", "Hassan",
    "Zhang", "Wang", "Li", "Liu", "Chen", "Yang", "Huang", "Wu", "Zhou", "Sun",
    "Tanaka", "Sato", "Suzuki", "Watanabe", "Ito", "Yamamoto", "Nakamura", "Kobayashi", "Kato", "Abe",
    "Mensah", "Asante", "Boateng", "Owusu", "Adjei", "Darko", "Amponsah", "Amoah", "Acheampong", "Frimpong",
    "Mueller", "Schmidt", "Fischer", "Weber", "Meyer", "Wagner", "Becker", "Schulz", "Hoffmann", "Schaefer",
]

occupations = [
    "Restaurant Owner", "Import/Export Trader", "Consultant", "Retail Manager", "Software Engineer",
    "Accountant", "Freelance Designer", "Financial Analyst", "Healthcare Worker", "Construction Manager",
    "Teacher", "Engineer", "Lawyer", "Doctor", "Pharmacist", "Real Estate Agent", "Marketing Manager",
    "Sales Representative", "HR Manager", "Operations Manager", "Logistics Coordinator", "IT Specialist",
    "Nurse", "Electrician", "Plumber", "Chef", "Journalist", "Photographer", "Architect", "Dentist",
]

streets = [
    "Bay St", "King St W", "Yonge St", "Front St W", "Bloor St W", "University Ave", "Queen St E",
    "Adelaide St E", "Dundas St W", "Spadina Ave", "College St", "Wellesley St", "Church St", "Jarvis St",
    "Parliament St", "Broadview Ave", "Danforth Ave", "St Clair Ave W", "Eglinton Ave E", "Lawrence Ave W",
]

cities = ["Toronto, ON", "Mississauga, ON", "Brampton, ON", "Markham, ON", "Vaughan, ON"]


def random_date(start_year=2015, end_year=2023):
    start = date(start_year, 1, 1)
    end = date(end_year, 12, 31)
    return start + timedelta(days=random.randint(0, (end - start).days))


def random_address():
    return f"{random.randint(1, 999)} {random.choice(streets)}, {random.choice(cities)}"


# Risk distribution: 70% Low, 20% Medium, 10% High
def random_risk():
    r = random.random()
    if r < 0.70:
        return "Low"
    elif r < 0.90:
        return "Medium"
    else:
        return "High"


customers_data = []

# Violation seed customers (fixed)
customers_data.append(Row(customer_id="C001", name="James Whitfield", account_opening_date=date(2019, 3, 15), address="120 Bay St, Toronto, ON", occupation="Restaurant Owner", risk_rating="Low"))
customers_data.append(Row(customer_id="C002", name="Arash Mohammadi", account_opening_date=date(2021, 7, 22), address="88 King St W, Toronto, ON", occupation="Import/Export Trader", risk_rating="High"))
customers_data.append(Row(customer_id="C003", name="Linda Chen", account_opening_date=date(2020, 1, 10), address="45 Yonge St, Toronto, ON", occupation="Consultant", risk_rating="Low"))

# Generate remaining 97 customers
used_names = {"James Whitfield", "Arash Mohammadi", "Linda Chen"}
i = 4
attempts = 0
while len(customers_data) < 100 and attempts < 10000:
    attempts += 1
    first = random.choice(first_names)
    last = random.choice(last_names)
    name = f"{first} {last}"
    if name in used_names:
        continue
    used_names.add(name)
    customers_data.append(Row(
        customer_id=f"C{i:03d}",
        name=name,
        account_opening_date=random_date(),
        address=random_address(),
        occupation=random.choice(occupations),
        risk_rating=random_risk(),
    ))
    i += 1

customers_df = spark.createDataFrame(customers_data)
customers_df.write.format("delta").mode("overwrite").saveAsTable("northstar.compliance.customers")
print(f"✓ customers table created: {customers_df.count()} rows")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 3 — Transactions table (10,000 transactions)
# MAGIC
# MAGIC Three planted violations, then ~9,994 routine transactions.

# COMMAND ----------

transaction_types = ["Cash Deposit", "Cash Withdrawal", "Wire Transfer", "ACH Transfer", "Check Deposit"]
channels = ["Branch", "ATM", "Online", "Mobile"]
clean_countries = ["USA", "Canada", "UK", "Germany", "France", "Australia", "Japan", "Singapore", "Switzerland", "Netherlands"]
counterparty_banks = ["RBC", "TD Bank", "Scotiabank", "BMO", "CIBC", "Barclays", "Deutsche Bank", "BNP Paribas", "UBS", "DBS"]
counterparty_names = [
    "Acme Corp", "Global Trading Ltd", "Northern Investments", "Pacific Rim Group", "Atlantic Holdings",
    "Summit Financial", "Crestwood Enterprises", "Meridian Partners", "Apex Solutions", "Horizon Group",
]


def random_datetime(start_date="2024-01-01", end_date="2024-06-01"):
    start = datetime(2024, 1, 1)
    end = datetime(2024, 6, 1)
    delta = end - start
    return start + timedelta(seconds=random.randint(0, int(delta.total_seconds())))


def random_amount(min_amt=100, max_amt=8000):
    return round(random.uniform(min_amt, max_amt), 2)


transactions_data = []
txn_id = 1

# ── VIOLATION 1: CTR – C001 single cash deposit over $10,000 ──────────────
transactions_data.append(Row(
    transaction_id="T0001", customer_id="C001",
    transaction_date=datetime(2024, 6, 10, 10, 30),
    amount=12500.00, transaction_type="Cash Deposit", channel="Branch", currency="USD",
    destination_country="USA", counterparty_bank=None, counterparty_name=None,
    notes="Cash deposit from restaurant weekend sales",
))
txn_id += 1

# ── VIOLATION 2: Structuring – C002 (High-risk) 4 deposits just under $10K in 7 days ──
structuring_amounts = [9800.00, 9500.00, 9200.00, 9900.00]
structuring_dates = [
    datetime(2024, 6, 3, 9, 15),
    datetime(2024, 6, 5, 11, 45),
    datetime(2024, 6, 7, 14, 20),
    datetime(2024, 6, 9, 16, 0),
]
for amt, dt in zip(structuring_amounts, structuring_dates):
    transactions_data.append(Row(
        transaction_id=f"T{txn_id:04d}", customer_id="C002",
        transaction_date=dt, amount=amt,
        transaction_type="Cash Deposit", channel="Branch", currency="USD",
        destination_country="USA", counterparty_bank=None, counterparty_name=None,
        notes="Business cash deposit",
    ))
    txn_id += 1

# ── VIOLATION 3: OFAC – C003 wire to Iran ─────────────────────────────────
transactions_data.append(Row(
    transaction_id=f"T{txn_id:04d}", customer_id="C003",
    transaction_date=datetime(2024, 6, 12, 8, 0),
    amount=15000.00, transaction_type="Wire Transfer", channel="Online", currency="USD",
    destination_country="Iran", counterparty_bank="Bank Mellat",
    counterparty_name="Tehran Trading Co.",
    notes="Payment for consulting services",
))
txn_id += 1

# ── Generate remaining clean transactions up to 10,000 total ──────────────
customer_ids = [f"C{i:03d}" for i in range(1, 101)]

while len(transactions_data) < 10000:
    cust_id = random.choice(customer_ids)
    txn_type = random.choice(transaction_types)
    channel = random.choice(channels)
    amount = random_amount(100, 8000)
    dt = random_datetime()
    country = random.choice(clean_countries)

    if txn_type == "Wire Transfer":
        c_bank = random.choice(counterparty_banks)
        c_name = random.choice(counterparty_names)
    else:
        c_bank = None
        c_name = None

    transactions_data.append(Row(
        transaction_id=f"T{txn_id:04d}",
        customer_id=cust_id,
        transaction_date=dt,
        amount=amount,
        transaction_type=txn_type,
        channel=channel,
        currency="USD",
        destination_country=country,
        counterparty_bank=c_bank,
        counterparty_name=c_name,
        notes="Routine transaction",
    ))
    txn_id += 1

transactions_df = spark.createDataFrame(transactions_data)
transactions_df.write.format("delta").mode("overwrite").saveAsTable("northstar.compliance.transactions")
print(f"✓ transactions table created: {transactions_df.count()} rows")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 4 — Sanctioned countries table

# COMMAND ----------

sanctioned_countries_data = [
    Row(country="Iran", sanction_program="OFAC-Iran", severity="Comprehensive"),
    Row(country="Cuba", sanction_program="OFAC-Cuba", severity="Comprehensive"),
    Row(country="North Korea", sanction_program="OFAC-DPRK", severity="Comprehensive"),
    Row(country="Syria", sanction_program="OFAC-Syria", severity="Comprehensive"),
    Row(country="Russia", sanction_program="OFAC-Russia", severity="Sectoral"),
    Row(country="Belarus", sanction_program="OFAC-Belarus", severity="Sectoral"),
    Row(country="Venezuela", sanction_program="OFAC-Venezuela", severity="Sectoral"),
    Row(country="Myanmar", sanction_program="OFAC-Myanmar", severity="Sectoral"),
    Row(country="Zimbabwe", sanction_program="OFAC-Zimbabwe", severity="Sectoral"),
]

sanctioned_df = spark.createDataFrame(sanctioned_countries_data)
sanctioned_df.write.format("delta").mode("overwrite").saveAsTable("northstar.compliance.sanctioned_countries")
print(f"✓ sanctioned_countries table created: {sanctioned_df.count()} rows")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 5 — Verify all 3 violations are detectable

# COMMAND ----------

print("\n[1] CTR Violation – Cash deposits over $10,000:")
spark.sql("""
    SELECT t.transaction_id, c.name, c.risk_rating, t.amount, t.transaction_type, t.transaction_date
    FROM northstar.compliance.transactions t
    JOIN northstar.compliance.customers c ON t.customer_id = c.customer_id
    WHERE t.transaction_type = 'Cash Deposit'
    AND t.amount > 10000
    ORDER BY t.amount DESC
""").show()

print("\n[2] Structuring – High-risk customer, 3+ sub-$10K cash deposits within 7 days:")
spark.sql("""
    SELECT c.name, c.risk_rating,
           COUNT(*) as num_deposits,
           SUM(t.amount) as total_amount,
           MIN(t.transaction_date) as first_txn,
           MAX(t.transaction_date) as last_txn
    FROM northstar.compliance.transactions t
    JOIN northstar.compliance.customers c ON t.customer_id = c.customer_id
    WHERE t.transaction_type = 'Cash Deposit'
    AND t.amount BETWEEN 9000 AND 9999
    AND c.risk_rating = 'High'
    AND t.transaction_date BETWEEN '2024-06-03' AND '2024-06-10'
    GROUP BY c.name, c.risk_rating
    HAVING COUNT(*) >= 3
""").show()

print("\n[3] OFAC – Wire transfers to sanctioned countries:")
spark.sql("""
    SELECT t.transaction_id, c.name, t.amount, t.destination_country,
           t.counterparty_name, s.sanction_program, s.severity
    FROM northstar.compliance.transactions t
    JOIN northstar.compliance.customers c ON t.customer_id = c.customer_id
    JOIN northstar.compliance.sanctioned_countries s ON t.destination_country = s.country
    WHERE t.transaction_type = 'Wire Transfer'
""").show()

# COMMAND ----------

print("── SUMMARY ──")
spark.sql("SELECT COUNT(*) as total_customers FROM northstar.compliance.customers").show()
spark.sql("SELECT COUNT(*) as total_transactions FROM northstar.compliance.transactions").show()
spark.sql("SELECT transaction_type, COUNT(*) as count FROM northstar.compliance.transactions GROUP BY transaction_type ORDER BY count DESC").show()
