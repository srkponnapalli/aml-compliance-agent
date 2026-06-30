"""Standalone synthetic-data generator for the NorthStar AML compliance demo.

Creates the ``northstar.compliance`` Delta Lake tables (customers, transactions,
sanctioned_countries) with three planted violations:

    1. CTR          – James Whitfield (C001) single $12,500 cash deposit
    2. Structuring  – Arash Mohammadi (C002, High-risk) 4 sub-$10K deposits in 7 days
    3. OFAC         – Linda Chen (C003) $15,000 wire transfer to Iran

This is the job-runnable equivalent of ``notebooks/02_synthetic_data.py``. Run it as a
Databricks job task (``spark`` is provided automatically) or via ``spark-submit`` against a
Unity Catalog + Delta enabled cluster. All data is fully synthetic.
"""

import random
from datetime import date, datetime, timedelta

from pyspark.sql import Row, SparkSession

random.seed(42)


# ─────────────────────────────────────────────
# Reference data
# ─────────────────────────────────────────────

FIRST_NAMES = [
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

LAST_NAMES = [
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

OCCUPATIONS = [
    "Restaurant Owner", "Import/Export Trader", "Consultant", "Retail Manager", "Software Engineer",
    "Accountant", "Freelance Designer", "Financial Analyst", "Healthcare Worker", "Construction Manager",
    "Teacher", "Engineer", "Lawyer", "Doctor", "Pharmacist", "Real Estate Agent", "Marketing Manager",
    "Sales Representative", "HR Manager", "Operations Manager", "Logistics Coordinator", "IT Specialist",
    "Nurse", "Electrician", "Plumber", "Chef", "Journalist", "Photographer", "Architect", "Dentist",
]

STREETS = [
    "Bay St", "King St W", "Yonge St", "Front St W", "Bloor St W", "University Ave", "Queen St E",
    "Adelaide St E", "Dundas St W", "Spadina Ave", "College St", "Wellesley St", "Church St", "Jarvis St",
    "Parliament St", "Broadview Ave", "Danforth Ave", "St Clair Ave W", "Eglinton Ave E", "Lawrence Ave W",
]

CITIES = ["Toronto, ON", "Mississauga, ON", "Brampton, ON", "Markham, ON", "Vaughan, ON"]

TRANSACTION_TYPES = ["Cash Deposit", "Cash Withdrawal", "Wire Transfer", "ACH Transfer", "Check Deposit"]
CHANNELS = ["Branch", "ATM", "Online", "Mobile"]
CLEAN_COUNTRIES = ["USA", "Canada", "UK", "Germany", "France", "Australia", "Japan", "Singapore", "Switzerland", "Netherlands"]
COUNTERPARTY_BANKS = ["RBC", "TD Bank", "Scotiabank", "BMO", "CIBC", "Barclays", "Deutsche Bank", "BNP Paribas", "UBS", "DBS"]
COUNTERPARTY_NAMES = [
    "Acme Corp", "Global Trading Ltd", "Northern Investments", "Pacific Rim Group", "Atlantic Holdings",
    "Summit Financial", "Crestwood Enterprises", "Meridian Partners", "Apex Solutions", "Horizon Group",
]


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def random_date(start_year=2015, end_year=2023):
    start = date(start_year, 1, 1)
    end = date(end_year, 12, 31)
    return start + timedelta(days=random.randint(0, (end - start).days))


def random_address():
    return f"{random.randint(1, 999)} {random.choice(STREETS)}, {random.choice(CITIES)}"


def random_risk():
    """Risk distribution: 70% Low, 20% Medium, 10% High."""
    r = random.random()
    if r < 0.70:
        return "Low"
    elif r < 0.90:
        return "Medium"
    return "High"


def random_datetime():
    start = datetime(2024, 1, 1)
    end = datetime(2024, 6, 1)
    delta = end - start
    return start + timedelta(seconds=random.randint(0, int(delta.total_seconds())))


def random_amount(min_amt=100, max_amt=8000):
    return round(random.uniform(min_amt, max_amt), 2)


# ─────────────────────────────────────────────
# Table builders
# ─────────────────────────────────────────────

def build_customers():
    customers_data = [
        Row(customer_id="C001", name="James Whitfield", account_opening_date=date(2019, 3, 15), address="120 Bay St, Toronto, ON", occupation="Restaurant Owner", risk_rating="Low"),
        Row(customer_id="C002", name="Arash Mohammadi", account_opening_date=date(2021, 7, 22), address="88 King St W, Toronto, ON", occupation="Import/Export Trader", risk_rating="High"),
        Row(customer_id="C003", name="Linda Chen", account_opening_date=date(2020, 1, 10), address="45 Yonge St, Toronto, ON", occupation="Consultant", risk_rating="Low"),
    ]

    used_names = {"James Whitfield", "Arash Mohammadi", "Linda Chen"}
    i = 4
    attempts = 0
    while len(customers_data) < 100 and attempts < 10000:
        attempts += 1
        name = f"{random.choice(FIRST_NAMES)} {random.choice(LAST_NAMES)}"
        if name in used_names:
            continue
        used_names.add(name)
        customers_data.append(Row(
            customer_id=f"C{i:03d}",
            name=name,
            account_opening_date=random_date(),
            address=random_address(),
            occupation=random.choice(OCCUPATIONS),
            risk_rating=random_risk(),
        ))
        i += 1

    return customers_data


def build_transactions():
    transactions_data = []
    txn_id = 1

    # VIOLATION 1: CTR – C001 single cash deposit over $10,000
    transactions_data.append(Row(
        transaction_id="T0001", customer_id="C001",
        transaction_date=datetime(2024, 6, 10, 10, 30),
        amount=12500.00, transaction_type="Cash Deposit", channel="Branch", currency="USD",
        destination_country="USA", counterparty_bank=None, counterparty_name=None,
        notes="Cash deposit from restaurant weekend sales",
    ))
    txn_id += 1

    # VIOLATION 2: Structuring – C002 (High-risk) 4 deposits just under $10K in 7 days
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

    # VIOLATION 3: OFAC – C003 wire to Iran
    transactions_data.append(Row(
        transaction_id=f"T{txn_id:04d}", customer_id="C003",
        transaction_date=datetime(2024, 6, 12, 8, 0),
        amount=15000.00, transaction_type="Wire Transfer", channel="Online", currency="USD",
        destination_country="Iran", counterparty_bank="Bank Mellat",
        counterparty_name="Tehran Trading Co.",
        notes="Payment for consulting services",
    ))
    txn_id += 1

    # Remaining clean transactions up to 10,000 total
    customer_ids = [f"C{i:03d}" for i in range(1, 101)]
    while len(transactions_data) < 10000:
        txn_type = random.choice(TRANSACTION_TYPES)
        if txn_type == "Wire Transfer":
            c_bank = random.choice(COUNTERPARTY_BANKS)
            c_name = random.choice(COUNTERPARTY_NAMES)
        else:
            c_bank = None
            c_name = None

        transactions_data.append(Row(
            transaction_id=f"T{txn_id:04d}",
            customer_id=random.choice(customer_ids),
            transaction_date=random_datetime(),
            amount=random_amount(100, 8000),
            transaction_type=txn_type,
            channel=random.choice(CHANNELS),
            currency="USD",
            destination_country=random.choice(CLEAN_COUNTRIES),
            counterparty_bank=c_bank,
            counterparty_name=c_name,
            notes="Routine transaction",
        ))
        txn_id += 1

    return transactions_data


def build_sanctioned_countries():
    return [
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


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main(spark):
    spark.sql("CREATE CATALOG IF NOT EXISTS northstar")
    spark.sql("CREATE SCHEMA IF NOT EXISTS northstar.compliance")
    spark.sql("USE northstar.compliance")

    customers_df = spark.createDataFrame(build_customers())
    customers_df.write.format("delta").mode("overwrite").saveAsTable("northstar.compliance.customers")
    print(f"✓ customers table created: {customers_df.count()} rows")

    transactions_df = spark.createDataFrame(build_transactions())
    transactions_df.write.format("delta").mode("overwrite").saveAsTable("northstar.compliance.transactions")
    print(f"✓ transactions table created: {transactions_df.count()} rows")

    sanctioned_df = spark.createDataFrame(build_sanctioned_countries())
    sanctioned_df.write.format("delta").mode("overwrite").saveAsTable("northstar.compliance.sanctioned_countries")
    print(f"✓ sanctioned_countries table created: {sanctioned_df.count()} rows")


if __name__ == "__main__":
    # On a Databricks cluster a `spark` session already exists; getOrCreate() reuses it.
    spark = SparkSession.builder.appName("northstar-synthetic-data").getOrCreate()
    main(spark)
