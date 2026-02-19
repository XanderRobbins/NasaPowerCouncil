from dotenv import load_dotenv
import os

load_dotenv()

print("=" * 60)
print("ENVIRONMENT CONFIGURATION CHECK")
print("=" * 60)

# Check all important vars
vars_to_check = [
    'NASA_POWER_BASE_URL',
    'DATA_STORAGE_PATH',
    'DATABASE_URL',
    'MARKET_DATA_PROVIDER',
    'TARGET_PORTFOLIO_VOL',
    'BACKTEST_START_DATE',
    'LOG_LEVEL'
]

for var in vars_to_check:
    value = os.getenv(var)
    status = "✅" if value else "❌"
    print(f"{status} {var}: {value}")

print("=" * 60)

# Check the critical DATABASE_URL format
db_url = os.getenv('DATABASE_URL')
if './data_storage/' in db_url:
    print("✅ Database URL format is CORRECT")
else:
    print("❌ Database URL format is WRONG")
    print("   Should contain: './data_storage/'")
    print(f"   Current value: {db_url}")

print("=" * 60)