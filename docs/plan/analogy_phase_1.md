The Detective Agency Returns: The Case of the Vanishing Customers

The Story
You're now the Chief Data Detective at TelcoMax, a phone/internet company. Business is good, but there's a problem:
Customers are mysteriously disappearing (canceling their subscriptions).
The CEO storms into your office: "We're losing money! Find out WHO is going to leave BEFORE they do, so we can stop them!"

Your Investigation Tools
Instead of photos of suspects (like MNIST), you have a filing cabinet with customer records:
Customer #1234
├── Name: John Smith
├── Been with us: 12 months
├── Monthly bill: $70
├── Contract type: Month-to-month
├── Has internet: Yes
├── Has phone: Yes
├── Tech support: No
└── STATUS: Left us last month ❌
You have 7,043 customer files like this. Some left (churned), most stayed.

Phase 1: Study the Files (Data Exploration)
1.1: Sort Through the Filing Cabinet
You dump all the files on your desk and organize them:
What you're looking for:

How many customers total? (Row count)
What info do we have on each? (Columns)
Are any files incomplete? (Missing data)
What do the numbers look like? (Statistics)

The Script = Your Assistant
When you run 01_explore_data.py, your assistant:

Opens the filing cabinet (loads CSV)
Counts the files (7,043 customers)
Lists what info each file has (21 features)
Tells you how many left vs stayed (Churn distribution)
Checks for missing papers (null values)
Summarizes the numbers (describe statistics)

Why this matters: You can't solve a case without knowing what evidence you have!

The Clues (Features Explained)
Think of each column as a different type of clue:
Numeric Clues (Numbers you can measure)

tenure = How long they've been a customer

Like: "Been with us 3 months" vs "Been with us 5 years"
Pattern: New relationships are fragile, long ones are stable


MonthlyCharges = Their monthly bill

Like: "Paying $30/month" vs "Paying $100/month"
Pattern: Higher bills = more likely to shop around


TotalCharges = Everything they've ever paid

Calculated from tenure × MonthlyCharges



Categorical Clues (Labels/categories)

Contract = Type of commitment

Month-to-month = Easy to leave (no penalty)
One year = Some commitment
Two year = Locked in (less likely to leave)
Like: Dating vs engaged vs married


InternetService = What service they have

DSL = Slower, cheaper
Fiber optic = Faster, expensive
No = Phone only
Pattern: Fiber customers pay more, might leave for better deal


TechSupport = Do they get help when needed?

Yes = Happy customer
No = Frustrated when things break
Pattern: No support = more likely to leave


PaymentMethod = How they pay

Electronic check = Manual every month (annoying)
Auto-pay = Set and forget (sticky)
Pattern: Manual payment = easier to stop




The Mystery (Class Imbalance)
Here's the twist: Most customers DON'T leave!
Out of 7,043 customers:

~5,174 stayed (73%)
~1,869 left (27%)

Why this is tricky:
If you predict "Everyone will stay," you'd be 73% accurate but completely useless!
It's like a detective saying "No crimes will happen today" every day. Technically right most of the time, but you never catch the criminals.
Your job: Find the 27% who WILL leave, not just be right about the 73% who won't.

Phase 1.2: Clean Up the Files (Data Cleaning)
Some customer files have problems:
Problem 1: Messy Handwriting

TotalCharges column has spaces (" ") instead of numbers
Like: A form that says "Total: [blank]" instead of "Total: $150"
Fix: Convert to numbers, handle blanks

Problem 2: Useless Info

customerID = Just a reference number, doesn't predict anything
Like: A case number doesn't tell you who the criminal is
Fix: Remove it

Problem 3: Text Labels

Computers need numbers, not words
"Yes"/"No" needs to become 1/0
"Month-to-month" needs to become something numeric
Fix: Encode categories as numbers (Phase 1.4)


Phase 1.3: Create New Clues (Feature Engineering)
Sometimes the raw data isn't enough. You need to create new clues from existing ones:
Example 1: Customer Loyalty Tier
Raw data: tenure = 3 months
New clue you create:

If tenure < 6 months → "New customer" (flight risk!)
If 6-24 months → "Medium" (settling in)
If 24+ months → "Loyal" (sticky)

Why: Models understand "New vs Loyal" better than raw month counts.
Example 2: Value Per Month
Raw data: tenure = 12, TotalCharges = $840
New clue you create:

charges_per_tenure = 840 / 12 = $70/month

Why: Shows if they're a high-value or low-value customer.
Example 3: Service Bundle
Raw data: InternetService = Yes, PhoneService = Yes, StreamingTV = Yes
New clue you create:

has_multiple_services = 3 services = True

Why: Customers with more services are more "locked in"

Phase 1.4: Translate Everything to Numbers (Encoding)
The problem: Your detective brain (the ML model) only understands numbers, not words.
Example translations:
OriginalEncodedChurn: "Yes"1Churn: "No"0Gender: "Male"0Gender: "Female"1Contract: "Month-to-month"[1, 0, 0]Contract: "One year"[0, 1, 0]Contract: "Two year"[0, 0, 1]
Why the weird [1,0,0] thing?
Because "Month-to-month" isn't "bigger" than "One year" - they're just different categories. So we create separate yes/no columns for each:

Contract_MonthToMonth: 1 or 0
Contract_OneYear: 1 or 0
Contract_TwoYear: 1 or 0

This is called one-hot encoding.

What Running the Script Does
When you run 01_explore_data.py, you're like a detective doing the initial case review:
pythondf = pd.read_csv('telco_churn.csv')  # Open the filing cabinet
print(df.head())                      # Look at first 5 files
print(df['Churn'].value_counts())    # Count how many left vs stayed
print(df.isnull().sum())             # Check for incomplete files
print(df.describe())                  # Summarize all the numbers
Output tells you:

✓ We have 7,043 case files
✓ Each has 21 pieces of information
✓ 27% of customers left (our targets)
✓ Some files might have missing data
✓ Monthly charges range from $18-$118


Next Steps After Exploration
Once you understand the files, you'll:

Clean them (fix messy data)
Create new clues (feature engineering)
Translate to numbers (encoding)
Train the model (find the patterns)

Then your model becomes like a psychic detective who can look at a new customer file and say:
"This customer has been here 2 months, pays $95/month, has month-to-month contract, no tech support... 85% chance they'll leave in 3 months!"
And the company can intervene before they go.

------------------------------------------------------------------------------------------