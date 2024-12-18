####################
### ASSIGNMENT 1 ###
####################

# Initial data
data = [
 (0, 6), (1, 10), (2, 8), (3, 8), (4, 1), (5, 9),
 (6, 3), (7, 4), (8, 1), # Up to here is at time 1:1:8
 (9, 3), (10, 7), # Up to time 1:1:10
 # 1:1:11 current time no data
 (12, 5), (13, 8), (14, 9), # Up to time 1:1:14
 (15, 1), (16, 6), (17, 3)  # Up to time 1:1:17
]

# Given initial Count-Min Sketch array at time 1:1:8
# Indexing: array[function_id][index]
cms = [
 [0,3,0,2,4], # h1
 [3,1,2,0,3], # h2
 [2,3,1,0,3], # h3
 [4,3,2,0,0]  # h4
]

def h1(x):
    return (3*(x**2) + 6*x + 8) % 5

def h2(x):
    return (5*(x**2) + x + 5) % 5

def h3(x):
    return (18*x + 3) % 5

def h4(x):
    return (4*(x**2) + 17*x) % 5

hash_functions = [h1, h2, h3, h4]

# Insert a value x into the Count-Min Sketch
def insert_value(x):
    for i, hf in enumerate(hash_functions):
        idx = hf(x)
        cms[i][idx] += 1

# Query approximate count of x
def query(x):
    counts = []
    for i, hf in enumerate(hash_functions):
        idx = hf(x)
        counts.append(cms[i][idx])
    return min(counts) # Count-Min Sketch returns minimum of the hash counts

# We know we have the array state at time 1:1:8
# Now we insert data from time 1:1:9 and 1:1:10, then show state at 1:1:11
for t,v in data:
    if t <= 8:
        continue  # already accounted for in initial table

    insert_value(v)
    
    # Check conditions when we need to output
    # 1.1) At time 1:1:11 (t=11) show the status after adding t=9 and t=10 data
    if t == 10: # after inserting time=10 data, next is current time=11
        # Print the array at time 1:1:11
        print("1.1) Status of index array at time 1:1:11:")
        for i, row in enumerate(cms, start=1):
            print(f"h{i}: {row}")
        print()

    # 1.2) At time 1:1:14 - after inserting t=12, t=13, t=14
    if t == 14:
        print("1.2) Status of index array at time 1:1:14:")
        for i, row in enumerate(cms, start=1):
            print(f"h{i}: {row}")
        # approximate count of "7","8","5"
        c7 = query(7)
        c8 = query(8)
        c5 = query(5)
        print(f"Approx count of 7 at time 1:1:14: {c7}")
        print(f"Approx count of 8 at time 1:1:14: {c8}")
        print(f"Approx count of 5 at time 1:1:14: {c5}")
        print()

    # 1.3) At time 1:1:17 - after inserting t=15, t=16, t=17
    if t == 17:
        print("1.3) Status of index array at time 1:1:17:")
        for i, row in enumerate(cms, start=1):
            print(f"h{i}: {row}")
        # approximate count of "6","4","3"
        c6 = query(6)
        c4 = query(4)
        c3 = query(3)
        print(f"Approx count of 6 at time 1:1:17: {c6}")
        print(f"Approx count of 4 at time 1:1:17: {c4}")
        print(f"Approx count of 3 at time 1:1:17: {c3}")
        print()

######################
### ASSIGNMENT 2.1 ###
######################

print("2.1) --//--")

# Given Information Gains
ig_values = {
    "Outlook": 0.284,
    "Temperature": 0.342,
    "Humidity": 0.678,
    "Wind": 0.343
}

# Hoeffding bound
epsilon = 0.331

# Sort attributes by IG
sorted_attrs = sorted(ig_values.items(), key=lambda x: x[1], reverse=True)
best_attr, best_ig = sorted_attrs[0]
second_best_attr, second_best_ig = sorted_attrs[1]

# Check difference against epsilon
if (best_ig - second_best_ig) > epsilon:
    chosen_attribute = best_attr
else:
    # If not enough evidence, we might continue to accumulate data
    chosen_attribute = None

if chosen_attribute:
    # For this problem, we know Humidity is chosen
    # Humidity attribute values: High, Normal
    attribute_values = ["High", "Normal"]

    print("The chosen attribute for the root node is:", chosen_attribute)
    print("Partial Decision Tree Structure:")
    print(chosen_attribute)
    for val in attribute_values:
        print(f"   ├── {val}")
else:
    print("Not enough evidence to choose an attribute yet.")

######################
### ASSIGNMENT 2.2 ###
######################

print("\n2.2) --//--")

# Given IG values from the table:

# Format: IG[(condition, attribute)] = value
# 'condition' is like "Humidity:High" or "Humidity:Normal"
ig_values = {
    ("Humidity:High", "Outlook"): 0.653,
    ("Humidity:High", "Temperature"): 0.183,
    ("Humidity:High", "Wind"): 0.318,

    ("Humidity:Normal", "Outlook"): 0.231,
    ("Humidity:Normal", "Temperature"): 0.129,
    ("Humidity:Normal", "Wind"): 0.568
}

# We have 2 branches from the root "Humidity": "High" and "Normal".
# For each branch, choose the attribute with the highest IG.
def choose_attribute(condition):
    candidates = []
    for (cond, attr), val in ig_values.items():
        if cond == condition:
            candidates.append((attr, val))
    # Sort and pick top
    candidates.sort(key=lambda x: x[1], reverse=True)
    best_attr, best_ig = candidates[0]
    return best_attr, best_ig

# For Humidity=High branch
best_attr_high, best_ig_high = choose_attribute("Humidity:High")

# For Humidity=Normal branch
best_attr_normal, best_ig_normal = choose_attribute("Humidity:Normal")

print("For Humidity=High branch, choose attribute:", best_attr_high, "with IG=", best_ig_high)
print("For Humidity=Normal branch, choose attribute:", best_attr_normal, "with IG=", best_ig_normal)

# Print a simple structure of the partial decision tree
print("\nPartial Decision Tree Structure:")
print("            Humidity")
print("           /       \\")
print("      High           Normal")
print("       |                |")
print(f"   {best_attr_high}          {best_attr_normal}")