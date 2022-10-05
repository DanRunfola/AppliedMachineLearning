import pandas as pd
import matplotlib.pyplot as plt

from helperFunctions.apyori import apriori
from helperFunctions import pyfim
#importlib.reload(apriori)


order_data = pd.read_csv("./data/order_products__train.csv")
product_data = pd.read_csv("./data/products.csv")

named_orders = pd.merge(order_data, product_data, on="product_id")

counts = named_orders["product_name"].value_counts()

#Visualization example
just_counts = counts.values[counts.values>1000]
plt.hist(just_counts, range=[0,10000])
plt.savefig("./outputs/foodHist.png")


#Select top 1000
counts = counts[counts > 1000]

selected_orders = named_orders[named_orders["product_name"].isin(counts.index.values.tolist())]
selected_orders["cols"] = selected_orders.groupby('order_id').cumcount()
selected_pivot = selected_orders.pivot(index='order_id', columns='cols')[["product_name"]]
#print(selected_pivot)

purchases = []
for i in range(0, len(selected_pivot)):
    purchases.append([str(selected_pivot.values[i,j]) for j in range(0,25)])

print(purchases[0])

#APRIORI
#Note this should be MAX length below.
rules = apriori(purchases, min_support=0.01, min_confidence=0.1, min_lift=3, max_length=20)

results = list(rules)
print(results[0])

rules = 0
for i in range(0, len(results)):
    result = results[i]
    supp = int(result.support*10000)/100
    conf = int(result.ordered_statistics[0].confidence*100)
    hypo = ''.join([x+' ' for x in result.ordered_statistics[0].items_base])
    conc = ''.join([x+' ' for x in result.ordered_statistics[0].items_add])
    #Removing nan from both sides of the equation
    if "nan" not in hypo and "nan" not in conc:
        rules = rules + 1
        print("If "+str(hypo)+ " are purchased, " + str(conf) + "% of the time " + str(conc) + "are purchased [support = "+str(supp)+"%]")

print("Total rules built, omitting NaN: " + str(rules))

#ECLAT
rules = pyfim.eclat(purchases, supp=2, zmin=2, out=[])
rule_count = 0
for i in range(0, len(rules)):
    supp = round(int(rules[i][1]) / len(purchases)*100,3)
    items = rules[i][0]
    if "nan" not in items:
        rule_count = rule_count + 1
        item_1 = rules[i][0][0]
        item_2 = rules[i][0][1]
        print("If " + str(item_1) + " are purchased, " + str(supp) + "% of the time " + str(item_2) + " is purchased [absolute support = " + str(int(rules[i][1])) + "]")
print("Total rules built, omitting NaN: " + str(rule_count))