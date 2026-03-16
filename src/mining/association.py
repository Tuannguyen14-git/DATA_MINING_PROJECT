from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules

def run_association_rules(df, min_support=0.01):

    freq_items = apriori(df, min_support=min_support, use_colnames=True)

    rules = association_rules(freq_items, metric="confidence", min_threshold=0.3)

    return rules