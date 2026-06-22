""" Test cases for the comparison core. No spider or model needed """

from scorer import compare_results, gold_requires_order

def check(name, got, want):
    status = "ok" if got == want  else "FAIL"
    print(f"[{status}] {name}: got={got} want={want}")
    assert got == want, name

# order sensitivity cases
check("order_by detected", gold_requires_order("SELECT x FROM t ORDER BY x"), True)
check("order_by_absent", gold_requires_order("SELECT x FROM t WHERE x > 1"), False)
check("order_by_case", gold_requires_order("select x from t order by x desc"), True)

# ordered comparision cases
check("ordered exact", compare_results([(1,), (2,), (3,)], [(1,), (2,), (3,)], True), True)
check("ordered exact", compare_results([(1,), (2,), (3,)], [(3,), (2,), (1,)], True), False)

# unordered (multiset) comparison cases
check("unordered reorder ok", compare_results([(1,), (2,)], [(2,), (1,)], False), True)
check("multiset dup-sensitive", compare_results([(1,), (1,), (2,)], [(1,), (2,)], False), False)
check("multiset dup ok", compare_results([(1,), (1,), (2,)], [(2,), (1,), (1,)], False), True)
 
# --- column permutation ---
check("col perm ok", compare_results([("a", 1), ("b", 2)], [(1, "a"), (2, "b")], False), True)
check("col perm but data differs", compare_results([("a", 1), ("b", 2)], [(2, "a"), (1, "b")], False), False)
 
# --- numeric normalization ---
check("int vs float equal", compare_results([(5,)], [(5.0,)], False), True)
check("float noise rounded", compare_results([(1.0000001,)], [(1.0,)], False), True)
check("float genuine diff", compare_results([(1.5,)], [(2.5,)], False), False)
 
# --- shape mismatches ---
check("row count mismatch", compare_results([(1,)], [(1,), (2,)], False), False)
check("col count mismatch", compare_results([(1, 2)], [(1,)], False), False)
check("both empty", compare_results([], [], False), True)
check("both empty ordered", compare_results([], [], True), True)
 
# --- None handling ---
check("nulls compare", compare_results([(None,), (1,)], [(1,), (None,)], False), True)
 
print("\nall scorer tests passed.")
