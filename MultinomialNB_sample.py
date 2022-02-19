#%%

#%%
import numpy as np
rng = np.random.RandomState(20)
X = rng.randint(7, size=(6, 4))
y = np.array([1, 2, 3, 1, 2, 3])
from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB()
clf.fit(X, y)

# %%
print(clf.predict(X[2:3]))
# %%
print(y)
# %%
print(X)
print(X.shape)
# %%
X_val = rng.randint(10, size=(1, 4))
print(X_val)
print(clf.predict_proba(X_val))
print(clf.predict(X_val))
# %%
print(clf.predict_log_proba(X_val))

# %%
print(clf.feature_log_prob_)
print(clf.feature_log_prob_.shape)

"""
for val in X_val*clf.feature_log_prob_[0]:
    num = 0
    num += val
    print(num)
"""
# %%
print(f"{X_val * clf.feature_log_prob_[0]}")
#print(f"{sum(X_val * clf.feature_log_prob_[0])}")
print(f"{sum(sum(X_val * clf.feature_log_prob_[0]))}")
print(f"{X_val * clf.feature_log_prob_[1]}")
#print(f"{sum(X_val * clf.feature_log_prob_[1])}")
print(f"{sum(sum(X_val * clf.feature_log_prob_[1]))}")
print(f"{X_val * clf.feature_log_prob_[2]}")
#print(f"{sum(X_val * clf.feature_log_prob_[2])}")
print(f"{sum(sum(X_val * clf.feature_log_prob_[2]))}")

# %%
print(clf.feature_log_prob_[0])

# %%
print(clf.feature_count_)
print(clf.feature_count_.shape)
# %%
print(clf.n_features_in_)
#print(clf.n_features_in_.shape)
# %%
print(clf.class_count_)
print(clf.class_log_prior_)
print(clf.classes_)
print(clf.classes_.shape)

# %%
# %%
