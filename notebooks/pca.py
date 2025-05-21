import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('data/lifehistory_df.csv')

features = df[['am', 'Wwi', 'Ri', 'Wwb', 'Li', 'ab']]

X = StandardScaler().fit_transform(features)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

df_pca = pd.DataFrame({
    "Species": df["species"],
    "PC1": X_pca[:, 0],
    "PC2": X_pca[:, 1]
})

#save in csv
df_pca.to_csv("data/lifehistory_pca_output.csv", index=False)


