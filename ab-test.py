import pandas as pd
import numpy as np
import sys
import glob
import os
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import shapiro, levene, ttest_ind, mannwhitneyu, f_oneway
sys.path.append("pythonProject/Github/Me/ab-test/")
from helpers.pandas_options import set_pandas_options
set_pandas_options(width=500, precision=4)

# We add determinant column to each dataframe and generate csv again to be aware of the distinction between test groups before merging dataframe.
def add_column_to_csv(data):
    os.makedirs("pythonProject/Github/Me/ab-test/datasets/adj_datasets", exist_ok=True)
    dataframe = pd.read_csv(data)
    dataframe["Version"] = data.split(",")[0].split("- ")[1].upper()
    dataframe.to_csv("pythonProject/Github/Me/ab-test/datasets/adj_datasets/" + data.split(",")[0].split("- ")[1] + ".csv")

for csv in list(glob.glob("ab-test/datasets" + "/*.csv")):
    add_column_to_csv(csv)


# We can read multiple csv using glob.
files = list(glob.glob("pythonProject/Github/Me/ab-test/datasets/adj_datasets" + "/*.csv"))
read_files = [pd.read_csv(file) for file in files]
df = pd.concat(read_files, ignore_index=True)
df = df[["Name", "No. clicks", "Version"]].rename(columns={"No. clicks": "Click"})
df.head()
#                                         Name  Click  Version
# 0                                       FIND    502  CONNECT
# 1                                        s.q    357  CONNECT
# 2                      lib.montana.edu/find/    171  CONNECT
# 3  Montana State University Libraries - Home     83  CONNECT
# 4                                      Hours     74  CONNECT


# we create function to analyze the proportion of click in each version
def group_test_assessment(dataframe, group_name):
    version_total_click = {group_name: dataframe[dataframe["Version"] == group_name]["Click"].sum()}

    for version, click in version_total_click.items():
        if version == "INTERACT":
            print("VERSION - INTERACT (CONTROL GROUP)")
        else:
            print(f"VERSION - {version} (TEST GROUP)")

        dataframe = dataframe[
            (dataframe["Name"].isin(["FIND", "REQUEST", version])) & (dataframe["Version"] == version)].groupby(
            "Name").agg({"Click": lambda x: (x.sum() / click) * 100}).reset_index()
        print("#################################")
        return dataframe


df_interact = group_test_assessment(df, "INTERACT")
df_connect = group_test_assessment(df, "CONNECT")
df_learn = group_test_assessment(df, "LEARN")
df_help = group_test_assessment(df, "HELP")
df_services = group_test_assessment(df, "SERVICES")

df_services.head()
#        Name    Click
# 0      FIND 29.45104
# 1   REQUEST  4.22849
# 2  SERVICES  3.33828


# Pie visualization of distribution of click in each version --> visualization/pie_charts.png
colors = ["slategray", "gray", "silver"]
fig, ax = plt.subplots(3, 3, figsize=(10, 10))
ax[0, 1].pie(df_interact["Click"].values, labels=df_interact["Name"].values, autopct='%.1f%%',
             wedgeprops={'edgecolor': "white", 'linewidth': 1}, textprops={'fontsize': 8}, radius=1.6,
             colors=colors, explode=(0, 0.2, 0))
ax[1, 0].pie(df_connect["Click"].values, labels=df_connect["Name"].values, autopct='%.1f%%',
             wedgeprops={'edgecolor': "white", 'linewidth': 1}, textprops={'fontsize': 8}, radius=1,
             colors=colors, explode=(0.2, 0, 0))
ax[1, 2].pie(df_learn["Click"].values, labels=df_learn["Name"].values, autopct='%.1f%%',
             wedgeprops={'edgecolor': "white", 'linewidth': 1}, textprops={'fontsize': 8}, radius=1,
             colors=colors, explode=(0, 0.2, 0))
ax[2, 0].pie(df_help["Click"].values, labels=df_help["Name"].values, autopct='%.1f%%',
             wedgeprops={'edgecolor': "white", 'linewidth': 1}, textprops={'fontsize': 8}, radius=1,
             colors=colors, explode=(0, 0.2, 0))
ax[2, 2].pie(df_services["Click"].values, labels=df_services["Name"].values, autopct='%.1f%%',
             wedgeprops={'edgecolor': "white", 'linewidth': 1}, textprops={'fontsize': 8}, radius=1,
             colors=colors, explode=(0, 0, 0.2))
ax[0, 0].axis("off")
ax[0, 2].axis("off")
ax[2, 1].axis("off")
ax[1, 1].axis("off")
plt.show()


# Defining AB Test Function
def ab_test(dataframe, control_group, test_group):
    # Hypothesis
        # H0: There is no statistically significant difference between the means of the two versions.
        # H1: There is a statistically significant difference between the means of the two versions.

    # Normality Assumption
        # H0: The assumption of normal distribution is not provided.
        # H1: The assumption of normal distribution is provided.

    # Assumption of Homogeneity of Variance
        # H0: The assumption of homogeneity of variance is not provided.
        # H1: The assumption of homogeneity of variance is provided.

    # Normality Assumption of Control Group:
    pvalue_control_group = shapiro(dataframe.loc[dataframe["Version"] == control_group, "Click"])[1]
    print("For Control Group Normality Assumption P-value = %.4f" % pvalue_control_group)
    # Normality Assumption of Test Group:
    pvalue_test_group = shapiro(dataframe.loc[dataframe["Version"] == test_group, "Click"])[1]
    print("For Test Group Normality Assumption P-value = %.4f" % pvalue_test_group)

    if pvalue_control_group and pvalue_test_group < 0.05:
        print("Normality assumption H0 hypothesis rejected. The assumption of normal distribution is provided.")
        pvalue_levene = levene(dataframe.loc[dataframe["Version"] == control_group, "Click"],
                               dataframe.loc[dataframe["Version"] == test_group, "Click"])[1]

        if pvalue_levene < 0.05:
            print("Assumption of homogeneity of variance H0 hypothesis rejected. The assumption of homogeneity of variance is provided.")
            ttest = ttest_ind(dataframe.loc[dataframe["Version"] == control_group, "Click"],
                              dataframe.loc[dataframe["Version"] == test_group, "Click"], equal_var=True)[1]
        else:
            print("Assumption of homogeneity of variance H0 hypothesis is not rejected. The assumption of homogeneity of variance is not provided.")
            ttest = ttest_ind(dataframe.loc[dataframe["Version"] == control_group, "Click"],
                              dataframe.loc[dataframe["Version"] == test_group, "Click"], equal_var=False)[1]

    else:
        print("Normality assumption H0 hypothesis is not rejected. The assumption of normal distribution is not provided.")
        ttest = mannwhitneyu(dataframe.loc[dataframe["Version"] == control_group, "Click"],
                                           dataframe.loc[dataframe["Version"] == test_group, "Click"])[1]


    df_result = pd.DataFrame(index=["Result"])
    df_result["Test Type"] = np.where((pvalue_control_group) and (pvalue_test_group) < 0.05, "Parametric", "Non-Parametric")
    df_result["P-value"] = ttest
    df_result["Hypothesis Result"] = np.where(ttest < 0.05, "Rejected", "Not Rejected")
    df_result["Comment"] = np.where(df_result["Hypothesis Result"] == "Rejected", "There is a statistically significant difference between the means of the two versions", "There is no statistically significant difference between the means of the two versions")

    return df_result



ab_test(df, "INTERACT", "CONNECT")
ab_test(df, "INTERACT", "LEARN")
ab_test(df, "INTERACT", "HELP")
ab_test(df, "INTERACT", "SERVICES")
#          Test Type  P-value Hypothesis Result                                            Comment
# Result  Parametric   0.2617      Not Rejected  There is no statistically significant difference between the means of the two versions



# Multiple Comparison of Means (ANOVA - Analysis of Variance)

# Since the assumption of normality is known, we can apply f_oneway test, if is not known we should use kruskal test
pvalue = f_oneway(df.loc[df["Version"] == "INTERACT", "Click"],
         df.loc[df["Version"] == "CONNECT", "Click"],
         df.loc[df["Version"] == "LEARN", "Click"],
         df.loc[df["Version"] == "HELP", "Click"],
         df.loc[df["Version"] == "SERVICES", "Click"])[1]
print("p-value = %.4f" % pvalue)
# p-value = 0.6150

# Comparisons within each group
from statsmodels.stats.multicomp import MultiComparison
comparison = MultiComparison(df["Click"], df["Version"])
tukey = comparison.tukeyhsd(0.05)
print(tukey.summary())

#    Multiple Comparison of Means - Tukey HSD, FWER=0.05
# =========================================================
#  group1   group2  meandiff p-adj   lower    upper  reject
# ---------------------------------------------------------
#  CONNECT     HELP   2.7607 0.9999 -58.0751 63.5966  False
#  CONNECT INTERACT   26.464 0.7219 -31.6426 84.5706  False
#  CONNECT    LEARN  -0.7169    1.0 -60.3027 58.8689  False
#  CONNECT SERVICES  -1.9281    1.0  -63.911 60.0548  False
#     HELP INTERACT  23.7033 0.7989 -34.6796 82.0861  False
#     HELP    LEARN  -3.4776 0.9999 -63.3329 56.3776  False
#     HELP SERVICES  -4.6888 0.9996 -66.9308 57.5531  False
# INTERACT    LEARN -27.1809 0.6871 -84.2601 29.8982  False
# INTERACT SERVICES -28.3921 0.6864 -87.9692  31.185  False
#    LEARN SERVICES  -1.2112    1.0 -62.2319 59.8095  False
# ---------------------------------------------------------
