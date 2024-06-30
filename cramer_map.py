import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency


def cramerV(label,x):
    confusion_matrix = pd.crosstab(label, x)
    chi2 = chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    r,k = confusion_matrix.shape
    phi2 = chi2/n
    phi2corr = max(0,phi2-((k-1)*(r-1))/(n-1))
    rcorr = r - ((r - 1) ** 2) / ( n - 1 )
    kcorr = k - ((k - 1) ** 2) / ( n - 1 )

    try:
        if min((kcorr - 1),(rcorr - 1)) == 0:
            warnings.warn(
            "Unable to calculate Cramer's V using bias correction. Consider not using bias correction",RuntimeWarning)
            v = 0
            print("If condition Met: ",v)
        else:
            v = np.sqrt(phi2corr / min((kcorr - 1), (rcorr - 1)))
            # print("Else condition Met: ",v)
    except:
        print("inside error")
        v = 0

    return v

def cramer_mat(df):
    categorical_plot = list(df.select_dtypes(['object']).columns)
    cramer = pd.DataFrame(index=categorical_plot,columns=categorical_plot)

    for column_of_interest in categorical_plot:
        try:
            temp = {}

            columns = list(df.select_dtypes(['object']).columns)
            for j in range(0,len(columns)):
                v = cramerV(df[column_of_interest],df[columns[j]])
                cramer.loc[column_of_interest,columns[j]] = v
                if (column_of_interest==columns[j]):
                    pass
                else:
                    temp[columns[j]] = v
            cramer.fillna(value=np.nan,inplace=True)
            cramer = cramer.infer_objects(copy=False)
        except:
            print('Dropping row:',column_of_interest)
            pass

    return cramer
