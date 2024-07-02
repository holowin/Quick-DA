import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

def one_hot_encode(df, discrete_vars):
    encoder = OneHotEncoder(drop=None)
    if discrete_vars:
        try:
            encoded_data = encoder.fit_transform(df[discrete_vars]).toarray()
            new_cols = encoder.get_feature_names_out(discrete_vars)
            encoded_df = pd.DataFrame(encoded_data, columns=new_cols, index=df.index)
            df = df.drop(discrete_vars, axis=1)
            df = pd.concat([df, encoded_df], axis=1)
            return df, new_cols.tolist()
        except Exception as e:
            st.error(f"Error during one-hot encoding: {e}")
            return df, []
    else:
        return df, []

def read_file(uploaded_file, file_type):
    try:
        if file_type == 'CSV':
            return pd.read_csv(uploaded_file)
        elif file_type == 'TXT':
            return pd.read_csv(uploaded_file, sep='\t')
        else:
            st.error("Unsupported file type")
            return None
    except Exception as e:
        st.error(f"Error reading the file: {e}")
        return None

def check_data_types(df, numeric_vars, discrete_vars):
    for var in discrete_vars:
        if df[var].dtype in ['int64', 'float64']:
            st.warning(f"The variable '{var}' was identified as discrete but is of numeric type. Please double check your selection. The program may continue, but the results might not be accurate.")
            return True
    return True

def plot_feature_importances(feature_importances, st):
    plt.figure(figsize=(10, 6))
    sns.barplot(x=feature_importances.importance, y=feature_importances.index, palette="deep")
    plt.title("Feature Importances")
    plt.xlabel("Importance")
    plt.ylabel("Features")
    st.pyplot(plt)


def plot_relationship(df, result_column, numeric_vars, hue_var, st):
    for var in numeric_vars:
        sns.lmplot(x=var, y=result_column, data=df, hue=hue_var, aspect=2, palette="deep")
        plt.title(f"Relationship between {var} and {result_column}")
        st.pyplot(plt)

def plot_pairplot(df, numeric_vars, discrete_var, st):
    if discrete_var:
        plt.figure(figsize=(10, 10))
        sns.pairplot(df, vars=numeric_vars, hue=discrete_var, palette="deep")
        # plt.title(f"Pair Plot of Numerical Variables with {discrete_var} as Hue")
        st.pyplot(plt)
    else:
        st.error("No discrete variable selected for hue in pair plot.")        

# Main app
def main():
    st.title("CSV/TXT Analysis Streamlit App")

    uploaded_file = st.file_uploader("Choose a CSV or TXT file")
    file_type = st.radio("Select the file type:", ('CSV', 'TXT'))

    if uploaded_file is not None:
        df = read_file(uploaded_file, file_type)
        if df is not None:
            result_column = st.selectbox("Select the result column", df.columns)
            variables = st.multiselect("Select the variables", [col for col in df.columns if col != result_column])

            numeric_vars = st.multiselect("Select numerical variables", variables)
            discrete_vars = [var for var in variables if var not in numeric_vars]
            hue_var = st.selectbox("Select the discrete variable for hue in pair plot", discrete_vars)

            if not check_data_types(df, numeric_vars, discrete_vars):
                return

            n_estimators = st.number_input("Enter the number of trees in the forest (multiples of 10):", min_value=10, value=100, step=10)
            random_state = st.number_input("Enter the random state:", value=42)

            if st.button("Analyze"):
                original_df = df.copy() 

                if discrete_vars:
                    df, new_encoded_vars = one_hot_encode(df, discrete_vars)
                    selected_vars = numeric_vars + new_encoded_vars
                else:
                    selected_vars = numeric_vars
    
                if hue_var not in df.columns and hue_var in original_df.columns:
                    df[hue_var] = original_df[hue_var]

                st.write("Pair plot of numerical variables")
                plot_pairplot(df, numeric_vars, hue_var, st)  

                # Prepare data for analysis
                X = df[selected_vars]
                y = df[result_column]

                rf_model = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state)
                rf_model.fit(X, y)

                feature_importances = pd.DataFrame(rf_model.feature_importances_,
                                                    index=X.columns,
                                                    columns=['importance']).sort_values('importance', ascending=False)
                plot_feature_importances(feature_importances, st)

                if numeric_vars:
                    plot_relationship(df, result_column, numeric_vars, hue_var, st)

if __name__ == "__main__":
    main()
