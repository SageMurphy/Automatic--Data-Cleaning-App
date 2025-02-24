import streamlit as st
import pandas as pd
# from pandas_profiling import ProfileReport
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.impute import KNNImputer
from sklearn.linear_model import LinearRegression
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from scipy import stats
from scipy.stats.mstats import winsorize
from sklearn.preprocessing import StandardScaler
from ydata_profiling import ProfileReport
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler




st.title("Automatic Data Cleaning App")
st.markdown("<h6 style='text-align: right; color: gray;'>Developed by Abhishek</h6>", unsafe_allow_html=True)


uploaded_file = st.file_uploader("Upload a CSV file", type = ['csv'])
if 'df' not in st.session_state:
    st.session_state.df = None

if uploaded_file is not None:
    try:
        if st.session_state.df is None:
            st.session_state.df = pd.read_csv(uploaded_file)
        df = st.session_state.df.copy()
        # df = pd.read_csv(uploaded_file)
        st.write('## File Information')
        st.write(f"Rows:{df.shape[0]}")
        st.write(f"Columns: {df.shape[1]}")
        st.write("### Column Information")
        st.write({col: str(dtype) for col, dtype in df.dtypes.items()})

        if st.checkbox("Show Data"):
            st.dataframe(df)

        if st.checkbox("Generate data profile report"):
            with st.spinner("Generating report... "):
                profile = ProfileReport(df,title = "Data Profile Report")
                st.components.v1.html(profile.to_html() , height = 600 , scrolling = True)
# handling duplicates

        st.subheader ("Duplicate Handling")
        st.write(f"Number of duplicate rows: {df.duplicated().sum()}")
        
        if st.checkbox("Remove Duplicats"):
            keep_option = st.selectbox("Keep", [ "First",'Last','All'])

            if st.button("Remove Duplicates"):
                with st.spinner("Removing duplicates...."):
                    if keep_option == "First":
                        df.drop_duplicates(keep = "first", inplace=True)
                    elif keep_option  == "Last":
                        df.drop_duplicates(keep="last", inplace=True)
                    elif keep_option == "All":
                        df.drop_duplicates(keep=False , inplace= True)
                    st.success("Duplicates Removed")
                    st.write(f"Number of duplicate rows after removal : {df.duplicated().sum()}")



        if st.checkbox("Show missing values heatmap"):
            plt.figure(figsize= (10,6))
            sns.heatmap(df.isnull(), cbar = False , cmap = "viridis")
            st.pyplot(plt)

        if st.checkbox("Impute missing values"):
            imputation_info = {
                'Mean' : "Best for normally distributed numerical data.Avoid if data has outliers.",
                "Median": "Best for skewed numerical data and outliers. Avoid if data is normally distributed.",
                "Mode" : "Best for categorical data with frequent values.Avoid if categories are highly unique.",
                "Forward Fill": "Best for time_series with logical trends. Avoid if data changes are unpredictable.",
                "Backward Fill" : "Best for time-series where future values make sense. Avoid if data is highly dynamic.",
                "KNN Imputation": "Uses nearest neighbors to estimate missing values. Best when data has patterns.",
                "Regression Imputation": "Predicts missing values using regression on other features.",
                "Multiple Imputation (MICE)": "Best for small missing data, iteratively fills values for better estimation.",
                "Interpolation": "Fills missing values using trends between known values. Best for time-series.",
                "Listwise Deletion": "Removes rows with missing values. Use only when missing data is minimal."

            }
# dropdown for selecting imputation method
            imputation_method = st.selectbox('Imputation Method', list(imputation_info.keys()))
            st.info(imputation_info[imputation_method])

# dropdown for selecting columns
            missing_cols = df.columns[df.isnull().any()].tolist()  # shows only columns with missing values
            selected_cols = st.multiselect("Choose columns to impute.",["All"]+missing_cols)

            if st.button("Apply Imputation"):
                with st.spinner("Imputing missing values...."):
                    if "All" in selected_cols:
                        selected_cols = missing_cols
                    # applying imputation columns wise
                    for col in selected_cols:
                        if imputation_method == 'Mean':
                            if df[col].dtype in ['int64', 'float64']:
                                df[col].fillna(df[col].mean(), inplace=True)
                            else:
                                st.warning(f"Skipping column '{col}' - Mean imputation only works on numeric data.")


                        elif imputation_method == 'Median':
                            if df[col].dtype in ['int64', 'float64']:
                                df[col].fillna(df[col].median(), inplace = True)
                            else:
                                st.warning(f"Skipping column '{col}'- Median imputation only works on numeric columns.")

                        elif imputation_method == 'Mode':
                            for col in df.select_dtypes(include = 'object'):
                                df[col].fillna(df[col].mode()[0], inplace = True)
                            for col in df.select_dtypes(exclude = ['object']):
                                df[col].fillna(df[col].mode()[0], inplace = True)

                        elif imputation_method == 'Forward Fill':
                            df[col].fillna(method = 'ffill', inplace = True)

                        elif imputation_method == 'Backward Fill':
                            df[col].fillna(method = "bfill", inplace=  True)


                        elif imputation_method == 'KNN Imputation' and df[col].dtype != 'object':
                            imputer = KNNImputer(n_neighbors = 3)
                            num_cols = df[selected_cols].select_dtypes(include=np.number).columns
                            df[num_cols] = imputer.fit_transform(df[num_cols])

                        elif imputation_method == "Regression Imputation" and df[col].dtype != 'object':
                            known = df.dropna()
                            unknown = df[df[col].isnull()]
                            if len(unknown)> 0 and known.shape[0]>1:
                                X_train = pd.get_dummies(known.drop(columns = [col]),drop_first = True)
                                y_train = known[col]
                                X_test = pd.get_dummies(unknown.drop(columns = [col]),drop_first= True)
                                X_test = X_test.reindex(columns = X_train.columns, fill_value = 0)
                                model = LinearRegression()
                                model.fit(X_train,y_train)
                                df.loc[df[col].isnull(),col] = model.predict(X_test)
                        elif imputation_method == "Multiple Imputation (MICE)" and df[col].dtype != 'object':
                            imputer = IterativeImputer(max_iter = 10, random_state = 0)
                            num_cols = df[selected_cols].select_dtypes(include=np.number).columns
                            df[num_cols] = imputer.fit_transform(df[num_cols])
                        elif imputation_method == 'Interpolation' and df[col].dtype != 'object':
                            df[col].interpolate(method = 'linear', inplace = True)
                        elif imputation_method == "Listwise Deletion":
                            df.dropna(inplace = True)
                selected_cols = [str(col) for col in selected_cols]       # Ensure all column names are strings
                st.success(f"Imputation Applied: {imputation_method} on {', '.join(selected_cols)}")
                st.write("Updated Data")
                st.write(df)
            




            
        ## outlier handling and detection

        st.subheader("Outlier Handling")

        if st.checkbox("Show Boxplot"):
            for col in df.select_dtypes(include = np.number).columns:
                plt.figure()
                sns.boxplot(x = df[col])
                st.pyplot(plt)
# converting numeric columns if they are stored as a string

        for col in df.columns:
            try:
                df[col] = pd.to_numeric(df[col], errors = "ignore")
            except:
                pass
                # defining numerical df for outlier detection

        numerical_df = df.select_dtypes(include = [np.number])
        st.write("Detected Numeric Columns:", numerical_df.columns.tolist())

        if numerical_df.empty:
            st.warning("No numerical columns found for Z-score calculation. Please check data types!")
        

        if st.checkbox("Detect and handle outliers"):
            outlier_method = st.selectbox("Outlier detection method", ["IQR", "Z-Score"])
            
# for IQR 
            if outlier_method == "IQR":
                
                if not numerical_df.empty:
                    q1 = numerical_df.quantile(0.25)
                    q3 = numerical_df.quantile(0.75)
                    iqr = q3- q1
                    lower_bound = q1 - 1.5 * iqr
                    upper_bound = q3 + 1.5*iqr
                    outliers = (numerical_df< lower_bound) | (numerical_df> upper_bound)
                    st.write(f"Nummber of outliers (IQR): {outliers.sum().sum()}")
                else:
                    st.warning(f"No numerical columns found for IQR calculation.")
                    outliers = pd.DataFrame(index=df.index, columns = df.columns ,data = False)

# for z- score 

            elif outlier_method == "Z-Score":
                
                # st.write("Numeric Columns:", numerical_df.columns.tolist())
                if not numerical_df.empty:
                    z  = numerical_df.apply(lambda x: np.abs(stats.zscore(x, nan_policy="omit")))  # Keeps same shape
                    threshold = st.number_input("Z-score Threshold", min_value=1.0, value = 3.0 , step = 0.1)
                    outliers = z > threshold
                    outliers = outliers.reindex(df.index)
                    st.write(f"Number of outliers (Z_score): {outliers.sum().sum()}")
                else:
                    st.warning("No numerical columns found for Z-score calculation.")
                    outliers = pd.DataFrame(index=df.index, columns = df.columns ,data = False)
# handling outliers 


            if st.checkbox("Remove Outliers"):
                
                if st.button("Remove outliers"):
                    outliers = pd.DataFrame(outliers, index=df.index, columns=df.columns).astype(bool) 
                    with st.spinner("Removing Outliers..."):
                        if outliers.any().any():
                            rows_to_drop = outliers.any(axis = 1)
                            df = df.loc[~rows_to_drop]
                        
                            st.success("Outliers Removed!")
                            st.write(f"Shape of Dataframe after outlier removal:{df.shape}")

                        else:
                            st.warning("No outliers to remove.")


            if st.checkbox("Handling outliers"):
                replacement_info = {
                    "Mean": "Best for normally distributed numerical data. Avoid if data has extreme outliers.",
                    "Median": "Best for skewed numerical data and outliers. Avoid if data is normally distributed.",
                    "Winsorization": "Best for reducing the effect of extreme values. Avoid if preserving raw data is important.",
                    "Capping": "Best for handling extreme values using fixed percentile limits. Avoid if strict statistical assumptions are needed.",
                    "Log Transformation": "Best for reducing skewness in data. Avoid if data contains zero or negative values.",
                    "Standardization (Scaling)": "Best for normalizing data for machine learning models. Avoid if original scale is important."
                }

                ###  imputation_method = st.selectbox('Imputation Method', list(imputation_info.keys()))
            # st.info(imputation_info[imputation_method]) 

                replacement_method = st.selectbox("Handling Method", list(replacement_info.keys()))
                st.info(replacement_info[replacement_method])
                if st.button("Replace Outliers"):
                    with st.spinner("Replacing outlers..."):
                        numerical_cols = df.select_dtypes(include=np.number).columns

                        if replacement_method == "Median":
                            replacement_values = df[numerical_cols].median().fillna(0)
                        elif replacement_method == "Mean":
                            replacement_values = df[numerical_cols].mean().fillna(0)

                            for col in numerical_cols:
                                df.loc[outliers[col].astype(bool).values,col] = replacement_values[col]

                        elif replacement_method == "Winsorization":
                            df[numerical_cols] = df[numerical_cols].apply(lambda x: winsorize (x,limits =[0.05,0.05])if np.issubdtype(x.dtype , np.number) else x)

                        elif replacement_method == "Capping":
                            lower, upper = np.percentile(df[numerical_cols],[1,99], axis = 0)
                            df[numerical_cols] = np.clip(df[numerical_cols],lower,upper)

                        elif replacement_method == "Log Transformation":
                            df[numerical_cols] = df[numerical_cols].apply(lambda x:np.log1p(x) if (x > 0).all() else x)

                        elif replacement_method == "Standardization (Scaling)":
                            df[numerical_cols] = pd.DataFrame(StandardScaler().fit_transform(df[numerical_cols]),index = df.index , columns = numerical_cols)



                        st.success("Outliers replaced!")


# data inconsistency handling 
        st.subheader("Inconsistent String Data Handling")
        
        # getting the string columns
        string_cols = df.select_dtypes(include ="object").columns.tolist()

        if string_cols:
            all_option = "Select All"
            selected_string_cols = st.multiselect("Select string columns to clean:",[all_option] + string_cols)
            if selected_string_cols:
                if all_option in selected_string_cols:
                    selected_string_cols = string_cols
                else:
                    selected_string_cols = [col for col in selected_string_cols if col != all_option]

            if st.checkbox("Convert Case"):
                case_option = st.selectbox("Case Conversion", ["Lowercase","Uppercase","Title Case"])
                if st.button("Apply Case Conversion"):
                    with st.spinner("Applying case conversion..."):
                        for col in string_cols:
                            if case_option == "Lowercase":
                                df[col] = df[col].str.lower()
                            elif case_option == "Uppercase":
                                df[col] = df[col].str.upper()
                            elif case_option == "Title Case":
                                df[col] = df[col].str.title()
                        st.success("Case conversion applied")
            
            # removing whitespaces

            if st.checkbox("Remove Whitespace"):
                if st.button("Apply Whitespace Removal"):
                    with st.spinner("Removing Whitespacess...."):
                        whitespace_count = 0

                        for col in selected_string_cols:
                            whitespace_count += df[col].str.count(r"^\s+|\s+$").sum()
                            df[col] = df[col].str.strip()
                        st.success(f"({whitespace_count}) Whitespaces Removed!")

            # removing special characters

            if st.checkbox("Remove Special Characters/Emoji(Basic)"):
                if st.button("Apply Character Removal"):
                    with st.spinner("Removing Characters..."):
                        char_count = 0
                        for col in selected_string_cols:
                            char_count += df[col].str.count(r"[^a-zA-Z0-9\s]").sum()
                            df[col] = df[col].str.replace(r"[^a-zA-Z0-9\s]","",regex = True)
                        st.success(f"({char_count})Characters Removed!")

        else:
            st.info("No string columns found.")


# date handling 

        st.subheader("Date Handling")
        date_cols = df.select_dtypes(include = ["datetime64", "object"]).columns.tolist()

        if date_cols:
            all_option = "Select All"
            selected_date_cols = st.multiselect("Select date columns to handle:",[all_option]+ date_cols)

            if selected_date_cols:
                if all_option in selected_date_cols:
                    selected_date_cols = date_cols
                else:
                    selected_date_cols = [col for col in selected_date_cols if col != all_option]

                for col in selected_date_cols:
                    try:
                        df[col] = pd.to_datetime(df[col], errors = "raise")
                        st.success(f"Column '{col}' converted to datetime.")
                    except ValueError:
                        st.warning(f"Column '{col}' could not be automatically converted.Please specify the format manually.")

                        date_format = st.text_input(f"Enter the date format for '{col}':")
                        if date_format:
                            try:
                                df[col] = pd.to_datetime(df[col],format = date_format,errors = "raise")
                                st.success(f"Column '{col}' converted using format '{date_format}'.")
                            except ValueError:
                                st.error(f"Invalid date format for '{col}'")
                if st.checkbox("Extract date features"):
                            for col in selected_date_cols:
                                if pd.api.types.is_datetime64_any_dtype(df[col]):
                                    df[f"{col}_year"] = df[col].dt.year
                                    df[f"{col}_month"] = df[col].dt.month
                                    df[f"{col}_day"] = df[col].dt.day
                                    df[f"{col}_weekday"] = df[col].dt.day_name()
                                    st.success(f"Date features extracted from '{col}'.")
                                else:
                                    st.warning(f"Column '{col}' is not in datetime format. Features cannot be extracted.")

                if st.checkbox("Handle missing dates"):
                    imputation_method = st.selectbox("Missing date imputation method", ["Forward Fill", "Backward Fill", "Remove Rows"])
                    if st.button("Apply Date Imputation"):
                        with st.spinner("Imputing missing dates..."):
                            for col in selected_date_cols:
                                if pd.api.types.is_datetime64_any_dtype(df[col]):
                                    if imputation_method == "Forward Fill":
                                        df[col].fillna(method="ffill", inplace=True)
                                    elif imputation_method == "Backward Fill":
                                        df[col].fillna(method="bfill", inplace=True)
                                    elif imputation_method == "Remove Rows":
                                        df.dropna(subset=[col], inplace=True)
                                    st.success(f"Missing dates in '{col}' handled.")
                                else:
                                    st.warning(f"Column '{col}' is not in datetime format. Imputation not possible.")
                else:
                    st.info("No date columns found.")

# data type correction

        st.subheader("Data Type Correction (Multiple Columns)")

        st.write("Current Data Types:")
        st.write(df.dtypes)

        cols_to_convert = st.multiselect("Select columns to convert:", df.columns.tolist())

        if cols_to_convert:
            target_types = ["int64", "float64", "object", "datetime64", "bool"]
            target_type_dict = {}

            for col in cols_to_convert:
                target_type_dict[col] = st.selectbox(f"Select target type for '{col}':", target_types, key=f"target_{col}")

            if st.button("Convert Selected Columns"):
                with st.spinner("Converting data types..."):
                    for col in cols_to_convert:
                        target_type = target_type_dict[col]
                        try:
                            if target_type == "datetime64":
                                df[col] = pd.to_datetime(df[col], errors="raise")
                            elif target_type == "int64":
                                df[col] = pd.to_numeric(df[col], errors='coerce').astype('Int64')
                            elif target_type == "float64":
                                df[col] = pd.to_numeric(df[col], errors='coerce')
                            elif target_type == "bool":
                                df[col] = df[col].astype('bool')
                            else:
                                df[col] = df[col].astype(target_type)  # Default to astype for other types

                            st.success(f"Column '{col}' converted to '{target_type}'.")
                        except ValueError as e:
                            st.error(f"Error converting column '{col}': {e}")
                        except Exception as e:
                            st.error(f"An unexpected error occurred during conversion of '{col}': {e}")




# encoding................

        st.subheader("Encoding Categorical Variables (Multiple Columns)")

        categorical_cols = df.select_dtypes(include = ["object"]).columns.tolist()

        if categorical_cols:
            all_option = "Select All"
            selected_cat_cols = st.multiselect("Select categorical columns to encode",[all_option]+categorical_cols)
            if selected_cat_cols:
                if all_option in selected_cat_cols:
                    selected_cat_cols = categorical_cols
                else:
                    selected_cat_cols = [col for col in selected_cat_cols if col != all_option]

            if selected_cat_cols:
                encoding_methods = ["One-Hot Encoding", "Label Encoding","Frequency Encoding"]

                encoding_method_dict = {}
                for col in selected_cat_cols:
                    encoding_method_dict[col] = st.selectbox(f"Select encoding method for '{col}':", encoding_methods, key=f"encoding_{col}_{id(encoding_method_dict)}")


                if st.button("Apply Encoding"):
                    with st.spinner("Applying Encodings..."):
                        for col in selected_cat_cols:
                            encoding_methods = encoding_method_dict[col]


                            if encoding_methods == "One-Hot Encoding":
                                df = pd.get_dummies(df, columns = [col],drop_first= True)
                                st.success(f"One-Hot Encoding applied to '{col}'.")

                            elif encoding_methods == "Label Encoding":
                                le = LabelEncoder()
                                df[col] = le.fit_transform([df[col]])
                                st.success(f"Label Encoding applied to '{col}'.")
                            elif encoding_methods == "Frequency Encoding":
                                frequency = df[col].value_counts(normalize = True)
                                df[col] = df[col].map(frequency)
                                st.success(f"Frequency Encoding applied to '{col}'.")
        

            else:
                st.info("No categorical columns found.")

# Feature Scaling/Normalization


                st.subheader("Feature Scaling/Normalization")

        numerical_cols = df.select_dtypes(include=["number"]).columns.tolist()

        if numerical_cols:
            all_option = "Select All"
            selected_num_cols = st.multiselect("Select numerical columns to scale/normalize:", [all_option] + numerical_cols)

            if selected_num_cols:
                if all_option in selected_num_cols:
                    selected_num_cols = numerical_cols  # Select all columns if "Select All" is chosen
                else:
                    selected_num_cols = [col for col in selected_num_cols if col != all_option]

                if selected_num_cols:
                    scaling_methods = ["StandardScaler", "MinMaxScaler", "RobustScaler"]
                    scaling_method_dict = {}

                    for col in selected_num_cols:
                        scaling_method_dict[col] = st.selectbox(f"Select scaling method for '{col}':", scaling_methods, key=f"scaling_{col}")

                    if st.button("Apply Scaling/Normalization"):
                        with st.spinner("Applying scaling/normalization...."):
                            for col in selected_num_cols:
                                scaling_methods = scaling_method_dict

                                scaling_method = scaling_method_dict[col]

                                if scaling_method == "StandardScaler":
                                    scaler = StandardScaler()
                                    df[col] = scaler.fit_transform(df[[col]])
                                    st.success(f"StandardScaler applied to '{col}'.") # Success message for each column

                                elif scaling_method == "MinMaxScaler":
                                    scaler = MinMaxScaler()
                                    df[col] = scaler.fit_transform(df[[col]])
                                    st.success(f"MinMaxScaler applied to '{col}'.") # Success message for each column

                                elif scaling_method == "RobustScaler":
                                    scaler = RobustScaler()
                                    df[col] = scaler.fit_transform(df[[col]])
                                    st.success(f"RobustScaler applied to '{col}'.") 

        else:
            st.info("Please select numerical columns to scale/normalize")
                

# download the cleaned file 
        st.session_state.df = df.copy()
        st.subheader("Download cleaned file")

        
        if st.button("Download Cleaned CSV"):
            csv = df.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name="cleaned_data.csv",
                mime="text/csv",
            )






    except pd.errors.ParserError:
        st.error("Invalid CSV file")
    except Exception as e:
        st.error(f"An error occured: {e}")










