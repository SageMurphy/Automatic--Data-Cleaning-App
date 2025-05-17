# No-Code Data Cleaning App

This Streamlit application provides an interactive interface for cleaning and preprocessing CSV datasets. It includes features for handling missing values, duplicates, outliers, inconsistent string data, date handling, data type conversion, encoding categorical variables, and feature scaling/normalization.

## Features

-   **File Information:** Displays basic information about the uploaded CSV file.
-   **Data Profiling:** Generates a comprehensive data profile report using `ydata_profiling`.
-   **Duplicate Handling:** Removes duplicate rows with options to keep first, last, or no duplicates.
-   **Missing Value Imputation:** Offers various imputation methods (mean, median, mode, KNN, regression, etc.).
-   **Outlier Handling:** Detects and handles outliers using IQR or Z-score methods.
-   **String Data Cleaning:** Converts case, removes whitespace, and removes special characters.
-   **Date Handling:** Converts columns to datetime, extracts date features, and imputes missing dates.
-   **Data Type Correction:** Allows changing the data type of columns.
-   **Categorical Encoding:** Provides options for one-hot encoding, label encoding, and frequency encoding.
-   **Feature Scaling/Normalization:** Offers StandardScaler, MinMaxScaler, and RobustScaler.
-   **Download Cleaned File:** Downloads the cleaned DataFrame as a CSV file.

Preview :
![{964F5D7E-7EF4-473F-A416-41F097063033}](https://github.com/user-attachments/assets/e92743ee-3f90-44c4-b769-cddee8debc9c)


## Installation

1.  Clone the repository:

    ```bash
    git clone https://github.com/SageMurphy/Atuomatic-Data-Cleaning-App
    ```

2.  Navigate to the project directory:

    ```bash
    cd <project_directory>
    ```

3.  Create a virtual environment (recommended):

    ```bash
    python -m venv venv
    ```

4.  Activate the virtual environment:

    -   On Windows: `venv\Scripts\activate`
    -   On macOS/Linux: `source venv/bin/activate`

5.  Install the required packages:

    ```bash
    pip install -r requirements.txt
    ```

## Usage

1.  Run the Streamlit app:

    ```bash
    streamlit run app.py
    ```

2.  Open the application in your web browser (usually `http://localhost:8501`).
3.  Upload a CSV file and use the interactive interface to clean your data.
4.  Download the cleaned file.

## Dependencies

-   streamlit
-   pandas
-   numpy
-   seaborn
-   matplotlib
-   scikit-learn
-   scipy
-   ydata_profiling

## Contributing

Contributions are welcome! Please feel free to submit a pull request.

## License

[MIT License]

## Author 

Abhishek Shrimali
