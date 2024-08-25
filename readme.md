# Streamlit Enrolments App

This project is a Streamlit web application that allows users to ask questions about student enrolments based on a dataset provided by Snowflake. The app uses a fine-tuned BERT model for question answering.

## Features

- Load and preprocess student enrolment data from Snowflake using a random seed.
- Manually generate question-answer pairs from 50 rows of data.
- Fine-tune or use a pre-trained BERT model to answer questions.
- Display 30 random rows of data and provide answers based on user queries.

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/Yash-mq/streamlit-enrolments-app.git
    ```

2. Navigate to the project directory:
    ```bash
    cd streamlit-enrolments-app
    ```

3. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

4. Set up the `.env` file in the root directory:
    ```bash
    touch .env
    ```

    Add the following content to the `.env` file:
    ```bash
    SNOWFLAKE_USER=your_snowflake_user
    SNOWFLAKE_PASSWORD=your_snowflake_password
    SNOWFLAKE_ACCOUNT=your_snowflake_account
    SNOWFLAKE_WAREHOUSE=your_snowflake_warehouse
    SNOWFLAKE_DATABASE=your_snowflake_database
    SNOWFLAKE_SCHEMA=your_snowflake_schema
    ```

## Steps to Prepare Data and Use the App

1. **Manually Fetch 50 Rows from Snowflake (Using Seed 42)**:
    - The app fetches 50 random rows of student enrolment data from Snowflake using the following query with a random seed of 42:
      ```sql
      SELECT YEAR, MONTH, REGION, NATIONALITY, LEVEL_OF_STUDY, BROAD_FIELD, STATE, ENROLMENTS 
      FROM STUDENT_ENROLMENTS_DB.ENROLMENTS_DATA.INTERNATIONAL_STUDENT_ENROLMENTS_V2
      ORDER BY RANDOM(42)
      LIMIT 50;
      ```
    - You can also manually execute this query in Snowflake to fetch the 50 rows.

2. **Manually Generate Question-Answer Pairs**:
    - After fetching the 50 rows of data, you need to generate question-answer pairs manually.
    - For example:
      - How many enrolments were there from a specific nationality?
      - Which state had the enrolments?
    - Once generated, paste these question-answer pairs into the `combined_question_answers.json` file.

3. **Train or Load the Model**:
    - Once the JSON file is populated with question-answer pairs, you can train the model.
    - You can choose to retrain the model using the new data or load an existing pre-trained model.
    - If you select **Retrain Model**, the app will fine-tune a BERT model on the question-answer pairs you provided in the JSON.
    - If you select **Use Pre-trained Model**, it will load an existing fine-tuned model if available.

4. **Display 30 Rows of Data**:
    - After training or loading the model, the app will display 30 random rows of data fetched from the Snowflake dataset. These rows will be used for answering user questions.

5. **Ask a Question**:
    - Once the data is displayed, input your question related to student enrolments. For example:
      - "How many enrolments from China nationality in NSW state?"
      - "How many enrolments from Bachelor Degree level of study?"
    - The app will filter the data and provide an answer based on the user's query.

## Usage

1. Start the Streamlit app:
    ```bash
    streamlit run main.py
    ```

2. Follow the steps in the web interface:
    - Manually fetch 50 rows of data from Snowflake using the provided SQL query with random seed 42.
    - Generate question-answer pairs and paste them into `combined_question_answers.json`.
    - Select whether to use a pre-trained model or retrain the model.
    - Submit your choice and refresh the data.
    - Input your question, and the model will generate an answer based on the provided conditions using 30 rows of data.

## Data Credits

The student enrolment data used in this project is provided by [Austrade's Student Data](https://education.austrade.gov.au/data-download-student-data). 

Please ensure to give appropriate credit if using the data in your own projects.

## Model

This project uses a fine-tuned BERT model (`bert-large-uncased-whole-word-masking-finetuned-squad`) for the question-answering task. The model can be retrained using the dataset generated from the Snowflake data.

## Contributing

Contributions are welcome! Feel free to submit issues or pull requests to help improve the project.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
