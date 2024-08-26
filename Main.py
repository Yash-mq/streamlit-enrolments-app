import os
import json
import pandas as pd
from transformers import BertTokenizerFast, BertForQuestionAnswering, Trainer, TrainingArguments, pipeline
from datasets import Dataset, DatasetDict, concatenate_datasets
import snowflake.connector
import streamlit as st
import re
from dotenv import load_dotenv


# Initialize session state for model_ready
if 'model_ready' not in st.session_state:
    st.session_state.model_ready = False

# Step 1: Load Pre-existing JSON Data
with open('combined_question_answers.json', 'r') as f:
    pre_generated_data = json.load(f)

qa_data_from_json = []
for entry in pre_generated_data:
    qa_data_from_json.append(entry)

# Convert JSON to Dataset format
json_contexts = []
json_questions = []
json_answers = []

for entry in qa_data_from_json:
    context = entry['context']
    for qa in entry['qas']:
        json_contexts.append(context)
        json_questions.append(qa['question'])
        json_answers.append({
            'text': qa['answers'][0]['text'],
            'answer_start': qa['answers'][0]['answer_start']
        })

json_dataset = Dataset.from_dict({
    'context': json_contexts,
    'question': json_questions,
    'answers': json_answers
})

# Load environment variables from .env file
load_dotenv()

# Step 2: Extract Data from Snowflake with Seed 42 using environment variables
conn = snowflake.connector.connect(
    user=os.getenv('SNOWFLAKE_USER'),
    password=os.getenv('SNOWFLAKE_PASSWORD'),
    account=os.getenv('SNOWFLAKE_ACCOUNT'),
    warehouse=os.getenv('SNOWFLAKE_WAREHOUSE'),
    database=os.getenv('SNOWFLAKE_DATABASE'),
    schema=os.getenv('SNOWFLAKE_SCHEMA')
)

query = """
    SELECT YEAR, MONTH, REGION, NATIONALITY, LEVEL_OF_STUDY, 
           BROAD_FIELD, STATE, ENROLMENTS 
    FROM STUDENT_ENROLMENTS_DB.ENROLMENTS_DATA.INTERNATIONAL_STUDENT_ENROLMENTS_V2
    ORDER BY RANDOM(42)
    LIMIT 50;
"""
df = pd.read_sql(query, conn)

# Step 3: Generate Question-Answer Pairs from Snowflake Seed 42 Data

qa_data_from_snowflake = []
for _, row in df.iterrows():
    context = f"In {row['YEAR']}, there were {row['ENROLMENTS']} enrolments from {row['NATIONALITY']} in {row['STATE']} for a {row['LEVEL_OF_STUDY']} in {row['BROAD_FIELD']}."

    qas = [
        {
            "question": f"How many enrolments were there in {row['YEAR']} for {row['LEVEL_OF_STUDY']} in {row['BROAD_FIELD']}?",
            "answers": [{"text": str(row['ENROLMENTS']), "answer_start": context.index(str(row['ENROLMENTS']))}]},
        {"question": f"Which nationality had the enrolments in {row['YEAR']} for {row['LEVEL_OF_STUDY']}?",
         "answers": [{"text": row['NATIONALITY'], "answer_start": context.index(row['NATIONALITY'])}]},
        {"question": f"Which state had the enrolments in {row['YEAR']}?",
         "answers": [{"text": row['STATE'], "answer_start": context.index(row['STATE'])}]},
        {"question": f"What was the level of study for enrolments in {row['YEAR']}?",
         "answers": [{"text": row['LEVEL_OF_STUDY'], "answer_start": context.index(row['LEVEL_OF_STUDY'])}]},
        {"question": f"What was the field of study for enrolments in {row['YEAR']}?",
         "answers": [{"text": row['BROAD_FIELD'], "answer_start": context.index(row['BROAD_FIELD'])}]}
    ]

    qa_data_from_snowflake.append({"context": context, "qas": qas})

# Convert Snowflake data to Dataset format
snowflake_contexts = []
snowflake_questions = []
snowflake_answers = []

for entry in qa_data_from_snowflake:
    context = entry['context']
    for qa in entry['qas']:
        snowflake_contexts.append(context)
        snowflake_questions.append(qa['question'])
        snowflake_answers.append({
            'text': qa['answers'][0]['text'],
            'answer_start': qa['answers'][0]['answer_start']
        })

snowflake_dataset = Dataset.from_dict({
    'context': snowflake_contexts,
    'question': snowflake_questions,
    'answers': snowflake_answers
})

# Combine the new dataset with your old JSON dataset for training
combined_dataset = concatenate_datasets([json_dataset, snowflake_dataset])

# Model directory
model_dir = './fine_tuned_model'

# Step 5: Add Model Selection and Submit Button
model_choice = st.radio("Choose Model", options=["Use Pre-trained Model", "Retrain Model"])

# Mapping from question words to DataFrame column names
question_column_map = {
    "nationality": "NATIONALITY",
    "state": "STATE",
    "level of study": "LEVEL_OF_STUDY",
    "broad field": "BROAD_FIELD",
    "year": "YEAR",
    "region": "REGION",
    "enrolments": "ENROLMENTS"
}


# Function to extract column and value from user's question
def extract_column_and_value(question):
    # Loop through the possible columns and check if any are mentioned in the question
    for key, column in question_column_map.items():
        if key in question.lower():
            # Extract the value the user is querying for, e.g., "Hong Kong" or "Diploma"
            match = re.search(rf"from (.*?) {key}", question.lower())
            if match:
                return column, match.group(1).strip()

    return None, None


if st.button("Submit"):
    if model_choice == "Use Pre-trained Model" and os.path.exists(model_dir):
        # Load pre-trained model
        model = BertForQuestionAnswering.from_pretrained(model_dir)
        tokenizer = BertTokenizerFast.from_pretrained(model_dir)
        qa_pipeline = pipeline("question-answering", model=model, tokenizer=tokenizer)
        st.write("Using the pre-trained model.")
        st.session_state.model_ready = True  # Set model ready in session state
        st.session_state.qa_pipeline = qa_pipeline  # Save the pipeline in session state

    elif model_choice == "Retrain Model":
        st.write("Training the model...")

        # Preprocess Function
        tokenizer = BertTokenizerFast.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')


        def preprocess_function(examples):
            # Tokenize the inputs and keep the offsets
            inputs = tokenizer(
                examples['question'],
                examples['context'],
                max_length=512,
                truncation=True,
                padding="max_length",
                return_offsets_mapping=True
            )

            start_positions = []
            end_positions = []
            valid_indices = []

            for i, offset in enumerate(inputs["offset_mapping"]):
                answer = examples['answers'][i]
                start_char = answer["answer_start"]
                end_char = start_char + len(answer["text"])

                start_idx = None
                end_idx = None

                # Find the start and end indices for the answer
                for idx, (start, end) in enumerate(offset):
                    if start <= start_char < end:
                        start_idx = idx
                    if start <= end_char <= end:
                        end_idx = idx
                        break

                # Check if both start and end indices are found
                if start_idx is not None and end_idx is not None:
                    start_positions.append(start_idx)
                    end_positions.append(end_idx)
                    valid_indices.append(i)

            # Filter valid examples
            if valid_indices:
                filtered_inputs = {key: [value[i] for i in valid_indices] for key, value in inputs.items()}
                filtered_inputs["start_positions"] = start_positions
                filtered_inputs["end_positions"] = end_positions
                filtered_inputs.pop("offset_mapping")  # Remove offset mapping
                return filtered_inputs
            else:
                return {}  # Return an empty dict if no valid indices are found


        # Split dataset
        dataset_dict = DatasetDict({'train': combined_dataset})
        split_datasets = dataset_dict['train'].train_test_split(test_size=0.1, seed=42)

        train_dataset = split_datasets['train']
        eval_dataset = split_datasets['test']

        train_dataset = train_dataset.map(preprocess_function, batched=True,
                                          remove_columns=combined_dataset.column_names)
        eval_dataset = eval_dataset.map(preprocess_function, batched=True, remove_columns=combined_dataset.column_names)

        # Fine-tune Model (reduced epochs)
        model = BertForQuestionAnswering.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')

        training_args = TrainingArguments(
            output_dir='./results',
            evaluation_strategy="epoch",
            learning_rate=2e-5,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=8,
            num_train_epochs=2,  # Reduced epochs
            weight_decay=0.01
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset
        )

        trainer.train()

        # Save Model and Tokenizer
        model.save_pretrained(model_dir)
        tokenizer.save_pretrained(model_dir)

        # Define qa_pipeline after training
        qa_pipeline = pipeline("question-answering", model=model, tokenizer=tokenizer)

        st.write("Training done! Now, please refresh the data.")
        st.session_state.model_ready = True  # Set model ready in session state
        st.session_state.qa_pipeline = qa_pipeline  # Save the pipeline in session state

    else:
        st.write("Model directory does not exist. Please train your model first.")

# Step 6: Add a Button for Data Refresh
if st.button('Refresh Data'):
    # Fetch New Random Data from Snowflake
    query_random = """
        SELECT YEAR, MONTH, REGION, NATIONALITY, LEVEL_OF_STUDY, 
               BROAD_FIELD, STATE, ENROLMENTS 
        FROM STUDENT_ENROLMENTS_DB.ENROLMENTS_DATA.INTERNATIONAL_STUDENT_ENROLMENTS_V2
        ORDER BY RANDOM()
        LIMIT 30;
    """
    df_random = pd.read_sql(query_random, conn)
    st.session_state['df_random'] = df_random  # Save data in session state

# Ensure data doesn't refresh every time unless explicitly refreshed
df_random = st.session_state.get('df_random', pd.DataFrame())

# Step 7: Display the Data in Streamlit
st.title('Preview Data')
st.write(df_random)

# Step 8: Ask a Question Based on Previewed Data
st.subheader('Ask a question:')
user_question = st.text_input("Enter your question:")


# Function to extract multiple conditions from the user's question
def extract_conditions(question):
    conditions = {}

    # Define patterns to extract relevant information for each column
    patterns = {
        "NATIONALITY": r"\bfrom\b\s+(\w+)",
        "STATE": r"\bin\b\s+(\w+)",
        "LEVEL_OF_STUDY": r"\b(?:degree|diploma|certificate|level of study)\b\s*(\w+\s?\w*)",
        "BROAD_FIELD": r"\bin\b\s*(\w+\s?\w*)\s*(field|broad field)",
        "YEAR": r"\b(?:in|during)\s*(\d{4})",
        "REGION": r"\bregion\b\s*(\w+\s?\w*)",
        "ENROLMENTS": r"(\d+)\s*enrolments"
    }

    # Iterate through the patterns to look for matches in the question
    for column, pattern in patterns.items():
        match = re.search(pattern, question.lower())
        if match:
            # Extract the value and assign it to the corresponding column in conditions
            conditions[column] = match.group(1).strip()

    return conditions


if st.button('Ask') and not df_random.empty:
    if st.session_state.model_ready:  # Ensure model is ready before using qa_pipeline
        qa_pipeline = st.session_state.qa_pipeline  # Retrieve the pipeline from session state

        # Extract conditions (e.g., nationality, state)
        conditions = extract_conditions(user_question)

        # Start with the full DataFrame
        filtered_df = df_random

        # Apply all conditions
        for column, value in conditions.items():
            filtered_df = filtered_df[filtered_df[column].str.lower() == value.lower()]

        # Sum the total enrolments for the filtered rows
        total_enrolments = filtered_df['ENROLMENTS'].sum()

        # Debugging: Print the result
        st.write(f"Total Enrolments for conditions {conditions}: {total_enrolments}")

        # Create context for the QA model
        context = f"There were a total of {total_enrolments} enrolments for the given conditions."

        # You can still use the model to answer if needed
        result = qa_pipeline({'question': user_question, 'context': context})

        st.write(result['answer'])
    else:
        st.write("Please load or train the model first.")
