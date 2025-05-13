import os
import time
import pandas as pd
import cohere
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from dotenv import load_dotenv

# === I. Load environment variables (e.g., COHERE_API_KEY) ===
load_dotenv()
COHERE_API_KEY = os.getenv("COHERE_API_KEY")

# === II. Connect to Cohere ===
co = cohere.Client(COHERE_API_KEY)

# === III. Load Member Conversations ===
def load_member_conversations(path):
    conversations = []
    for file in os.listdir(path):
        if file.endswith('.txt'):
            with open(os.path.join(path, file), 'r', encoding='utf-8') as f:
                text = f.read()
                member_lines = [line.split(":", 1)[1].strip()
                                for line in text.splitlines()
                                if line.strip().lower().startswith("member:")]
                member_text = " ".join(member_lines)
                conversations.append({'filename': file, 'member_text': member_text})
    return pd.DataFrame(conversations)

# === IV. Classification Function using Cohere Chat ===
def classify_text_with_chat(text):
    try:
        prompt = f"""
You are a helpful assistant for an insurance call center. Please analyze the following customer dialogue and determine:
1. The overall sentiment (positive, negative, or neutral)
2. The outcome of the call (issue resolved or follow-up action needed)

Customer says:
\"\"\"{text}\"\"\"

Respond in the following format:
Sentiment: <positive/negative/neutral>
Call Outcome: <issue resolved/follow-up action needed>
"""
        response = co.chat(
            message=prompt,
            model='command-r-plus',
            temperature=0.3
        )
        return response.text.strip()
    except Exception as e:
        return f"Error: {e}"

# === V. Split DataFrame into Chunks ===
def split_dataframe(df, chunk_size=20):
    return [df.iloc[i:i + chunk_size].copy() for i in range(0, len(df), chunk_size)]

# === VI. Process Transcripts and Extract Sentiment/Outcome ===
def process_chunks(chunks):
    processed_chunks = []
    for i, chunk in enumerate(chunks):
        print(f"\n Processing chunk {i+1}/{len(chunks)}")
        results = []
        for text in tqdm(chunk['member_text'], desc=f"Chunk {i+1}"):
            result = classify_text_with_chat(text)
            results.append(result)
            time.sleep(1.5)
        chunk['response'] = results
        chunk['sentiment'] = chunk['response'].str.extract(r"Sentiment:\s*(\w+)", expand=False).str.lower()
        chunk['call_outcome'] = chunk['response'].str.extract(r"Call Outcome:\s*(.*)", expand=False).str.lower()
        processed_chunks.append(chunk)
    return pd.concat(processed_chunks, ignore_index=True)

# === VII. Visualise Results ===
def plot_results(df):
    sns.countplot(data=df, x='sentiment')
    plt.title('Sentiment Distribution Across Conversations')
    plt.savefig("output/sentiment_distribution.png")
    plt.clf()

    sns.countplot(data=df, x='call_outcome')
    plt.title('Call Outcome Distribution')
    plt.savefig("output/outcome_distribution.png")
    plt.clf()

# === VIII. Main Execution ===
def main():
    input_path = "transcripts_v3"
    output_path = "output/analyzed_transcripts.csv"

    os.makedirs("output", exist_ok=True)

    print("Loading transcripts...")
    df = load_member_conversations(input_path)

    print("Splitting into chunks...")
    chunks = split_dataframe(df, chunk_size=20)

    print("Classifying conversations...")
    final_df = process_chunks(chunks)

    print("Saving results...")
    final_df.to_csv(output_path, index=False)

    print("Generating visualisations...")
    plot_results(final_df)

    print("Process complete. Results saved to output/")

if __name__ == "__main__":
    main()
