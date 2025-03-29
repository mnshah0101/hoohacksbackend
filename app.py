from supabase import create_client
from langchain.callbacks.base import BaseCallbackHandler
import io
from datetime import datetime
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from flask import Flask, request, Response, stream_with_context, jsonify
from config import AppConfig
from botocore.exceptions import ClientError, NoCredentialsError
from langchain_anthropic import ChatAnthropic
from backtester import BacktestRunner
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

import boto3
import pandas as pd
import numpy as np
import boto3
import json
import os
import dotenv
import queue
import threading

dotenv.load_dotenv()    

app = Flask(__name__)

# Initialize Supabase client and S3 client
supabase = create_client(AppConfig.SUPABASE_URL, AppConfig.SUPABASE_KEY)

def analyze_intent(chat_message, chat_history):
    """
    Analyze the chat message and history using LangChain to determine the intent.
    It returns one of: 'backtest', 'question', or 'conversation'.
    """
    # Prepare the chat history as a string (if it's a list, join it)
    if isinstance(chat_history, list):
        history_str = "\n".join(chat_history)
    else:
        history_str = str(chat_history)

    prompt_template = (
        "You are an expert in financial backtesting and natural language understanding. "
        "Your task is to classify the following user message into one of three categories:\n"
        "1. 'backtest' - when the user is instructing the system to run a backtest (e.g., specifying strategy parameters).\n"
        "2. 'question' - when the user is asking for details or clarification about a backtest (e.g., trade details, performance, equity curve image).\n"
        "3. 'conversation' - when the user is simply having a general conversation.\n\n"
        "Only respond with one of these exact words: backtest, question, or conversation.\n\n"
        "Chat History Context:\n{chat_history}\n\n"
        "User Message:\n{chat_message}\n\n"
        "Classification:"
    )

    prompt = PromptTemplate(
        input_variables=["chat_message", "chat_history"],
        template=prompt_template
    )

    # Instantiate the Anthropic LLM via LangChain. Ensure your API key is set in Config.
    print(AppConfig.ANTHROPIC_API_KEY)
    llm = ChatAnthropic(api_key=AppConfig.ANTHROPIC_API_KEY, model="claude-3-5-haiku-20241022")
    chain = LLMChain(llm=llm, prompt=prompt)

    output = chain.run({"chat_message": chat_message,
                       "chat_history": history_str})
    # Clean and validate the output
    intent = output.strip().lower()
    if intent not in ["backtest", "question", "conversation"]:
        intent = "conversation"  # Default to conversation if the result is unexpected
    return intent


def run_backtest_command(chat_message, chat_history, conversation_id):
    """
    Run a backtest based on the user's command extracted from the chat message.
    """
    prompt_template = """
    You are an expert in trading strategy backtesting. Parse the following user message, chat history, and conversation ID to extract the following information in valid JSON format:
    - "conversation_id": the provided conversation ID.
    - "strategy_type": one of "ma_crossover", "rsi", "bollinger", "mean_reversion_simple", "mean_reversion_zscore", "macd", "trend_following_adx", "breakout", "momentum", or "vwap".
    - "strategy_params": a JSON object containing any strategy-specific parameters ("short_window", "long_window", "rsi_period", "period", "std_multiplier", "threshold", "zscore_threshold", "fast_period", "slow_period", "signal_period", "adx_period", "adx_threshold", "breakout_window", "momentum_period", "momentum_threshold", "vwap_period").
    - "money_management_params": a JSON object containing money management parameters ("stop_loss", "take_profit", "risk_fraction", "max_position_siz).

    User message: {chat_message}
    Chat history: {chat_history}
    Conversation ID: {conversation_id}

    Only output valid JSON.
    """

    prompt = PromptTemplate(
        input_variables=["chat_message", "chat_history", "conversation_id"],
        template=prompt_template
    )
    
    llm = ChatAnthropic(api_key=AppConfig.ANTHROPIC_API_KEY,
                        model="claude-3-5-haiku-20241022")
    chain = LLMChain(llm=llm, prompt=prompt)

    response = chain.run(chat_message=chat_message, chat_history=chat_history, conversation_id=conversation_id)

    try:
        parameters = json.loads(response)
    except json.JSONDecodeError:
        # Fallback
        parameters = {
            "conversation_id": conversation_id,
            "strategy_type": "ma_crossover",
            "strategy_params": {},
            "money_management_params": {}
        }

    dates = pd.date_range('2020-01-01', periods=200)
    prices = np.cumsum(np.random.randn(200)) + 100
    volume = np.random.randint(100, 1000, size=200)
    data = pd.DataFrame({
        'Date': dates,
        'Close': prices,
        'Open': prices,
        'Low': prices,
        'High': prices,
        'Volume': volume
    }).set_index('Date')

    runner = BacktestRunner(data)

    result = runner.run_backtest(
        parameters["strategy_type"],
        parameters.get("strategy_params", {}),
        parameters.get("money_management_params", {})
    )

    images = plot_backtest_results(result, conversation_id)
    
    
    return result, images


def upload_backtest_result(backtest_result, images,conversation_id):
    """
    Upload a backtest result to the database.
    Converts non-serializable parts (like stats objects) into JSON-serializable types.
    """
    # Convert the stats to a JSON-serializable dict. If stats is a namedtuple,
    # you might also try: results = dict(backtest_result.get("results", {}))
    results = json.loads(backtest_result.to_json() )



    equity_curve = json.dumps(results['_equity_curve'])
    trades = json.dumps(results['_trades'])
    results_upload = json.dumps(results)

    print(images)


   

    # Insert the serialized JSON data into the database.
    supabase.table("backtest_results").insert({
        "conversation_id": conversation_id,
        "results": results_upload,
        "equity_curve": equity_curve,
        "trades": trades,
        "drawdown_curve": images['drawdown_curve'],
        "equity_curve_image": images['equity_curve'],
        "trade_return_histogram": images['trade_return_histogram'],
        "trade_duration_vs_return": images['trade_duration_vs_return']
    }).execute()

    return True


def plot_backtest_results(results, conversation_id):
    """
    Generate and upload plots from backtest results as image URLs.

    Parameters:
        results (dict): A dictionary containing backtest results with keys such as:
            - '_equity_curve': A dict with timestamp keys and dict values containing 'Equity' and 'DrawdownPct'
            - '_trades': A dict where each trade has keys 'ReturnPct' and 'Duration'
        conversation_id (str): The id of the conversation to make the image file names unique.
        
    Returns:
        dict: A dictionary containing the S3 URLs for each plot.
              Keys: 'equity_curve', 'drawdown_curve', 'trade_return_histogram', 'trade_duration_vs_return'
    """
    images = {}

    results = json.loads(results.to_json())

    print(results)

    # Helper function to write image bytes to a file, upload it, then remove the file.
    def save_and_upload(image_bytes, file_suffix):
        file_name = f"{conversation_id}_{file_suffix}.png"
        with open(file_name, "wb") as f:
            f.write(image_bytes)
        # Upload the file to S3; assumes store_image_in_s3 is defined elsewhere.
        url = store_image_in_s3(file_path=file_name)
        os.remove(file_name)
        return url

    # -------------------------------
    # 1. Equity Curve and Drawdown Plot
    # -------------------------------
    equity_curve = results['_equity_curve']
    
    dates = [datetime.fromtimestamp(int(ts) / 1000)
             for ts in equity_curve.keys()]
    equity_values = [data_point['Equity']
                     for data_point in equity_curve.values()]
    drawdown_pct = [data_point['DrawdownPct'] *
                    100 for data_point in equity_curve.values()]  # Convert to percentage

    # Equity Curve Plot
    plt.figure(figsize=(12, 6))
    plt.plot(dates, equity_values, label='Equity Curve', linewidth=2)
    plt.xlabel("Date")
    plt.ylabel("Equity ($)")
    plt.title("Equity Curve Over Time")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    images['equity_curve'] = save_and_upload(buf.getvalue(), "equity_curve")
    plt.close()

    # Drawdown Curve Plot
    plt.figure(figsize=(12, 6))
    plt.plot(dates, drawdown_pct, label='Drawdown (%)',
             color='red', linewidth=2)
    plt.xlabel("Date")
    plt.ylabel("Drawdown (%)")
    plt.title("Drawdown Curve Over Time")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    images['drawdown_curve'] = save_and_upload(
        buf.getvalue(), "drawdown_curve")
    plt.close()

    # -------------------------------
    # 2. Trade Performance Histogram
    # -------------------------------
    trades_df = pd.DataFrame(results['_trades']).T
    print(trades_df)
    trades_df['ReturnPct'] = pd.to_numeric(trades_df['ReturnPct'])

    plt.figure(figsize=(8, 6))
    plt.hist(trades_df['ReturnPct'] * 100, bins=20,
             edgecolor='black')  # convert to percentage
    plt.xlabel("Trade Return (%)")
    plt.ylabel("Frequency")
    plt.title("Histogram of Trade Returns")
    plt.grid(True)
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    images['trade_return_histogram'] = save_and_upload(
        buf.getvalue(), "trade_return_histogram")
    plt.close()

    # -------------------------------
    # 3. Trade Duration vs. Trade Return Scatter Plot
    # -------------------------------
    trades_df['DurationDays'] = trades_df['Duration'] / \
        86400000  # Convert duration to days
    trades_df['ReturnPct'] = trades_df['ReturnPct'] * \
        100  # Convert to percentage

    plt.figure(figsize=(8, 6))
    plt.scatter(trades_df['DurationDays'],
                trades_df['ReturnPct'], color='green')
    plt.xlabel("Trade Duration (Days)")
    plt.ylabel("Trade Return (%)")
    plt.title("Trade Duration vs. Trade Return")
    plt.grid(True)
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    images['trade_duration_vs_return'] = save_and_upload(
        buf.getvalue(), "trade_duration_vs_return")
    plt.close()

    return images

# def upload_images_to_s3(images: dict, conversation_id: str) -> dict:
#     """
#     Uploads image bytes to an S3 bucket in a folder named after the conversation_id.

#     Parameters:
#         images (dict): Dictionary of image names → image bytes
#         conversation_id (str): Unique ID for the conversation (used as folder name)

#     Returns:
#         dict: image name → public S3 URL
#     """
#     uploaded_urls = {}

#     for name, image_bytes in images.items():
#         file_key = f"{conversation_id}/{name}.png"

#         # Upload to S3
#         s3.put_object(
#             Bucket=AppConfig.S3_BUCKET_NAME,
#             Key=file_key,
#             Body=image_bytes,
#             ContentType="image/png",
#             ACL='public-read'  # Allow public viewing
#         )

#         # Generate URL
#         s3_url = f"https://{AppConfig.S3_BUCKET_NAME}.s3.amazonaws.com/{file_key}"
#         uploaded_urls[name] = s3_url

#     return uploaded_urls

def answer_backtest_question(chat_message, chat_history, conversation_id):
    """
    Answer a question related to a backtest (e.g., details about trades, performance metrics).
    It retrieves the backtest results from the database and then uses LangChain to generate an answer.
    """
    # Query the database for backtest results matching the conversation_id.
    response = supabase.table("backtest_results").select(
        "*").eq("conversation_id", conversation_id).execute()
    
    print("response: ", response)

  
    results = response.data
    if not results:
        return f"No backtest results found for conversation ID {conversation_id}."

    # Use the first matching record (or pick the latest if multiple exist).
    backtest_result = results[0]

    # Extract and structure the backtest details.
    details = (
        f"Results: {backtest_result.get('results')}\n"
        f"Equity Curve: {backtest_result.get('equity_curve')}\n"
        f"Trades: {backtest_result.get('trades')}\n"
        f"Drawdown Curve: {backtest_result.get('drawdown_curve')}\n"
        f"Equity Curve Image: {backtest_result.get('equity_curve_image')}\n"
        f"Trade Return Histogram: {backtest_result.get('trade_return_histogram')}\n"
        f"Trade Duration vs Return: {backtest_result.get('trade_duration_vs_return')}\n"
    )

    # Define a prompt that includes the backtest details and asks the LLM to answer the user's question.
    prompt_template = """
    You are an expert in quantitative finance and trading strategy backtesting.
    Below are the detailed results of a backtest:
    {backtest_details}

    
    
    Based on the above information, please answer the following question:
    {chat_message}
    
    Provide a clear and concise answer that refers to performance metrics, trade details, or equity curve insights as appropriate.
    """

    prompt = PromptTemplate(
        input_variables=["backtest_details", "chat_message", "drawdown_curve", "equity_curve_image", "trade_return_histogram", "trade_duration_vs_return"],
        template=prompt_template
    )

    llm = ChatAnthropic(api_key=AppConfig.ANTHROPIC_API_KEY,
                        model="claude-3-5-haiku-20241022")
    chain = LLMChain(llm=llm, prompt=prompt)

    answer = chain.run(backtest_details=details, chat_message=chat_message)
    return answer



def chat_with_llm(chat_message, chat_history, conversation_id):
    """
    Handle a normal conversation using LangChain and Claude Anthropic.
    """
    prompt_template = """
    You are a friendly, knowledgeable chatbot with expertise in quantitative finance and backtesting.
    You are having a natural, engaging conversation with a user.

    Conversation ID: {conversation_id}
    Chat History: {chat_history}
    User: {chat_message}

    Please provide a helpful and clear response in plain text.
    """

    prompt = PromptTemplate(
        input_variables=["conversation_id", "chat_history", "chat_message"],
        template=prompt_template
    )

    llm = ChatAnthropic(api_key=AppConfig.ANTHROPIC_API_KEY, model="claude-3-5-haiku-20241022")
    chain = LLMChain(llm=llm, prompt=prompt)

    answer = chain.run(
        conversation_id=conversation_id,
        chat_history=chat_history,
        chat_message=chat_message
    )

    return answer

def store_image_in_s3(file_path, object_name=None):
    """
    Upload an image file to an S3 bucket using AWS credentials from environment variables, then return the public URL.

    Parameters:
      - file_path (str): Local path to the image file.
      - object_name (str, optional): S3 object name. If not specified, the file's base name is used.

    Returns:
      - str: The public URL to the uploaded image, or None if upload fails.
    """
    # Use the file's basename if object_name is not provided.
    if object_name is None:
        object_name = os.path.basename(file_path)
    
    # Retrieve credentials and bucket details from environment variables.
    aws_access_key_id = os.environ.get("AWS_ACCESS_KEY_ID")
    aws_secret_access_key = os.environ.get("AWS_SECRET_ACCESS_KEY")
    region_name = os.environ.get("AWS_REGION")
    bucket_name = os.environ.get("S3_BUCKET_NAME")

    if not all([aws_access_key_id, aws_secret_access_key, region_name, bucket_name]):
        print("One or more AWS configuration variables are missing.")
        return None
    
    # Create an S3 client using the credentials from the environment.
    s3_client = boto3.client(
        's3',
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
        region_name=region_name
    )
    
    try:
        # Upload the file to S3
        s3_client.upload_file(file_path, bucket_name, object_name)
    except NoCredentialsError:
        print("AWS credentials not available.")
        return None
    except ClientError as e:
        print(f"Failed to upload file: {e}")
        return None

    # Construct the public URL for the uploaded object.
    url = f"https://{bucket_name}.s3.amazonaws.com/{object_name}"
    return url

def generate_backtest_summary(backtest_results):
    """
    Generate a human-readable summary of backtest results using Claude.
    
    Parameters:
        backtest_results: The results object from the backtest
        
    Returns:
        str: A natural language summary of the backtest performance
    """
    prompt_template = """
    You are an expert in quantitative finance and trading strategy backtesting.
    Please provide a clear, concise summary of the following backtest results:

    {backtest_results}

    Include key metrics like:
    - Overall performance
    - Maximum drawdown
    - Number of trades
    - Win rate
    - Any notable patterns or observations

    Keep the summary professional but easy to understand.
    """

    prompt = PromptTemplate(
        input_variables=["backtest_results"],
        template=prompt_template
    )

    llm = ChatAnthropic(api_key=AppConfig.ANTHROPIC_API_KEY, model="claude-3-5-haiku-20241022")
    chain = LLMChain(llm=llm, prompt=prompt)

    return chain.run(backtest_results=json.loads(backtest_results.to_json()))


def stream_chat_with_llm(chat_message, chat_history, conversation_id):
    """
    Simulate a streaming response by chunking the final output.
    (Replace this with your LLM's native streaming if available.)
    """
    # Use your existing synchronous call to get the full answer.
    result = chat_with_llm(chat_message, chat_history, conversation_id)
    # For example, split into 20-character chunks.
    chunk_size = 20
    for i in range(0, len(result), chunk_size):
        yield result[i: i + chunk_size]

class StreamingCallbackHandler(BaseCallbackHandler):
    """
    A custom callback handler that uses a thread-safe queue to stream tokens.
    """
    def __init__(self):
        self.queue = queue.Queue()

    def on_llm_new_token(self, token: str, **kwargs):
        self.queue.put(token)

    def on_llm_end(self, response, **kwargs):
        # Signal the end of the stream by putting a None token.
        self.queue.put(None)

def stream_llm_response(prompt_template, input_vars, chain_kwargs):
    """
    Helper that creates an LLMChain with streaming enabled and yields tokens as they are generated.
    """
    handler = StreamingCallbackHandler()
    llm = ChatAnthropic(
        api_key=AppConfig.ANTHROPIC_API_KEY,
        model="claude-3-5-haiku-20241022",
        streaming=True,
        callbacks=[handler]
    )
    prompt = PromptTemplate(input_variables=list(input_vars.keys()), template=prompt_template)
    chain = LLMChain(llm=llm, prompt=prompt)
    
    def run_chain():
        chain.run(**chain_kwargs)
    thread = threading.Thread(target=run_chain)
    thread.start()
    
    while True:
        token = handler.queue.get()
        if token is None:
            break
        yield token
    thread.join()

@app.route('/chat', methods=['POST'])
def chat_endpoint():
    data = request.get_json()
    chat_message = data.get('chat_message')
    chat_history = data.get('chat_history', [])
    conversation_id = data.get('conversation_id', None)

    if not chat_message:
        return jsonify({"error": "No chat message provided"}), 400

    intent = analyze_intent(chat_message, chat_history)

    if intent == 'backtest':
        result, images = run_backtest_command(chat_message, chat_history, conversation_id)
        res = upload_backtest_result(result, images, conversation_id)
        if res:
            summary = generate_backtest_summary(result)
            response = {"type": "backtest", "result": summary}
        else:
            response = {"type": "error", "result": "Failed to upload backtest result"}
    elif intent == 'question':
        result = answer_backtest_question(chat_message, chat_history, conversation_id)
        response = {"type": "question", "result": result}
    else:
        # Default to a conversational reply
        result = chat_with_llm(chat_message, chat_history)
        response = {"type": "conversation", "result": result}

    return jsonify(response)

@app.route('/backtest-results/<conversation_id>', methods=['GET'])
def get_backtest_results(conversation_id):
    """
    Retrieve backtest results for a specific conversation ID.
    
    Parameters:
        conversation_id (str): The ID of the conversation to fetch results for
        
    Returns:
        JSON response containing the backtest results or an error message
    """
    try:
        # Query the database for backtest results
        response = supabase.table("backtest_results").select(
            "*").eq("conversation_id", conversation_id).execute()
        
        if not response.data:
            return jsonify({
                "error": f"No backtest results found for conversation ID {conversation_id}"
            }), 404
            
        # Return the most recent result if multiple exist
        return jsonify({
            "success": True,
            "data": response.data[0]
        })
        
    except Exception as e:
        return jsonify({
            "error": f"Failed to retrieve backtest results: {str(e)}"
        }), 500

@app.route('/chat/stream', methods=['POST'])
def chat_stream_endpoint():
    """New streaming endpoint that mirrors the functionality of /chat but streams the response"""
    data = request.get_json()
    chat_message = data.get('chat_message')
    chat_history = data.get('chat_history', [])
    conversation_id = data.get('conversation_id', None)

    if not chat_message:
        return jsonify({"error": "No chat message provided"}), 400

    intent = analyze_intent(chat_message, chat_history)

    if intent == 'backtest':
        result, images = run_backtest_command(chat_message, chat_history, conversation_id)
        res = upload_backtest_result(result, images, conversation_id)
        
        # Stream the backtest summary
        prompt_template = """
        You are an expert in quantitative finance and trading strategy backtesting.
        Please provide a clear, concise summary of the following backtest results:

        {backtest_results}

        Include key metrics like:
        - Overall performance
        - Maximum drawdown
        - Number of trades
        - Win rate
        - Any notable patterns or observations

        Keep the summary professional but easy to understand.
        """
        return Response(
            stream_llm_response(
                prompt_template,
                {"backtest_results": None},
                {"backtest_results": json.loads(result.to_json())}
            ),
            mimetype='text/plain'
        )
        
    elif intent == 'question':
        # Stream the answer to the backtest question
        response = supabase.table("backtest_results").select("*").eq("conversation_id", conversation_id).execute()
        if not response.data:
            return Response(f"No backtest results found for conversation ID {conversation_id}.", mimetype='text/plain')
            
        backtest_result = response.data[0]
        details = (
            f"Results: {backtest_result.get('results')}\n"
            f"Equity Curve: {backtest_result.get('equity_curve')}\n"
            f"Trades: {backtest_result.get('trades')}\n"
            f"Images: {json.dumps(backtest_result.get('images', {}))}\n"
        )
        
        prompt_template = """
        You are an expert in quantitative finance and trading strategy backtesting.
        Below are the detailed results of a backtest:
        {backtest_details}

        Based on the above information, please answer the following question:
        {chat_message}
        
        Provide a clear and concise answer that refers to performance metrics, trade details, or equity curve insights as appropriate.
        """
        return Response(
            stream_llm_response(
                prompt_template,
                {"backtest_details": None, "chat_message": None},
                {"backtest_details": details, "chat_message": chat_message}
            ),
            mimetype='text/plain'
        )
    else:
        # Stream the conversation response
        prompt_template = """
        You are a friendly, knowledgeable chatbot with expertise in quantitative finance and backtesting.
        You are having a natural, engaging conversation with a user.

        Conversation ID: {conversation_id}
        Chat History: {chat_history}
        User: {chat_message}

        Please provide a helpful and clear response in plain text.
        """
        return Response(
            stream_llm_response(
                prompt_template,
                {"conversation_id": None, "chat_history": None, "chat_message": None},
                {"conversation_id": conversation_id, "chat_history": chat_history, "chat_message": chat_message}
            ),
            mimetype='text/plain'
        )

if __name__ == '__main__':
    app.run(debug=True, port=8080)
