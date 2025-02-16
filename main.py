from fastapi import FastAPI, Query, HTTPException
import requests
import os
import subprocess
import json
import sqlite3
from datetime import datetime
import pytesseract
from PIL import Image
import re
import markdown
import pandas as pd
import librosa
import soundfile as sf
import numpy as np 
import speech_recognition as sr
from pydub import AudioSegment
import uvicorn
import glob
import wget
import sys 

app = FastAPI()

DATA_DIR = "/data"  # Ensure operations are restricted to /data

# AI details
OPENAI_API_URL = "https://aiproxy.sanand.workers.dev/openai/v1/chat/completions"
OPENAI_API_KEY = os.environ["AIPROXY_TOKEN"]

headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}

def validate_path(file_path: str):
    """Ensure the file path is within /data"""
    full_path = os.path.abspath(os.path.join(DATA_DIR, file_path))
    if not full_path.startswith(DATA_DIR):
        raise HTTPException(status_code=403, detail="Access outside /data is forbidden")
    return full_path

def is_uv_installed():
    try:
        subprocess.run(["uv", "--version"], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except FileNotFoundError:
        return False

def parse_date(date_str):
    """Try parsing a date string with multiple formats."""
    date_formats = ["%Y-%m-%d", "%d-%m-%Y", "%d/%m/%Y", "%d %b %Y", "%d-%b-%Y", "%B %d, %Y"]
    
    for fmt in date_formats:
        try:
            return datetime.strptime(date_str.strip(), fmt)
        except ValueError:
            continue
    return None  # Return None if no format matches

def get_first_line(file_path):
    """Reads the first line of a file safely."""
    with open(file_path, "r") as f:
        return f.readline().strip()

def extract_first_h1(file_path):
    """Extracts the first occurrence of an H1 (line starting with #) from a Markdown file."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line.startswith("# "):  # H1 detected
                    return line[2:].strip()  # Remove "# " and return the title
    except Exception as e:
        return f"Error reading {file_path}: {e}"
    
    return "Untitled"  # Default if no H1 found

def query_llm(task_description: str):
    """Queries ChatGPT API to interpret the task description."""
    payload = {
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "system", "content": "Extract structured intent from the given task description. Ensure the JSON follows this format:\n"
                "{ 'action': str, 'input_file': str, 'output_file': str, 'params': dict, 'filter_column':str,'filter_value':str,'filter_operator':<,>,<=,>=,==, 'url':str,'user_email':str } Pick the actions from the following: generate_data,doc_titles,recent_logs,email_extract,format_file, count_wednesdays, sort_json, query_database, extract_text_from_image,fetch_api_data,clone_git_repo,scrape_website,compress_image,transcribe_audio,convert_md_to_html,filter_csv"},
            {"role": "user", "content": task_description}
        ],
        "temperature": 0.2
    }
    response = requests.post(OPENAI_API_URL, headers=headers, json=payload)
    if response.status_code == 200:
        raw_content = response.json()["choices"][0]["message"]["content"]
        # Remove markdown-style JSON formatting 
        cleaned_content = re.sub(r"```json\n(.*?)\n```", r"\1", raw_content, flags=re.DOTALL).strip()

        # Ensure proper JSON formatting (convert single quotes to double quotes)
        cleaned_content = cleaned_content.replace("'", "\"")

        try:
            parsed_response = json.loads(cleaned_content)  # Parse only if it's valid JSON

            if not parsed_response.get("output_file"):
                parsed_response["output_file"] = "formatted_" + parsed_response.get("input_file", "output.md")

            return parsed_response
        except json.JSONDecodeError:
            raise HTTPException(status_code=500, detail=f"Invalid JSON response from LLM after cleanup: {cleaned_content}")

    else:
        raise HTTPException(status_code=500, detail="LLM API error")

def execute_task(parsed_task):
    """Executes a task based on parsed intent."""
    try:
        keys = ["action", "intent", "task"]

        # Find the first existing key in parsed_task
        action = next((parsed_task[key] for key in keys if key in parsed_task), None)

        # Ensure all keys have the same value
        if action is not None:
            for key in keys:
                parsed_task[key] = action
        file_path=''
        output_path=''

        if "input_file" in parsed_task:
            if(parsed_task["input_file"]!=""):
                file_path = validate_path(parsed_task["input_file"]).lstrip("/")
        if "output_file" in parsed_task:
            if(parsed_task["input_file"]!=""):
                output_path = validate_path(parsed_task["output_file"]).lstrip("/")
        

        # Chooses the action to perform

        # Phase A

        # A1: Generate Data

        if action == "generate_data":
            if not is_uv_installed():
                print("Installing 'uv'...")
                subprocess.run([sys.executable, "-m", "pip", "install", "uv"], check=True)

            # Define the URL of the script
            script_url = "https://raw.githubusercontent.com/sanand0/tools-in-data-science-public/tds-2025-01/project-1/datagen.py"

            # Define the user email (change this if needed)
            user_email = parsed_task["user_email"] 
            
            # Download the script
            script_path = "datagen.py"
            print(f"Downloading {script_path} from {script_url}...")
            if os.path.exists(script_path):
                os.remove(script_path)  # Remove existing file to avoid (1) duplication
            wget.download(script_url, script_path)
            print("\nDownload complete.")

            # Run the script with the user email as the argument
            print("\nRunning datagen.py...")
            subprocess.run(["uv", "run", script_path, user_email], check=True)

            print("Data generation completed successfully!")

        # A2: Format format.md 

        elif action == "format_file":
            file_path = "/data/format.md"
            subprocess.run(["npx", "prettier", "--write", file_path], check=True)
            return "File formatted successfully."
        
        # A3: Counts wednesdays in dates.txt 

        elif action == "count_wednesdays":
            file_path = file_path if file_path else "/data/dates.txt"
            with open(file_path, "r") as f:
                dates = [parse_date(line) for line in f]

            # Filter out None values and count Wednesdays (weekday() == 2)
            count = sum(1 for date in dates if date and date.weekday() == 2)

            # Save the count to a file
            with open('/data/dates-wednesdays.txt', "w") as f:
                f.write(str(count))

            return f"Wednesdays counted: {count}"
        
        # A4: Sorts contacts.json 

        elif action == "sort_json":
            file_path = "/data/contacts.json"
            with open(file_path, "r") as f:
                data = json.load(f)
            sorted_data = sorted(data, key=lambda x: (x["last_name"], x["first_name"]))
            with open('/data/contacts-sorted.json', "w") as f:
                json.dump(sorted_data, f, indent=4)
            return "Contacts sorted successfully."
        
        # A5: Write frist line of 10 most recent log files in logs-recent.txt 

        elif action == "recent_logs":
            
            log_dir = "/data/logs/"
            output_file = "/data/logs-recent.txt"

            # Find all .log files in /data/logs/, sorted by modification time (most recent first)
            log_files = sorted(
                glob.glob(os.path.join(log_dir, "*.log")),
                key=os.path.getmtime,
                reverse=True  # Most recent first
            )[:10]  # Take only the 10 most recent logs

            # Read first lines from the selected logs
            first_lines = [get_first_line(log) for log in log_files]

            # Write to output file
            with open(output_file, "w") as f:
                f.write("\n".join(first_lines))

            return "Recent log lines written successfully."

        # A6: Writes the headings and titles of files in docs to index.json

        elif action == "doc_titles":

            docs_dir = "/data/docs/"
            output_file = "/data/docs/index.json"

            # Ensure directory exists
            if not os.path.exists(docs_dir):
                return f"Error: Directory '{docs_dir}' does not exist."

            # Find all Markdown (.md) files in /data/docs/
            md_files = glob.glob(os.path.join(docs_dir, "**/*.md"), recursive=True)

            if not md_files:
                return "Error: No Markdown files found in '/data/docs/'."

            # Create an index mapping filenames (without full path) to their extracted H1 titles
            index = {}
            for file_path in md_files:
                file_name = os.path.relpath(file_path, docs_dir)  # Remove '/data/docs/' prefix
                index[file_name] = extract_first_h1(file_path)

            # Write to JSON file
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(index, f, indent=4)

            return f"Markdown index created successfully at '{output_file}'."

        # A7: Extracts sender email

        elif action == "email_extract":
            input_file = "/data/email.txt"
            output_file = "/data/email-sender.txt"

            # Read email content
            try:
                with open(input_file, "r", encoding="utf-8") as f:
                    email_content = f.read()
            except FileNotFoundError:
                raise HTTPException(status_code=404, detail="email.txt not found")

            # Query LLM to extract sender's email
            task_description = "Extract only the sender's email address from the following email content:\n" + email_content
            payload = {
                "model": "gpt-4o-mini",
                "messages": [
                    {"role": "system", "content": "Extract only the sender's email address from the given email content."},
                    {"role": "user", "content": email_content}
                ],
                "temperature": 0
            }
            response = requests.post(OPENAI_API_URL, headers=headers, json=payload)

            if response.status_code != 200:
                raise HTTPException(status_code=500, detail="LLM API error")

            sender_email = response.json()["choices"][0]["message"]["content"].strip()

            # Validate extracted email
            email_pattern = r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"
            match = re.search(email_pattern, sender_email)
            if not match:
                raise HTTPException(status_code=500, detail="Invalid email extracted")

            # Save to file
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(match.group(0))

            return {"status": "success", "sender_email": match.group(0)}

        # A8: Extracts number from credit card image

        elif action == "extract_text_from_image":
            file_path = "/data/credit_card.png"
            text = pytesseract.image_to_string(Image.open(file_path))
            text = text.strip().replace(" ", "")
            text = text.split()[0] if text else ""
            with open('/data/credit-card.txt', "w") as f:
                f.write(text)
            return "Extracted text from image successfully."

        # A10: Writes total sales of Gold ticket type items to ticket-sales-gold.txt

        elif action == "query_database":
            file_path = "/data/ticket-sales.db"
            output_path = "/data/ticket-sales-gold.txt"
            conn = sqlite3.connect(file_path)
            cursor = conn.cursor()
            cursor.execute("SELECT SUM(units * price) FROM tickets WHERE type = 'Gold'")
            
            total_sales = cursor.fetchone()[0]
            with open(output_path, "w") as f:
                f.write(str(total_sales))
            conn.close()
            return f"Total Gold ticket sales: {total_sales}"
        
        # Phase B

        # B3: Fetch API data

        elif action == "fetch_api_data":
            response = requests.get(parsed_task["url"], params=parsed_task.get("params", {}))
            print("API response: ",response.text)  # Print response
            with open('/data/api_output.txt', "w", encoding="utf-8") as f:
                f.write(response.text)  # Save as plain text instead of JSON
            return "API data fetched and saved to /data/api_output.txt"

        # B4: Clones Git repo
        
        elif action == "clone_git_repo":
            output_path = '/data/repo'
            subprocess.run(["git", "clone", parsed_task["url"], output_path], check=True)
            return "Git repository cloned."

        # B6: Scrapes website
        
        elif action == "scrape_website":
            from bs4 import BeautifulSoup
            response = requests.get(parsed_task["url"])
            soup = BeautifulSoup(response.text, "html.parser")
            with open('/data/scrape_output.txt', "w") as f:
                f.write(soup.prettify())
            return "Website scraped."
        
        # B7: Compresses image

        elif action == "compress_image":
            img = Image.open(file_path)
            img.save('/data/compressed_image', "JPEG", quality=parsed_task.get("quality", 50))
            return "Image compressed."
        
        # B8: Transcribes audio

        elif action == "transcribe_audio":
            try:
                # Load recognizer and audio file
                recognizer = sr.Recognizer()
                audio_file = file_path

                # Ensure the input file exists
                if not os.path.exists(audio_file):
                    raise HTTPException(status_code=404, detail=f"Audio file '{audio_file}' not found")

                # Convert MP3 to WAV
                wav_output_path = "/data/output.wav"
                audio = AudioSegment.from_mp3(audio_file)
                audio.export(wav_output_path, format="wav")

                # Ensure file was converted
                if not os.path.exists(wav_output_path):
                    raise HTTPException(status_code=500, detail="WAV conversion failed")

                # Transcribe audio
                with sr.AudioFile(wav_output_path) as source:
                    audio_data = recognizer.record(source)

                # Recognize speech using Google Speech Recognition
                try:
                    text = recognizer.recognize_google(audio_data)
                except sr.UnknownValueError:
                    raise HTTPException(status_code=500, detail="Google Speech Recognition could not understand the audio")
                except sr.RequestError:
                    raise HTTPException(status_code=500, detail="Could not request results from Google Speech Recognition")

                # Save transcription
                output_text_file = "/data/audio_transcribed.txt"
                with open(output_text_file, "w") as f:
                    f.write(text)

                return {"status": "success", "message": "Transcription completed", "output_file": output_text_file}

            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Error during transcription: {str(e)}")

        # B9: Converts markdown files to html
        
        elif action == "convert_md_to_html":
            with open(file_path, "r", encoding="utf-8") as f:
                md_content = f.read()
            html_content = markdown.markdown(md_content)
            with open('/data/converted_markdown.html', "w", encoding="utf-8") as f:
                f.write(html_content)
            return "Markdown converted to HTML."
        
        # B10: Filters csv file

        elif action == "filter_csv":
            # Read CSV file
            df = pd.read_csv(file_path)

            # Get filter parameters
            filter_column = parsed_task.get("filter_column")
            filter_value = parsed_task.get("filter_value")
            filter_operator = parsed_task.get("filter_operator", "==")  # Default to equality

            # Validate column existence
            if filter_column not in df.columns:
                raise HTTPException(status_code=400, detail=f"Column '{filter_column}' not found in CSV file.")

            # Convert filter column to numeric if possible
            if df[filter_column].dtype == object:
                df[filter_column] = pd.to_numeric(df[filter_column], errors="coerce")

            # Convert filter value to numeric
            filter_value = pd.to_numeric(filter_value, errors="coerce")

            # Apply filtering based on operator
            if filter_operator == ">":
                filtered_df = df[df[filter_column] > filter_value]
            elif filter_operator == "<":
                filtered_df = df[df[filter_column] < filter_value]
            elif filter_operator == ">=":
                filtered_df = df[df[filter_column] >= filter_value]
            elif filter_operator == "<=":
                filtered_df = df[df[filter_column] <= filter_value]
            elif filter_operator == "!=":
                filtered_df = df[df[filter_column] != filter_value]
            else:  # Default: equality check
                filtered_df = df[df[filter_column] == filter_value]

            # Debugging output
            print(f"Filtering {len(df)} rows → {len(filtered_df)} rows where {filter_column} {filter_operator} {filter_value}")

            # Validate output path
            output_path = '/data/filtered_output.json'

            # Save to JSON
            filtered_df.to_json(output_path, orient="records", indent=4)
            
            return f"CSV filtered and saved to {output_path}."
        else:
            raise HTTPException(status_code=400, detail="Unknown task action.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/run")
def run_task(task: str = Query(..., description="Plain-English task description")):
    """Executes a task using LLM and internal logic."""
    try:
        if not task.strip():
            raise HTTPException(status_code=400, detail="Task description cannot be empty")
        
        # Use LLM to parse task
        parsed_task = query_llm(f"Extract structured intent from: {task}")
        # Execute the mapped task
        result = execute_task(parsed_task)
        return {"status": "success", "result": result}
    
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/read")
def read_file(path: str = Query(..., description="File path to read")):
    """Returns the content of the specified file."""
    try:
        file_path = path
        if not os.path.isfile(file_path):
            raise HTTPException(status_code=404, detail="File not found")
        
        with open(file_path, "r", encoding="utf-8") as file:
            content = file.read()
        
        return {"status": "success", "content": content}
    
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
