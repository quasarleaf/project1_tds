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


app = FastAPI()

DATA_DIR = "/data"  # Ensure operations are restricted to /data

# AI details
OPENAI_API_URL = "https://aiproxy.sanand.workers.dev/openai/v1/chat/completions"
OPENAI_API_KEY = os.environ["AIPROXY_TOKEN"]

headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}

def validate_path(file_path: str):
    """Ensure the file path is within /data"""
    full_path = os.path.abspath(os.path.join(DATA_DIR, file_path.lstrip("/")))
    if not full_path.startswith(DATA_DIR):
        raise HTTPException(status_code=403, detail="Access outside /data is forbidden")
    return full_path

def query_llm(task_description: str):
    """Queries ChatGPT API to interpret the task description."""
    payload = {
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "system", "content": "Extract structured intent from the given task description. Ensure the JSON follows this format:\n"
                "{ 'action': str, 'input_file': str, 'output_file': str, 'params': dict, 'filter_column':str,'filter_value':str,'filter_operator':<,>,<=,>=,==, 'url':str } Pick the actions from the following: email_extract, run_script,format_file, count_wednesdays, sort_json, query_database, extract_text_from_image,fetch_api_data,clone_git_repo,query_sql,scrape_website,compress_image,transcribe_audio,convert_md_to_html,filter_csv"},
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
        
        if action == "run_script":
            script_path = file_path if file_path else "data/b.py"
            subprocess.run(["python3", script_path], check=True)
            return "Script executed successfully."
        
        elif action == "format_file":
            file_path = file_path if file_path else "data/format.md"
            subprocess.run(["npx", "prettier", "--write", file_path], check=True)
            return "File formatted successfully."
        
        elif action == "count_wednesdays":
            with open(file_path, "r") as f:
                dates = [datetime.strptime(line.strip(), "%Y-%m-%d").weekday() for line in f]
            count = dates.count(2)  # Wednesday
            with open(output_path, "w") as f:
                f.write(str(count))
            return f"Wednesdays counted: {count}"
        
        elif action == "sort_json":
            with open(file_path, "r") as f:
                data = json.load(f)
            sorted_data = sorted(data, key=lambda x: (x["last_name"], x["first_name"]))
            with open('data/contacts-sorted', "w") as f:
                json.dump(sorted_data, f, indent=4)
            return "Contacts sorted successfully."
        
        elif action == "query_database":
            conn = sqlite3.connect(file_path)
            cursor = conn.cursor()
            cursor.execute("SELECT SUM(units * price) FROM tickets WHERE type = 'Gold'")
            total_sales = cursor.fetchone()[0]
            with open(output_path, "w") as f:
                f.write(str(total_sales))
            conn.close()
            return f"Total Gold ticket sales: {total_sales}"
        
        elif action == "email_extract":
            input_file = "data/email.txt"
            output_file = "data/email-sender.txt"

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

        elif action == "extract_text_from_image":
            file_path = file_path if file_path else "data/credit-card.png"
            text = pytesseract.image_to_string(Image.open(file_path))
            with open('data/credit-card.txt', "w") as f:
                f.write(text.strip().replace(" ", ""))
            return "Extracted text from image successfully."
        

        elif action == "fetch_api_data":
            response = requests.get(parsed_task["api_url"], params=parsed_task.get("params", {}))
            with open('data/api_output.txt', "w") as f:
                json.dump(response.json(), f, indent=4)
            return "API data saved."
        
        elif action == "clone_git_repo":
            subprocess.run(["git", "clone", parsed_task["repo_url"], output_path], check=True)
            return "Git repository cloned."
        
        elif action == "query_sql":
            db_path = validate_path(parsed_task["db_file"])
            query = parsed_task["query"]
            db_type = parsed_task.get("db_type", "sqlite")
            
            if db_type == "sqlite":
                conn = sqlite3.connect(db_path)
            
            df = pd.read_sql_query(query, conn)
            df.to_json(output_path, orient="records", indent=4)
            conn.close()
            return "SQL query executed."
        
        elif action == "scrape_website":
            from bs4 import BeautifulSoup
            response = requests.get(parsed_task["url"])
            soup = BeautifulSoup(response.text, "html.parser")
            with open('data/scrape_output.txt', "w") as f:
                f.write(soup.prettify())
            return "Website scraped."
        
        elif action == "compress_image":
            img = Image.open(file_path)
            img.save('data/compressed_image', "JPEG", quality=parsed_task.get("quality", 50))
            return "Image compressed."
        
        elif action == "transcribe_audio":
            try:
                # Load recognizer and audio file
                recognizer = sr.Recognizer()
                audio_file = file_path if file_path else "data/audio.mp3"

                # Ensure the input file exists
                if not os.path.exists(audio_file):
                    raise HTTPException(status_code=404, detail=f"Audio file '{audio_file}' not found")

                # Convert MP3 to WAV
                wav_output_path = "data/output.wav"
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
                output_text_file = "data/audio_transcribed.txt"
                with open(output_text_file, "w") as f:
                    f.write(text)

                return {"status": "success", "message": "Transcription completed", "output_file": output_text_file}

            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Error during transcription: {str(e)}")


        
        elif action == "convert_md_to_html":
            file_path = file_path if file_path else "data/markdown_file.md"
            with open(file_path, "r", encoding="utf-8") as f:
                md_content = f.read()
            html_content = markdown.markdown(md_content)
            with open('data/converted_markdown.html', "w", encoding="utf-8") as f:
                f.write(html_content)
            return "Markdown converted to HTML."
        
        elif action == "filter_csv":
            file_path = file_path if file_path else "data/abcd.csv"
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
            output_path = 'data/filtered_output.json'

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
        file_path = path.lstrip("/")  # Remove leading slash
        if not os.path.isfile(file_path):
            raise HTTPException(status_code=404, detail="File not found")
        
        with open(file_path, "r", encoding="utf-8") as file:
            content = file.read()
        
        return {"status": "success", "content": content}
    
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
