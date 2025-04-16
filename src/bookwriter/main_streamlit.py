import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")

import streamlit as st
from agents import BookWriterAgents
from tasks import BookWriterTasks
from crewai import Crew , Process ,LLM
from dotenv import load_dotenv
import os
import sys
from io import StringIO
from strip_ansi import strip_ansi


# Custom class to redirect stdout to both Streamlit and terminal
class StreamlitStream(StringIO):
    def __init__(self):
        super().__init__()
        self.buffer = []

    def write(self, text):
        # Strip ANSI codes and append to buffer
        clean_text = strip_ansi(text)
        self.buffer.append(clean_text)
        # Write to terminal
        sys.__stdout__.write(text)
        sys.__stdout__.flush()
        # Update Streamlit UI
        if hasattr(st.session_state, 'log_container'):
            # Join buffer lines and preserve formatting
            formatted_log = "".join(self.buffer)
            # Replace bold ANSI with Markdown bold
            formatted_log = formatted_log.replace("Crew Execution Started", "**Crew Execution Started**")
            formatted_log = formatted_log.replace("Crew Manager", "**Crew Manager**")
            formatted_log = formatted_log.replace("Content Strategist", "**Content Strategist**")
            formatted_log = formatted_log.replace("Writer", "**Writer**")
            formatted_log = formatted_log.replace("Task Completed", "**Task Completed**")
            # st.session_state.log_container.markdown(formatted_log, unsafe_allow_html=True)
            st.session_state.log_container.markdown(
                f'<div id="log-output" style="max-height: 400px; overflow-y: auto; font-family: \'Courier New\', Courier, monospace; white-space: pre; font-size: 14px;">{formatted_log}</div>'
                '<script>var logDiv = document.getElementById("log-output"); logDiv.scrollBottom = logDiv.scrollHeight;</script>',
                unsafe_allow_html=True
            )

    def flush(self):
        sys.__stdout__.flush()
        pass

# Load environment variables
load_dotenv()
api_key=os.getenv("GEMINI_API_KEY")
model = LLM(model="gemini/gemini-2.0-flash-exp" ,api_key=api_key)

# Streamlit UI
st.title("Book Writer AI")
st.write("Enter the details below to generate a book draft.")

# Input form
with st.form("book_writer_form"):
    word_count = st.text_input("Word Count (e.g., 2000)", value="2000")
    Book_Title = st.text_input("Book Title", value="The Era of Artificial Intelligence")
    Author_Name = st.text_input("Author Name", value="John Doe")
    Target_Audience = st.text_input("Target Audience", value="Beginners")
    Writing_Style = st.text_input("Writing Style", value="Conversational")
    submit_button = st.form_submit_button("Generate Book Draft")

# word_count = 2000
# Book_Title = "The era of artificial intelligence"
# Writing_Style = "Conversational"
# Author_Name = "John Doe"
# Target_Audience = "Beginners"

# Placeholder for logs
log_placeholder = st.empty()

# Set monospace font style to mimic terminal
log_placeholder.markdown("""
<style>
pre, code, .stMarkdown {
    font-family: 'Courier New', Courier, monospace;
    white-space: pre;
    font-size: 14px;
}
</style>
""", unsafe_allow_html=True)
st.session_state.log_container = log_placeholder

# # Function to update progress
# def update_progress(task_name: str, status: str, agent: str = None):
#     message = f"Task: {task_name[:50]}... | Status: {status}"
#     if agent:
#         message += f" | Agent: {agent}"
#     progress_placeholder.write(message)

# Function to save output to Markdown
def save_to_markdown(task_output, filename: str = "output.md") -> None:
    text = task_output.result if hasattr(task_output, 'result') else str(task_output)
    with open(filename, "w", encoding="utf-8") as file:
        file.write(text)
    st.session_state.output_text = text
    st.session_state.output_file = filename
    print(f"📝 Saved book draft to {filename}")


# Process form submission
if submit_button:
    try:
        word_count = int(word_count)
        if word_count <= 0:
            st.error("Word count must be a positive number.")
            st.stop()

        with st.spinner("Processing... Please wait"):
            # Redirect stdout to Streamlit
            original_stdout = sys.stdout
            sys.stdout = StreamlitStream()

            # Initialize agents and tasks
            print("🚀 Initializing crew...")
            agents = BookWriterAgents()
            tasks = BookWriterTasks()

            # Agents
            print("🤖 Creating agents...")
            Content_Strategist = agents.Content_Strategist()
            Writer = agents.Writer()

            # Tasks
            print("📋 Setting up Content Strategist Task...")
            Content_Strategist_Task = tasks.Content_Strategist_Task(
                agent=Content_Strategist,
                word_count=word_count,
                Book_Title=Book_Title,
                Author_Name=Author_Name,
                Target_Audience=Target_Audience,
                Writing_Style=Writing_Style,
            )

            print("📋 Setting up Writer Task...")
            Writer_Task = tasks.Writer_Task(
                agent=Writer,
                context=[Content_Strategist_Task],
                callback=save_to_markdown,
            )

            # Crew
            print("⚙️ Configuring Crew...")
            crew = Crew(
                agents=[Content_Strategist, Writer],
                tasks=[Content_Strategist_Task, Writer_Task],
                verbose=True,
                process=Process.hierarchical,
                manager_llm=model,
            )

            # Execute
            print("🏃 Starting book generation...")
            crew.kickoff()
            print("✅ Book generation complete!")

            # Restore stdout
            sys.stdout = original_stdout

        # Display results
        if "output_text" in st.session_state:
            st.success(f"Book draft saved to {st.session_state.output_file}")
            st.write("### Book Draft Preview")
            st.write(st.session_state.output_text[:1000] + "..." if len(st.session_state.output_text) > 1000 else st.session_state.output_text)
            with open(st.session_state.output_file, "r", encoding="utf-8") as file:
                st.download_button(
                    label="Download Book Draft",
                    data=file,
                    file_name="book_draft.md",
                    mime="text/markdown"
                )
        else:
            st.warning("No output generated. Check the logs above for errors.")

    except ValueError:
        st.error("Invalid word count. Please enter a number (e.g., 2000).")
    except Exception as e:
        st.error(f"An error occurred: {e}")
        if hasattr(st.session_state, 'log_container'):
            st.session_state.log_container.text(f"Error: {e}")
    finally:
        # Ensure stdout is restored
        if 'original_stdout' in locals():
            sys.stdout = original_stdout