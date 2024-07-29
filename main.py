from flask import Flask, render_template, request, redirect, url_for, flash, session
from werkzeug.security import generate_password_hash, check_password_hash
import json
from flask_session import Session
import os
from langchain_openai import ChatOpenAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.chains import create_history_aware_retriever
from langchain_core.prompts import MessagesPlaceholder
from datetime import timedelta, datetime
import time

app = Flask(__name__)
app.config["SESSION_PERMANENT"] = False
app.config["SESSION_TYPE"] = "filesystem"
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(minutes=300)
os.environ[
"OPENAI_API_KEY"] = 'Yours'
Session(app)
app.secret_key = 'Yours'

# Load user data from JSON file
with open('Userdata/users.json', 'r') as file:
    users = json.load(file)

#session = {}


def updateWrite():
    with open("Userdata/users.json", "w") as f:
        f.write(json.dumps(users))


def updateRead():
    with open("Userdata/users.json", "r") as f:
        global users
        users = json.load(f)


store = {}

llm = ChatOpenAI(model="gpt-4o", temperature=1, max_tokens=1000)

file_path = r"Data/data.pdf"
loader = PyPDFLoader(file_path)
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000,
                                               chunk_overlap=200)
splits = text_splitter.split_documents(docs)
vectorstore = Chroma.from_documents(documents=splits,
                                    embedding=OpenAIEmbeddings())
retriever = vectorstore.as_retriever()

contextualize_q_system_prompt = (
    "Given a chat history and the latest user question "
    "which might reference context in the chat history, "
    "formulate a standalone question which can be understood "
    "without the chat history. Do NOT answer the question, "
    "just reformulate it if needed and otherwise return it as is.")

contextualize_q_prompt = ChatPromptTemplate.from_messages([
    ("system", contextualize_q_system_prompt),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])
history_aware_retriever = create_history_aware_retriever(
    llm, retriever, contextualize_q_prompt)

system_prompt = (
    "You are a tool for doctors to use to help diagnose migrain patients answer in full detail with explanations"
    "\n\n"
    "{context}")

qa_prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])

question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

rag_chain = create_retrieval_chain(history_aware_retriever,
                                   question_answer_chain)


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]


conversational_rag_chain = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer",
)


def get_answer(question: str, session_id: str) -> str:
    response = conversational_rag_chain.invoke(
        {"input": question},
        config={"configurable": {
            "session_id": str(session_id)
        }})["answer"]
    return response


@app.route("/")
def home():
    return render_template("home.html")

@app.route("/tools")
def tools():
    return render_template("tools.html")

@app.route("/help")
def help():
    return render_template("help.html")


@app.route("/chat", methods=["GET", "POST"])
def chat():
    if session.get("username"):
        if request.method == 'POST':
            query = request.form['query']
            return get_answer(query, session['username'])
        else:
            return render_template("chat.html")
    else:
        return redirect(url_for("login"))


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        remember = request.form.get('remember')

        print(f"Attempting login with username: {username}")
        print(f"Form Data: {request.form}")
        print(f"Current session before setting username: {session}")

        if username in users:
            print("Username found in users")
            if check_password_hash(users[username]['password'], password):
                print("Password is correct")
                session['username'] = username
                if remember:
                    session.permanent = True
                print(f"Session after setting username: {session}")
                return redirect(url_for('home'))
            else:
                print("Incorrect password")
                flash('Invalid username or password')
        else:
            print("Username not found")
            flash('Invalid username or password')

    print(f"Current session after login attempt: {session}")
    return render_template('login.html')


@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        confirm_password = request.form['confirm_password']

        if username in users:
            flash('Username already exists')
        elif password != confirm_password:
            flash('Passwords do not match')
        else:
            users[username] = {'password': generate_password_hash(password)}
            flash('Signup successful, please login')
            updateWrite()
            return redirect(url_for('login'))

    return render_template('signup.html')


@app.route("/logout")
def logout():
    session["username"] = None
    return redirect("/")


if __name__ == "__main__":
    updateRead()
    app.run()
    
